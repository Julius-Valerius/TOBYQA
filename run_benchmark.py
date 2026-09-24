#!/usr/bin/env python3
"""
run_benchmark.py
===================
TOBYQA 基准评测：问题 × 维度 × 时变环境 × 种子 × 求解器。

设计要点
--------
* **SQLite (WAL) 实时落盘**：每个实例跑完立刻写库；中断后重跑自动跳过已完成项。
* **多核并行**：``--workers``（默认 cpu_count-1）。每个 worker 独立写库，
  WAL 模式 + busy_timeout 处理并发。
* **分片**：``--shard-id / --num-shards`` 把作业按稳定哈希切片，
  可在多台机器/多个 SLURM 任务里并行跑同一个库（或各写各的库后 ``--merge``）。
* **紧凑存储**：不存完整轨迹，只存 incumbent 的**改进点序列**
  ``[(eval_index, best_f_noisefree), ...]``。这足以在事后对任意 τ 计算
  求解次数与数据剖面，且体积通常 < 完整轨迹的 5%。
* **两遍评分**：f_L（Moré–Wild 的参考最优值）需要跨求解器取最好值，
  故求解率在 ``--analyze`` 阶段计算，与运行解耦。

用法
----
    # 0) 自检（10 个实例，确认环境就绪）
    python run_benchmark.py --smoke

    # 1) 单机全量
    python run_benchmark.py --db results.sqlite --workers 32

    # 2) 分 8 片（8 台机器 / 8 个作业），各写各的库
    python run_benchmark.py --db shard0.sqlite --num-shards 8 --shard-id 0 --workers 16
    ...
    python run_benchmark.py --merge shard*.sqlite --db results.sqlite

    # 3) 出表（LaTeX）
    python run_benchmark.py --db results.sqlite --analyze --latex-out tables/

    # 4) 使用 CUTEst 问题集
    python run_benchmark.py --db results.sqlite --problem-source pycutest

依赖
----
必需: numpy, scipy
可选: pdfo (NEWUOA/BOBYQA), dfols (DFO-LS), pycutest (真 CUTEst 问题)
     缺失的基线会被自动跳过并在日志中提示。

求解器模块（TOBYQA.py / ARC_subproblem.py / KKT_posterior.py）
需可被 import：放在同目录，或用 --src-dir 指定。
"""

from __future__ import annotations
import argparse, glob, hashlib, json, os, signal, sqlite3, sys, time, traceback
import multiprocessing as mp
from contextlib import closing

# 每个 worker 进程限用单线程 BLAS：并行由 multiprocessing 提供，
# 若再让 BLAS 自行开线程会在核数有限时相互抢占。必须在 import numpy 前设置。
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np

# ---------------------------------------------------------------- 配置默认值
DEFAULT_DIMS     = [6, 10, 20]
DEFAULT_SEEDS    = [0, 1, 2, 3, 4]
DEFAULT_BUDGET   = 1500
DEFAULT_NOISEREL = 1e-3          # sigma_eps = noise_rel * max(|f(x0)|, 1)
TAUS             = [1e-1, 1e-3, 1e-5, 1e-7]

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS runs (
    problem   TEXT NOT NULL,
    dim       INTEGER NOT NULL,
    env       TEXT NOT NULL,
    seed      INTEGER NOT NULL,
    solver    TEXT NOT NULL,
    f0        REAL,
    scale     REAL,
    noise_std REAL,
    n_evals   INTEGER,
    early     INTEGER,
    best_f    REAL,
    improves  TEXT,       -- JSON: [[eval_idx, best_f_noisefree], ...]
    wall_s    REAL,
    error     TEXT,
    ts        REAL,
    PRIMARY KEY (problem, dim, env, seed, solver)
);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
"""


# ---------------------------------------------------------------- 数据库工具
def db_connect(path, timeout=120.0):
    con = sqlite3.connect(path, timeout=timeout, isolation_level=None)
    con.execute('PRAGMA busy_timeout=120000;')
    return con


def db_init(path, meta=None):
    with closing(db_connect(path)) as con:
        con.executescript(SCHEMA)
        if meta:
            for k, v in meta.items():
                con.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',
                            (k, json.dumps(v, ensure_ascii=False)))
    return path


def db_done_keys(path):
    if not os.path.exists(path):
        return set()
    with closing(db_connect(path)) as con:
        try:
            rows = con.execute(
                'SELECT problem,dim,env,seed,solver FROM runs').fetchall()
        except sqlite3.OperationalError:
            return set()
    return set(rows)


def db_write(path, rec):
    with closing(db_connect(path)) as con:
        con.execute(
            'INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
            (rec['problem'], rec['dim'], rec['env'], rec['seed'], rec['solver'],
             rec['f0'], rec['scale'], rec['noise_std'], rec['n_evals'],
             int(rec['early']), rec['best_f'], json.dumps(rec['improves']),
             rec['wall_s'], rec['error'], time.time()))


def db_merge(sources, target):
    db_init(target)
    total = 0
    with closing(db_connect(target)) as con:
        for src in sources:
            if os.path.abspath(src) == os.path.abspath(target):
                continue
            con.execute("ATTACH DATABASE ? AS s", (src,))
            cur = con.execute('INSERT OR REPLACE INTO runs SELECT * FROM s.runs')
            total += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
            con.execute('DETACH DATABASE s')
    print(f'[merge] 合并 {len(sources)} 个库 -> {target}')
    return total


# ---------------------------------------------------------------- 作业构建
def build_jobs(problems, dims, envs, seeds, solvers):
    return [(p, d, e, s, v)
            for p in problems for d in dims for e in envs
            for s in seeds for v in solvers]


def shard_of(job, num_shards):
    h = hashlib.blake2b('|'.join(map(str, job)).encode(), digest_size=8).digest()
    return int.from_bytes(h, 'big') % num_shards


# ---------------------------------------------------------------- 单实例执行
_G = {}


def _worker_init(src_dir, source, budget, noise_rel):
    if src_dir and src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    np.seterr(all='ignore')
    import bench_problems, bench_solvers
    _G.update(P=bench_problems, S=bench_solvers, source=source,
              budget=budget, noise_rel=noise_rel)
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def run_one(job):
    problem, dim, env, seed, solver = job
    P, S = _G['P'], _G['S']
    t0 = time.time()
    rec = dict(problem=problem, dim=dim, env=env, seed=seed, solver=solver,
               f0=None, scale=None, noise_std=None, n_evals=0, early=False,
               best_f=None, improves=[], wall_s=0.0, error=None)
    try:
        f, x0 = P.get_problem(problem, dim, source=_G['source'])
        f0 = f(x0)
        oracle, noise_std, scale = P.make_oracle(
            f, x0, env, seed, _G['noise_rel'])
        rec.update(f0=float(f0) if np.isfinite(f0) else None,
                   scale=float(scale), noise_std=float(noise_std))

        out = S.run_solver(solver, oracle, x0, dim, noise_std, _G['budget'])
        rec.update(n_evals=out['n_evals'], early=out['terminated_early'],
                   error=out['error'])

        # 在无噪声静态目标上重新评分
        best = None
        improves = []
        for i, x in enumerate(out['xs'], start=1):
            v = f(x)
            if not np.isfinite(v):
                continue
            if best is None or v < best - 1e-15*max(1.0, abs(best)):
                best = float(v)
                improves.append([i, best])
        if best is None:
            best = np.inf
        rec['best_f'] = None if not np.isfinite(best) else float(best)
        rec['improves'] = improves
    except Exception as e:                                       # noqa: BLE001
        rec['error'] = f'{type(e).__name__}: {e}'[:300]
        rec['traceback'] = traceback.format_exc()[-500:]
    rec['wall_s'] = time.time() - t0
    return rec


# ---------------------------------------------------------------- 分析
def load_runs(path):
    with closing(db_connect(path)) as con:
        rows = con.execute(
            'SELECT problem,dim,env,seed,solver,f0,n_evals,early,best_f,improves,error'
            ' FROM runs').fetchall()
    out = []
    for r in rows:
        out.append(dict(problem=r[0], dim=r[1], env=r[2], seed=r[3], solver=r[4],
                        f0=r[5], n_evals=r[6], early=bool(r[7]), best_f=r[8],
                        improves=json.loads(r[9]) if r[9] else [], error=r[10]))
    return out


def compute_solve(runs, taus=TAUS):
    """
    按 Moré–Wild 判据计算求解情况。
    f_L = 同一 (problem,dim,env,seed) 下所有求解器在预算内达到的最好值。
    返回 solved[(key,solver,tau)] = 首次达标的求值次数 or None。
    """
    from collections import defaultdict
    by_inst = defaultdict(dict)
    for r in runs:
        by_inst[(r['problem'], r['dim'], r['env'], r['seed'])][r['solver']] = r

    solved, f0s, fLs = {}, {}, {}
    for key, d in by_inst.items():
        vals = [v['best_f'] for v in d.values() if v['best_f'] is not None]
        if not vals:
            continue
        fL = min(vals)
        f0 = next((v['f0'] for v in d.values() if v['f0'] is not None), None)
        if f0 is None or not np.isfinite(f0) or f0 <= fL:
            continue
        f0s[key], fLs[key] = f0, fL
        for sv, r in d.items():
            for tau in taus:
                thr = fL + tau*(f0 - fL)
                hit = None
                for idx, bf in r['improves']:
                    if bf <= thr:
                        hit = idx
                        break
                solved[(key, sv, tau)] = hit
    return solved, f0s, fLs


def _rate(solved, solver, tau, dim=None, env=None):
    ok = tot = 0
    for (key, sv, t), hit in solved.items():
        if sv != solver or t != tau:
            continue
        if dim is not None and key[1] != dim:
            continue
        if env is not None and key[2] != env:
            continue
        tot += 1
        ok += (hit is not None)
    return (100.0*ok/tot if tot else float('nan')), tot


def _median_nf(solved, solver, tau, dim=None, env=None):
    v = [hit for (key, sv, t), hit in solved.items()
         if sv == solver and t == tau and hit is not None
         and (dim is None or key[1] == dim) and (env is None or key[2] == env)]
    return float(np.median(v)) if v else float('nan')


SOLVER_LABEL = {
    'ours':             r'\textbf{TOBYQA}',
    'ours-quadH':       r'\quad + interpolated $H$',
    'ours-nodrift':     r'\quad -- time column',
    'ours-purelinear':  r'\quad -- curvature kernel $A$',
    'ours-hessupd':     r'\quad + Powell update',
    'ours-nosigma':     r'\quad frozen $\sigma$',
    'newuoa':           r'NEWUOA',
    'bobyqa':           r'BOBYQA',
    'pybobyqa':         r'Py-BOBYQA',
    'dfols':            r'DFO-LS',
    'neldermead':       r'Nelder--Mead',
    'bfgs-fd':          r'BFGS (FD)',
}


def latex_main_table(solved, solvers, dims, taus=(1e-1, 1e-3, 1e-5, 1e-7),
                     budget=DEFAULT_BUDGET):
    L = []
    A = L.append
    ncols = len(dims) * len(taus)
    # With many data columns the per-cell median would overflow the page
    # width; keep the median only for narrow tables and tighten the column
    # separation otherwise.
    keep_median = ncols <= 8
    A(r'\begin{table}[t]')
    A(r'\centering' + (r'\small' if keep_median else r'\footnotesize'))
    if not keep_median:
        A(r'\setlength{\tabcolsep}{3pt}')
    if keep_median:
        A(r'\caption{Solve rates (\%) with median number of function evaluations '
          r'among solved instances in parentheses. Budget: ' + str(budget)
          + r' evaluations.}')
    else:
        A(r'\caption{Solve rates (\%). Budget: ' + str(budget)
          + r' evaluations per instance.}')
    A(r'\label{tab:solve-rates}')
    A(r'\begin{tabular}{l' + 'r'*ncols + '}')
    A(r'\toprule')
    A('Solver & ' + ' & '.join(
        r'\multicolumn{' + str(len(dims)) + r'}{c}{$\tau=10^{'
        + str(int(np.log10(t))) + r'}$}' for t in taus) + r' \\')
    A(''.join(r'\cmidrule(lr){' + str(2 + i*len(dims)) + '-'
              + str(1 + (i+1)*len(dims)) + '}' for i in range(len(taus))))
    A(' & ' + ' & '.join('$n{=}' + str(d) + '$' for _ in taus for d in dims) + r' \\')
    A(r'\midrule')
    for sv in solvers:
        cells = []
        for t in taus:
            for d in dims:
                r, _ = _rate(solved, sv, t, dim=d)
                m = _median_nf(solved, sv, t, dim=d)
                if np.isnan(r):
                    cells.append('--')
                elif np.isnan(m) or not keep_median:
                    cells.append('{:.1f}'.format(r))
                else:
                    cells.append('{:.1f}'.format(r) + r'\,(' + str(int(m)) + ')')
        A(SOLVER_LABEL.get(sv, sv) + ' & ' + ' & '.join(cells) + r' \\')
    A(r'\bottomrule')
    A(r'\end{tabular}')
    A(r'\end{table}')
    return '\n'.join(L)


def latex_env_table(solved, solvers, envs, tau=1e-3, label='tab:env-breakdown',
                    caption=None):
    if caption is None:
        caption = (r'Solve rates (\%) at $\tau=10^{' + str(int(np.log10(tau)))
                   + r'}$ broken down by drift regime, pooled over all $n$ and seeds.')
    L = []
    A = L.append
    A(r'\begin{table}[t]')
    A(r'\centering\footnotesize')
    A(r'\setlength{\tabcolsep}{3pt}')
    A(r'\caption{' + caption + '}')
    A(r'\label{' + label + '}')
    A(r'\begin{tabular}{l' + 'r'*len(envs) + '}')
    A(r'\toprule')
    A('Solver & ' + ' & '.join(e.replace('Drift', '') for e in envs) + r' \\')
    A(r'\midrule')
    for sv in solvers:
        cells = []
        for e in envs:
            r, _ = _rate(solved, sv, tau, env=e)
            cells.append('--' if np.isnan(r) else '{:.1f}'.format(r))
        A(SOLVER_LABEL.get(sv, sv) + ' & ' + ' & '.join(cells) + r' \\')
    A(r'\bottomrule')
    A(r'\end{tabular}')
    A(r'\end{table}')
    return '\n'.join(L)


def latex_budget_table(runs, solvers, budget=DEFAULT_BUDGET):
    """TR 早停的直接证据表：实际消耗求值 vs 预算。"""
    from collections import defaultdict
    ev = defaultdict(list); early = defaultdict(int); tot = defaultdict(int)
    for r in runs:
        ev[r['solver']].append(r['n_evals'])
        tot[r['solver']] += 1
        early[r['solver']] += int(r['early'])
    L = []; A = L.append
    A(r'\begin{table}[t]\centering\small')
    A(r'\caption{Budget utilisation. A solver that stops before the budget is '
      r'exhausted has declared convergence; under noise this is the dominant '
      r'failure mode of heuristic stopping rules.}')
    A(r'\label{tab:budget}')
    A(r'\begin{tabular}{lrrr}\toprule')
    A(r'Solver & median evals used & \% of budget & \% runs stopping early \\\midrule')
    for sv in solvers:
        if not ev[sv]:
            continue
        m = float(np.median(ev[sv]))
        A('{} & {} & {:.1f} & {:.1f}'.format(
            SOLVER_LABEL.get(sv, sv), int(m), 100.0*m/budget,
            100.0*early[sv]/max(tot[sv], 1)) + r' \\')
    A(r'\bottomrule\end{tabular}\end{table}')
    return '\n'.join(L)


def data_profile_csv(solved, solvers, dims, tau=1e-3):
    """输出数据剖面所需的 (solver, alpha, fraction) 三元组，alpha = evals/(n+1)。"""
    from collections import defaultdict
    pts = defaultdict(list)
    for (key, sv, t), hit in solved.items():
        if t != tau:
            continue
        n = key[1]
        pts[sv].append(np.inf if hit is None else hit/(n + 1.0))
    lines = ['solver,alpha,fraction']
    grid = np.concatenate([[0], np.logspace(-1, 3, 200)])
    for sv in solvers:
        v = np.array(pts[sv], float)
        if v.size == 0:
            continue
        for a in grid:
            lines.append('{},{:.6g},{:.6f}'.format(sv, a, float(np.mean(v <= a))))
    return '\n'.join(lines)


def do_analyze(args):
    runs = load_runs(args.db)
    if not runs:
        print('[analyze] 库为空'); return
    # 预算以库 meta 为准（analyze 可能没带 --budget）
    try:
        with closing(db_connect(args.db)) as con:
            row = con.execute("SELECT v FROM meta WHERE k='budget'").fetchone()
        if row:
            args.budget = int(json.loads(row[0]))
    except Exception:
        pass
    present = [s for s in (args.solvers or sorted({r['solver'] for r in runs}))
               if any(r['solver'] == s for r in runs)]
    solved, f0s, fLs = compute_solve(runs, TAUS)

    errs = [r for r in runs if r['error']]
    print(f'[analyze] 记录 {len(runs)} 条，实例 {len(f0s)} 个，求解器 {len(present)} 个，'
          f'报错 {len(errs)} 条')
    if errs:
        from collections import Counter
        for k, v in Counter(e['error'].split(':')[0] for e in errs).most_common(5):
            print(f'          {k}: {v}')

    print('\n求解率 (%)：')
    hdr = f'{"solver":<26}' + ''.join(f'{"tau=1e"+str(int(np.log10(t))):>12}' for t in TAUS)
    print(hdr); print('-'*len(hdr))
    for sv in present:
        row = ''.join(f'{_rate(solved, sv, t)[0]:>11.1f}%' for t in TAUS)
        print(f'{sv:<26}{row}')

    envs = sorted({r['env'] for r in runs})
    print(f'\n分环境 (tau=1e-3)：')
    hdr = f'{"solver":<26}' + ''.join(f'{e[:12]:>14}' for e in envs)
    print(hdr); print('-'*len(hdr))
    for sv in present:
        print(f'{sv:<26}' + ''.join(
            f'{_rate(solved, sv, 1e-3, env=e)[0]:>13.1f}%' for e in envs))

    print(f'\n预算利用率：')
    for sv in present:
        v = [r['n_evals'] for r in runs if r['solver'] == sv]
        e = [r['early'] for r in runs if r['solver'] == sv]
        if v:
            print(f'  {sv:<24} 中位 {int(np.median(v)):>5} / {args.budget}'
                  f'   提前停机 {100.0*sum(e)/len(e):5.1f}%')

    if args.latex_out:
        os.makedirs(args.latex_out, exist_ok=True)
        dims = sorted({r['dim'] for r in runs})
        main_solvers = [s for s in ['ours', 'newuoa', 'bobyqa', 'pybobyqa',
                                    'dfols', 'neldermead', 'bfgs-fd'] if s in present]
        if 'pybobyqa' in main_solvers and 'dfols' in main_solvers:
            main_solvers.remove('dfols')
        abl_solvers = [s for s in ['ours', 'ours-quadH', 'ours-nodrift',
                                   'ours-purelinear', 'ours-hessupd',
                                   'ours-nosigma'] if s in present]
        w = lambda fn, s: open(os.path.join(args.latex_out, fn), 'w').write(s)
        w('table_solve_rates.tex', latex_main_table(solved, main_solvers, dims,
                                                     budget=args.budget))
        w('table_env_breakdown.tex', latex_env_table(solved, main_solvers, envs))
        w('table_ablation.tex', latex_env_table(solved, abl_solvers, envs,
                                                label='tab:ablation',
                                                caption=r'Ablation of TOBYQA components: solve rates (\%) at $\tau=10^{-3}$ by drift regime, pooled over all $n$ and seeds.'))
        w('table_budget.tex', latex_budget_table(runs, main_solvers, args.budget))
        w('data_profile.csv', data_profile_csv(solved, main_solvers, dims))
        print(f'\n[analyze] LaTeX 表与数据剖面已写入 {args.latex_out}/')


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(
        description='TOBYQA 基准评测',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--db', default='results_o2.sqlite')
    ap.add_argument('--src-dir', default='.', help='求解器源码目录')
    ap.add_argument('--problem-source', choices=['builtin', 'pycutest'],
                    default='builtin')
    ap.add_argument('--problems', nargs='*', default=None, help='默认使用全部')
    ap.add_argument('--dims', type=int, nargs='*', default=DEFAULT_DIMS)
    ap.add_argument('--envs', nargs='*', default=None)
    ap.add_argument('--seeds', type=int, nargs='*', default=DEFAULT_SEEDS)
    ap.add_argument('--solvers', nargs='*', default=None)
    ap.add_argument('--budget', type=int, default=DEFAULT_BUDGET)
    ap.add_argument('--noise-rel', type=float, default=DEFAULT_NOISEREL)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--shard-id', type=int, default=0)
    ap.add_argument('--chunksize', type=int, default=1)
    ap.add_argument('--timeout', type=float, default=0,
                    help='单实例墙钟上限（秒），0 表示不限')
    ap.add_argument('--analyze', action='store_true')
    ap.add_argument('--latex-out', default=None)
    ap.add_argument('--merge', nargs='*', default=None)
    ap.add_argument('--smoke', action='store_true', help='10 实例自检')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.src_dir))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import bench_problems, bench_solvers

    if args.merge:
        srcs = [p for pat in args.merge for p in glob.glob(pat)]
        db_merge(srcs, args.db); return

    if args.analyze:
        do_analyze(args); return

    envs = args.envs or bench_problems.ENVIRONMENTS
    problems = args.problems or bench_problems.list_problems(args.problem_source)
    solvers = args.solvers or bench_solvers.ALL_SOLVERS

    if args.smoke:
        problems = problems[:2]; args.dims = [6]; envs = envs[:2]
        args.seeds = [0]; args.budget = min(args.budget, 200)
        solvers = ['ours', 'ours-nodrift']
        print('[smoke] 自检模式')

    # 可选依赖体检
    for name, mod in [('newuoa/bobyqa', 'pdfo'), ('dfols', 'dfols'),
                      ('pycutest', 'pycutest')]:
        try:
            __import__(mod)
        except ImportError:
            if mod == 'pdfo' and any(s in solvers for s in ('newuoa', 'bobyqa')):
                print(f'[warn] 未安装 {mod}，NEWUOA/BOBYQA 将全部记为 error')
            elif mod == 'dfols' and 'dfols' in solvers:
                print(f'[warn] 未安装 {mod}，DFO-LS 将全部记为 error')
            elif mod == 'pycutest' and args.problem_source == 'pycutest':
                print(f'[error] --problem-source pycutest 需要安装 {mod}'); sys.exit(2)

    jobs = build_jobs(problems, args.dims, envs, args.seeds, solvers)
    if args.num_shards > 1:
        jobs = [j for j in jobs if shard_of(j, args.num_shards) == args.shard_id]

    db_init(args.db, meta=dict(
        budget=args.budget, noise_rel=args.noise_rel, dims=args.dims,
        envs=envs, seeds=args.seeds, solvers=solvers,
        problem_source=args.problem_source, n_problems=len(problems),
        env_params=bench_problems.ENV_PARAMS, taus=TAUS))

    done = db_done_keys(args.db)
    todo = [j for j in jobs if j not in done]
    print(f'[plan] 问题 {len(problems)} × 维度 {len(args.dims)} × 环境 {len(envs)} '
          f'× 种子 {len(args.seeds)} × 求解器 {len(solvers)}')
    print(f'[plan] 本分片作业 {len(jobs)}，已完成 {len(jobs)-len(todo)}，待跑 {len(todo)}')
    if args.dry_run or not todo:
        return

    t0 = time.time()
    # Windows/macOS 没有(或不宜用) fork; 仅 Linux 用 fork.
    ctx = mp.get_context('fork' if sys.platform.startswith('linux') else 'spawn')
    with ctx.Pool(args.workers, initializer=_worker_init,
                  initargs=(os.path.abspath(args.src_dir), args.problem_source,
                            args.budget, args.noise_rel)) as pool:
        it = pool.imap_unordered(run_one, todo, chunksize=args.chunksize)
        n_err = 0
        for i, rec in enumerate(it, 1):
            db_write(args.db, rec)
            n_err += bool(rec['error'])
            if i % 50 == 0 or i == len(todo):
                el = time.time() - t0
                rate = i/max(el, 1e-9)
                eta = (len(todo) - i)/max(rate, 1e-9)
                print(f'[{i}/{len(todo)}] {rate:.1f} it/s  已用 {el/60:.1f}min  '
                      f'ETA {eta/60:.1f}min  错误 {n_err}', flush=True)
    print(f'[done] 用时 {(time.time()-t0)/60:.1f} min，错误 {n_err} 条')
    print(f'[next] 出表：python {os.path.basename(__file__)} --db {args.db} '
          f'--analyze --latex-out tables/')


if __name__ == '__main__':
    main()
