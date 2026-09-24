"""
run_dimsweep.py
===============
维度扫描：对每个 n，以固定的每变量求值次数（budget = evals_per_var * n）
运行 ours 与 ours-quadH 的配对比较，结果写入同一个库。
库主键含 dim，各维度互不覆盖。

预设:
    quick  3 环境 × 3 种子 × 100 次/变量   —— 诊断趋势用，约为 full 的 1/6 成本
    full   7 环境 × 5 种子 × 150 次/变量   —— 完整统计功效

用法:
    python run_dimsweep.py --preset quick --workers 12
    python run_dimsweep.py --preset full  --workers 12 --dims 8 10 12
"""
import argparse
import os
import subprocess
import sys

DIMS = [4, 6, 8, 10, 12, 16, 20, 28, 40]
SOLVERS = ['ours', 'ours-quadH']
PRESETS = {
    'quick': dict(envs=['Static', 'LinearDrift', 'ChaoticDrift'],
                  seeds=['0', '1', '2'], evals_per_var=100),
    'full':  dict(envs=None, seeds=['0', '1', '2', '3', '4'],
                  evals_per_var=150),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=None)
    ap.add_argument('--preset', choices=sorted(PRESETS), default='quick')
    ap.add_argument('--workers', type=int,
                    default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--dims', type=int, nargs='*', default=DIMS)
    a = ap.parse_args()

    cfg = PRESETS[a.preset]
    db = a.db or os.path.join('results', f'dimsweep_{a.preset}.sqlite')
    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(here, 'run_benchmark.py')
    os.makedirs(os.path.dirname(os.path.abspath(db)), exist_ok=True)

    for n in a.dims:
        budget = cfg['evals_per_var'] * n
        print(f"\n===== n={n}, budget={budget} "
              f"({cfg['evals_per_var']}/变量) =====", flush=True)
        cmd = [sys.executable, script, '--db', db, '--dims', str(n),
               '--budget', str(budget), '--solvers', *SOLVERS,
               '--seeds', *cfg['seeds'], '--workers', str(a.workers)]
        if cfg['envs']:
            cmd += ['--envs', *cfg['envs']]
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f'[warn] n={n} 退出码 {rc}')

    print(f'\n完成。出结果： python analyze_dimsweep.py --db {db}')


if __name__ == '__main__':
    main()
