"""
run_alphasweep.py
=================
曲率收缩系数扫描：对每个维度 n，比较 H <- alpha*H 中
alpha ∈ {0, 0.25, 0.5, 0.75, 1} 的表现（alpha=0 即 ours，alpha=1 即 ours-quadH）。
每个 n 的预算 = EVALS_PER_VAR * n，使每变量求值次数在各维度上一致。

用法:
    python run_alphasweep.py --workers 12
"""
import argparse
import os
import subprocess
import sys

DIMS = [6, 10, 20, 40]
EVALS_PER_VAR = 100
SOLVERS = ['ours', 'ours-h25', 'ours-h50', 'ours-h75', 'ours-quadH']
ENVS = ['Static', 'LinearDrift', 'ChaoticDrift']
SEEDS = ['0', '1', '2']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=os.path.join('results', 'alphasweep.sqlite'))
    ap.add_argument('--workers', type=int,
                    default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--dims', type=int, nargs='*', default=DIMS)
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(here, 'run_benchmark.py')
    os.makedirs(os.path.dirname(os.path.abspath(a.db)), exist_ok=True)

    for n in a.dims:
        budget = EVALS_PER_VAR * n
        print(f'\n===== n={n}, budget={budget} =====', flush=True)
        cmd = [sys.executable, script, '--db', a.db, '--dims', str(n),
               '--budget', str(budget), '--solvers', *SOLVERS,
               '--envs', *ENVS, '--seeds', *SEEDS, '--workers', str(a.workers)]
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f'[warn] n={n} 退出码 {rc}')

    print(f'\n完成。出结果： python analyze_alphasweep.py --db {a.db}')


if __name__ == '__main__':
    main()
