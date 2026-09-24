"""
analyze_dimsweep.py
===================
维度扫描的配对分析：逐维度比较 ours（零 Hessian）与 ours-quadH（插值 Hessian）。

两个口径：
  * Moré–Wild 求解率 + McNemar 配对检验（依赖参考值 f_L）；
  * 最终 best_f 的直接配对比较（不依赖 f_L 与容差）。
并给出 quadH 胜率随维度的 Spearman 秩相关，用于判断趋势是否单调。

用法:
    python analyze_dimsweep.py --db results/dimsweep.sqlite
"""
import argparse
import collections
from math import comb

import numpy as np

from run_benchmark import load_runs, compute_solve


def two_sided_sign_p(a, b):
    n = a + b
    if n == 0:
        return 1.0
    return min(sum(comb(n, i) for i in range(min(a, b) + 1)) * 2 / 2 ** n, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='results/dimsweep.sqlite')
    ap.add_argument('--tau', type=float, default=1e-3)
    a = ap.parse_args()

    runs = load_runs(a.db)
    dims = sorted({r['dim'] for r in runs})
    budgets = {d: max(r['n_evals'] for r in runs if r['dim'] == d) for d in dims}
    solved, _, _ = compute_solve(runs, [a.tau])

    best = collections.defaultdict(dict)
    for r in runs:
        best[(r['problem'], r['dim'], r['env'], r['seed'])][r['solver']] = r['best_f']

    print(f'维度扫描配对分析  (tau={a.tau:.0e})')
    print(f"{'n':>4} {'预算':>6} {'每变量':>7} {'ours':>7} {'quadH':>7} "
          f"{'ours+':>6} {'quadH+':>7} {'p':>10} | {'best_f 配对':>14} {'p':>10}")
    print('-' * 104)

    frac = []
    for d in dims:
        A, B = {}, {}
        for (k, s, t), h in solved.items():
            if k[1] != d:
                continue
            if s == 'ours':
                A[k] = h is not None
            elif s == 'ours-quadH':
                B[k] = h is not None
        ks = set(A) & set(B)
        aw = sum(1 for k in ks if A[k] and not B[k])
        bw = sum(1 for k in ks if B[k] and not A[k])
        ra = 100 * sum(A[k] for k in ks) / len(ks) if ks else float('nan')
        rb = 100 * sum(B[k] for k in ks) / len(ks) if ks else float('nan')

        fa = fb = 0
        for k, dd in best.items():
            if k[1] != d:
                continue
            x, y = dd.get('ours'), dd.get('ours-quadH')
            if x is None or y is None:
                continue
            if x < y - 1e-12 * max(1, abs(y)):
                fa += 1
            elif y < x - 1e-12 * max(1, abs(y)):
                fb += 1

        frac.append(bw / (aw + bw) if aw + bw else 0.5)
        print(f'{d:4d} {budgets[d]:6d} {budgets[d]/d:7.0f} {ra:6.1f}% {rb:6.1f}% '
              f'{aw:6d} {bw:7d} {two_sided_sign_p(aw, bw):10.2e} | '
              f'{fa:5d}:{fb:<6d} {two_sided_sign_p(fa, fb):10.2e}')

    if len(dims) >= 3:
        r = np.corrcoef(np.argsort(np.argsort(dims)),
                        np.argsort(np.argsort(frac)))[0, 1]
        print(f'\nquadH 配对胜率 vs 维度的 Spearman 秩相关: rho = {r:+.3f} '
              f'({len(dims)} 个维度点)')
        print('胜率序列:', ' '.join(f'n={d}:{f:.2f}' for d, f in zip(dims, frac)))


if __name__ == '__main__':
    main()
