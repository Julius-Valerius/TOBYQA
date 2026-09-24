"""
analyze_alphasweep.py
=====================
曲率收缩扫描的分析：逐维度给出各 alpha 的平均秩（按同一实例上最终 best_f 排名，
秩越小越好）与 Moré–Wild 求解率，并报告每个维度的最优 alpha。

用法:
    python analyze_alphasweep.py --db results/alphasweep.sqlite
"""
import argparse
import collections

import numpy as np

from run_benchmark import load_runs, compute_solve

ALPHA = {'ours': 0.0, 'ours-h25': 0.25, 'ours-h50': 0.50,
         'ours-h75': 0.75, 'ours-quadH': 1.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='results/alphasweep.sqlite')
    ap.add_argument('--tau', type=float, default=1e-3)
    a = ap.parse_args()

    runs = load_runs(a.db)
    dims = sorted({r['dim'] for r in runs})
    order = [s for s in ALPHA if any(r['solver'] == s for r in runs)]
    solved, _, _ = compute_solve(runs, [a.tau])

    best = collections.defaultdict(dict)
    for r in runs:
        best[(r['problem'], r['dim'], r['env'], r['seed'])][r['solver']] = r['best_f']

    print(f'曲率收缩扫描  (tau={a.tau:.0e})，平均秩越小越好\n')
    hdr = ' '.join(f'a={ALPHA[s]:<5.2f}' for s in order)
    print(f"{'n':>4}  {hdr}   最优alpha(秩)  最优alpha(求解率)")
    print('-' * (8 + 8 * len(order) + 34))

    for d in dims:
        ranks = {s: [] for s in order}
        for k, dd in best.items():
            if k[1] != d:
                continue
            vals = [(s, dd.get(s)) for s in order]
            if any(v is None for _, v in vals):
                continue
            arr = np.array([v for _, v in vals], float)
            rk = np.argsort(np.argsort(arr)) + 1.0
            # 并列取平均秩
            for val in set(arr):
                idx = arr == val
                if idx.sum() > 1:
                    rk[idx] = rk[idx].mean()
            for (s, _), r_ in zip(vals, rk):
                ranks[s].append(r_)
        mr = {s: float(np.mean(ranks[s])) if ranks[s] else float('nan') for s in order}

        rate = {}
        for s in order:
            hs = [h for (k, sv, t), h in solved.items() if sv == s and k[1] == d]
            rate[s] = 100.0 * np.mean([h is not None for h in hs]) if hs else float('nan')

        bs_rank = min(order, key=lambda s: mr[s])
        bs_rate = max(order, key=lambda s: rate[s])
        print(f'{d:4d}  ' + ' '.join(f'{mr[s]:6.3f}' for s in order) +
              f'   a={ALPHA[bs_rank]:.2f}         a={ALPHA[bs_rate]:.2f} '
              f'({rate[bs_rate]:.1f}%)')

    print('\n各维度求解率明细 (%)')
    print(f"{'n':>4}  " + ' '.join(f'a={ALPHA[s]:<5.2f}' for s in order))
    for d in dims:
        row = []
        for s in order:
            hs = [h for (k, sv, t), h in solved.items() if sv == s and k[1] == d]
            row.append(100.0 * np.mean([h is not None for h in hs]) if hs else float('nan'))
        print(f'{d:4d}  ' + ' '.join(f'{v:6.1f}' for v in row))


if __name__ == '__main__':
    main()
