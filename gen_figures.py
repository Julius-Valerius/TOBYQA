#!/usr/bin/env python3
"""
gen_figures.py — generate benchmark figures from a results database,
in the unified style of plot_style.py.

Usage:  python gen_figures.py --db results.sqlite --out figures
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import apply_style
import matplotlib.pyplot as plt
from run_benchmark import load_runs, compute_solve, TAUS

# Solver display names, colours and linestyles.
DISPLAY = {
    'ours':       ('TOBYQA',        '#003153', '-'),
    'newuoa':     ('NEWUOA',        '#A23838', '--'),
    'bobyqa':     ('BOBYQA',        '#2D5A4C', '-.'),
    'pybobyqa':   ('Py-BOBYQA',     '#7B9CBB', ':'),
    'dfols':      ('DFO-LS',        '#7B9CBB', ':'),
    'neldermead': ('Nelder-Mead',   '#C5951D', (0, (3, 1, 1, 1))),
    'bfgs-fd':    ('BFGS',          '#5C4033', (0, (5, 2))),
}
ORDER = ['ours', 'newuoa', 'bobyqa', 'pybobyqa', 'dfols', 'neldermead', 'bfgs-fd']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='results/full65.sqlite',
                    help='Path to benchmark SQLite database (default: results/full65.sqlite)')
    ap.add_argument('--out', default='Final_paper/figures',
                    help='Output directory for generated figures (default: Final_paper/figures)')
    ap.add_argument('--budget', type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    apply_style()

    runs = load_runs(a.db)
    solved, f0s, fLs = compute_solve(runs, TAUS)
    present = [s for s in ORDER
               if any(r['solver'] == s and r['best_f'] is not None for r in runs)]
    if 'pybobyqa' in present and 'dfols' in present:
        present.remove('dfols')
    budget = a.budget or max(r['n_evals'] for r in runs)

    # ---- Figure 1: data profiles at tau = 1e-3 -------------------------
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    grid = np.arange(1, budget + 1)
    for sv in present:
        hits = [hit for (key, s, t), hit in solved.items()
                if s == sv and t == 1e-3]
        if not hits:
            continue
        arr = np.array([h if h is not None else np.inf for h in hits])
        frac = [(arr <= b).mean() * 100.0 for b in grid]
        nm, col, ls = DISPLAY[sv]
        ax.plot(grid, frac, color=col, linestyle=ls, linewidth=1.9, label=nm)
    ax.set_xlabel('function evaluations')
    ax.set_ylabel('instances solved (%)')
    ax.set_xlim(0, budget)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left', fontsize=9)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(a.out, f'data_profiles_tau1e3.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ---- Figure 2: solve rate versus tolerance -------------------------
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for sv in present:
        ys = []
        for t in TAUS:
            hs = [hit for (key, s, tt), hit in solved.items()
                  if s == sv and tt == t]
            ys.append(100.0 * np.mean([h is not None for h in hs]) if hs
                      else np.nan)
        nm, col, ls = DISPLAY[sv]
        ax.plot(range(len(TAUS)), ys, color=col, linestyle=ls,
                linewidth=1.9, marker='o', markersize=3.5, label=nm)
    ax.set_xticks(range(len(TAUS)))
    ax.set_xticklabels([r'$10^{%d}$' % int(np.log10(t)) for t in TAUS])
    ax.set_xlabel(r'tolerance $\tau$')
    ax.set_ylabel('instances solved (%)')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', fontsize=9)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(a.out, f'solve_rate_vs_tau.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('figures written to', a.out)


if __name__ == '__main__':
    main()
