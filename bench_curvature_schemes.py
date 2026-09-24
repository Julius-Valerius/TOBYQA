"""
bench_curvature_schemes.py
==========================
End-to-end comparison of curvature recovery schemes inside the same
pipeline (Appendix A.4): 6 problems x dims {6,10} x 3 drift regimes x
2 seeds = 72 instances per variant, 600 evaluations each.

Variants: ARC-default (interpolated H), ARC-zeroH, ARC-diagLS,
ARC-hessupd (incremental update), zeroH-nodrift, BFGS-fd.

Usage:
    python bench_curvature_schemes.py            # run (checkpointed), then report
    python bench_curvature_schemes.py --report   # report from results/curvature_schemes.pkl
"""
import collections
import os
import pickle
import sys
import time
from math import comb

import numpy as np

import TOBYQA
import ARC_subproblem
import KKT_posterior


# ---------------------------------------------------------------- problems
def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

def arwhead(x):
    return np.sum((x[:-1] ** 2 + x[-1] ** 2) ** 2 - 4 * x[:-1] + 3)

def bdqrtic(x):
    n = len(x); s = 0.0
    for i in range(n - 4):
        s += (-4 * x[i] + 3) ** 2 + (x[i] ** 2 + 2 * x[i + 1] ** 2
             + 3 * x[i + 2] ** 2 + 4 * x[i + 3] ** 2 + 5 * x[n - 1] ** 2) ** 2
    return s

def engval1(x):
    return np.sum((x[:-1] ** 2 + x[1:] ** 2) ** 2) + np.sum(-4 * x[:-1] + 3)

def sinquad(x):
    return ((x[0] - 1) ** 4
            + np.sum((np.sin(x[1:-1] - x[-1]) - x[0] ** 2 + x[1:-1] ** 2) ** 2)
            + (x[-1] ** 2 - x[0] ** 2) ** 2)

def chnrosnb(x):
    return np.sum(100 * (x[:-1] - x[1:] ** 2) ** 2 + (1 - x[1:]) ** 2)

PROBS = dict(rosenbrock=rosenbrock, arwhead=arwhead, bdqrtic=bdqrtic,
             engval1=engval1, sinquad=sinquad, chnrosnb=chnrosnb)

NOISE = 0.01
ENVS = ['Static', 'LinearDrift', 'ChaoticDrift']
DIMS = [6, 10]
SEEDS = [0, 1]
TAUS = [1e-1, 1e-3, 1e-5]
BUDGET = 600
WH = ['ARC-default', 'ARC-zeroH', 'ARC-diagLS', 'ARC-hessupd',
      'zeroH-nodrift', 'BFGS-fd']
RESULT = os.path.join('results', 'curvature_schemes.pkl')
CKPT = os.path.join('results', 'curvature_schemes_ckpt.pkl')


def make_obj(f, env, rng, f0ref):
    beta = 0.01 * f0ref
    def obj(x, t):
        v = f(x)
        if env == 'LinearDrift':
            v = v + beta * t
        elif env == 'ChaoticDrift':
            v = v + beta * np.sin(0.7 * t) * np.cos(0.31 * t) * 3
        return v + rng.normal(0, NOISE)
    return obj


# ------------------------------------------------ variant hooks
_ob = ARC_subproblem.build_hessian_from_multipliers
_okkt = KKT_posterior.solve_kkt_with_posterior
_store = {}

def _setH(f):
    ARC_subproblem.build_hessian_from_multipliers = f
    TOBYQA.ARC_subproblem.build_hessian_from_multipliers = f

def _setK(f):
    KKT_posterior.solve_kkt_with_posterior = f
    TOBYQA.KKT_posterior.solve_kkt_with_posterior = f

def _diagLS(lam, dX):
    # Diagonal least-squares fit of the residuals r = F - c - dX @ g
    # against A = 0.5 * dX**2.
    F = _store.get('F'); g = _store.get('g')
    if F is None or g is None:
        return _ob(lam, dX)
    A = 0.5 * dX ** 2
    c0 = float(np.median(F - dX @ g))
    r = F - c0 - dX @ g
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(r))):
        return _ob(lam, dX)
    try:
        h, *_ = np.linalg.lstsq(A, r, rcond=1e-10)
    except np.linalg.LinAlgError:
        return _ob(lam, dX)
    return np.diag(h) if np.all(np.isfinite(h)) else _ob(lam, dX)

def _hookStore(X, F, T, xk, tn, rho, noise_std=None):
    r = _okkt(X, F, T, xk, tn, rho, noise_std=noise_std)
    _store['F'] = np.asarray(F, float)
    _store['g'] = np.asarray(r[1], float)
    return r

def _hookNoDrift(X, F, T, xk, tn, rho, noise_std=None):
    # gamma := 0 (removes the effect of the time column).
    r = _okkt(X, F, T, xk, tn, rho, noise_std=noise_std)
    return (r[0], r[1], 0.0) + tuple(r[3:])


def run(prob, env, dim, seed, which):
    f = PROBS[prob]
    rng = np.random.default_rng(10000 * seed + 7)
    x0 = np.linspace(-1.5, 1.2, dim)
    obj = make_obj(f, env, rng, abs(f(x0)) + 1.0)
    _store.clear()
    try:
        if which == 'BFGS-fd':
            from scipy.optimize import minimize
            cnt = [0]; hist = []
            def fw(x):
                if cnt[0] >= BUDGET:
                    raise StopIteration
                cnt[0] += 1
                hist.append(f(np.asarray(x)))
                return obj(np.asarray(x), float(cnt[0]))
            try:
                minimize(fw, x0, method='BFGS',
                         options={'maxiter': BUDGET, 'eps': 1e-4})
            except StopIteration:
                pass
            return float(np.min(hist)) if hist else None
        else:
            kw = dict(obj_func=obj, n_vars=dim, noise_std=NOISE, x_start=x0,
                      max_iter=BUDGET, verbose=0, step_bound=100.0)
            if which == 'ARC-default':
                kw['use_zero_hessian'] = False
            elif which == 'ARC-zeroH':
                pass
            elif which == 'ARC-diagLS':
                kw['use_zero_hessian'] = False
                _setH(_diagLS); _setK(_hookStore)
            elif which == 'ARC-hessupd':
                kw['use_zero_hessian'] = False
                kw['use_hessian_update'] = True
            elif which == 'zeroH-nodrift':
                _setK(_hookNoDrift)
            _, h = TOBYQA.tobyqa_optimize(**kw)
    except Exception:
        return None
    finally:
        _setH(_ob); _setK(_okkt)
    return float(np.min([f(r['x']) for r in h][:BUDGET]))


def report(data):
    inst, fstar, f0s = data['inst'], data['fstar'], data['f0s']
    def rate(w, tau, envf=None):
        ok = tot = 0
        for k, v in inst.items():
            p, e, d, s = k
            if envf and e != envf:
                continue
            tot += 1
            b = v.get(w); fs = fstar[k]
            if b is None or fs is None:
                continue
            if b <= fs + tau * (f0s[(p, d)] - fs):
                ok += 1
        return 100.0 * ok / tot if tot else float('nan')
    print(f"{len(inst)} instances/variant, budget {data['BUDGET']}")
    print(f"{'variant':<16}" + "".join(f"{'tau=' + str(t):>14}" for t in data['TAUS']))
    for w in data['WH']:
        print(f"{w:<16}" + "".join(f"{rate(w, t):>13.1f}%" for t in data['TAUS']))
    print("\npaired sign tests:")
    def pair(a, b):
        w1 = w2 = 0
        for k, v in inst.items():
            x, y = v.get(a), v.get(b)
            if x is None or y is None:
                continue
            if x < y - 1e-12 * max(1, abs(y)):
                w1 += 1
            elif y < x - 1e-12 * max(1, abs(y)):
                w2 += 1
        nn = w1 + w2
        pv = min(sum(comb(nn, i) for i in range(min(w1, w2) + 1)) * 2 / 2 ** nn, 1.0) if nn else 1.0
        print(f"  {a:<15} vs {b:<15}: {w1:>3} : {w2:<3}  p={pv:.4f}")
    for a, b in [('ARC-default', 'ARC-zeroH'), ('ARC-diagLS', 'ARC-zeroH'),
                 ('ARC-hessupd', 'ARC-zeroH'),
                 ('ARC-zeroH', 'zeroH-nodrift'), ('ARC-zeroH', 'BFGS-fd')]:
        pair(a, b)


def main():
    if '--report' in sys.argv:
        report(pickle.load(open(RESULT, 'rb')))
        return
    os.makedirs('results', exist_ok=True)
    jobs = [(p, e, d, s, w) for p in PROBS for d in DIMS for e in ENVS
            for s in SEEDS for w in WH]
    res = pickle.load(open(CKPT, 'rb')) if os.path.exists(CKPT) else {}
    new = 0
    for j in jobs:
        if j in res:
            continue
        res[j] = run(*j)
        new += 1
        if new % 15 == 0:
            pickle.dump(res, open(CKPT, 'wb'))
            print(f"[progress] {sum(1 for j in jobs if j in res)}/{len(jobs)}", flush=True)
    pickle.dump(res, open(CKPT, 'wb'))
    f0s = {(p, d): PROBS[p](np.linspace(-1.5, 1.2, d)) for p in PROBS for d in DIMS}
    inst = collections.defaultdict(dict)
    for (p, e, d, s, w), v in res.items():
        inst[(p, e, d, s)][w] = v
    fstar = {}
    for k, v in inst.items():
        z = [x for x in v.values() if x is not None]
        fstar[k] = min(z) - 1e-8 if z else None
    data = {'inst': dict(inst), 'fstar': fstar, 'f0s': f0s, 'WH': WH,
            'TAUS': TAUS, 'ENVS': ENVS, 'BUDGET': BUDGET}
    pickle.dump(data, open(RESULT, 'wb'))
    report(data)


if __name__ == '__main__':
    main()
