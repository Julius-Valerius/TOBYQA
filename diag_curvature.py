"""
diag_curvature.py
=================
Fixed-geometry curvature diagnostics at the 2n+1 budget
(Appendix A.1-A.3).  Extended Rosenbrock, n=10, center (0.5,...,0.5),
axis-probe geometry, no observation noise; all outputs deterministic.
"""
import numpy as np
import KKT_posterior as KP
from ARC_subproblem import build_hessian_from_multipliers


def rosen(x):
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def rosen_hess(x):
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n - 1):
        H[i, i] += -400 * (x[i + 1] - 3 * x[i] ** 2) + 2
        H[i, i + 1] += -400 * x[i]
        H[i + 1, i] += -400 * x[i]
        H[i + 1, i + 1] += 200
    return H


def make_set(n, delta, center):
    m = 2 * n + 1
    X = np.zeros((m, n))
    T = np.zeros(m)
    X[0] = center
    idx = 1
    t = 0.0
    for i in range(n):
        t += 1; X[idx] = center.copy(); X[idx][i] += delta; T[idx] = t; idx += 1
        t += 1; X[idx] = center.copy(); X[idx][i] -= delta; T[idx] = t; idx += 1
    return X, T, t


def solve_H(delta, rho):
    X, T, tn = make_set(n, delta, xc)
    F = np.array([rosen(x) for x in X])
    c, g, gam, lam, P, Re, Ro = KP.solve_kkt_with_posterior(X, F, T, xc, tn, rho)
    return build_hessian_from_multipliers(lam, X - xc)


n = 10
xc = np.full(n, 0.5)
Ht = rosen_hess(xc)
nHt = np.linalg.norm(Ht, 2)
DELTAS = [0.4, 0.2, 0.1, 0.05, 0.025, 0.0125]

print("A.1: relative Hessian error ||H - Hf||_2 / ||Hf||_2")
print(f"  {'delta':>8} | {'rho=1e3':>10} | {'rho=1e12':>10}")
for delta in DELTAS:
    e = [np.linalg.norm(solve_H(delta, rho) - Ht, 2) / nHt for rho in (1e3, 1e12)]
    print(f"  {delta:8.4f} | {e[0]:10.3f} | {e[1]:10.3f}")

print()
print("A.2: second-order accuracy condition along steps of length delta (rho=1e3)")
print(f"  {'delta':>8} | {'||(H-Hf)s||/||s||^2':>20} | {'||(H-Hf)s||/||s||':>18}")
for delta in DELTAS[:-1]:
    H = solve_H(delta, 1e3)
    s = np.full(n, delta / np.sqrt(n))
    v = (H - Ht) @ s
    print(f"  {delta:8.4f} | {np.linalg.norm(v) / np.linalg.norm(s) ** 2:20.0f}"
          f" | {np.linalg.norm(v) / np.linalg.norm(s):18.0f}")

print()
print("A.3: diagonal recovery on the same 2n+1 samples (delta=0.1, rho=1e3)")
delta = 0.1
X, T, tn = make_set(n, delta, xc)
F = np.array([rosen(x) for x in X])
# Second-order difference quotients from the +/- delta*e_i probe pairs.
fd = np.array([(F[1 + 2 * i] + F[2 + 2 * i] - 2 * F[0]) / delta ** 2 for i in range(n)])
H = solve_H(delta, 1e3)
dHt = np.diag(Ht)
d = X - xc
A = 0.5 * (d @ d.T) ** 2
print(f"  difference-quotient diagonal relative error = "
      f"{np.linalg.norm(fd - dHt) / np.linalg.norm(dHt):.4f}")
print(f"  interpolation-Hessian diagonal relative error = "
      f"{np.linalg.norm(np.diag(H) - dHt) / np.linalg.norm(dHt):.4f}")
print(f"  ||offdiag H||_F = {np.linalg.norm(H - np.diag(np.diag(H)), 'fro'):.3e}")
print(f"  rank(A) = {np.linalg.matrix_rank(A, tol=1e-8 * np.abs(A).max())}  (m = {2 * n + 1})")
