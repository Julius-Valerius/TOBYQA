"""
verify_drift_decoupling.py
==========================
Numerical verification of the two structural results (Section 5.3).

A: affine drift F = f + beta*t injected into fixed interpolation data;
   compares the recovered gradient and gamma with the drift-free solve (Theorem 3.2).
B: quadratic drift mu(t) = 0.5*L_mu*t^2; measures the bias
   |gamma - gamma_stat - mu'(t_k)| as the time span tau_max is scaled (Theorem 3.3).
"""
import numpy as np
import KKT_posterior as KP


def make_set(n, delta):
    # Center + axis probes +/- delta*e_i, time stamps increasing by 1.
    m = 2 * n + 1
    X = np.zeros((m, n))
    T = np.zeros(m)
    idx = 1
    t = 0.0
    for i in range(n):
        t += 1.0; X[idx] = 0.0; X[idx][i] += delta; T[idx] = t; idx += 1
        t += 1.0; X[idx] = 0.0; X[idx][i] -= delta; T[idx] = t; idx += 1
    return X, T, t


def rosen(x):
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


n = 10
delta = 0.1
rho = 1e3
xc = np.full(n, 0.5)

print("A: affine drift F = f + beta*t   (n=10, delta=0.1, rho=1e3)")
X, T, tn = make_set(n, delta)
Xa = X + xc
Fstat = np.array([rosen(x) for x in Xa])
for beta in [0.0, 1.0, 1e3, 1e6]:
    F = Fstat + beta * T
    c, g, gam, lam, P, Re, Ro = KP.solve_kkt_with_posterior(Xa, F, T, xc, tn, rho)
    if beta == 0.0:
        g0, gam0 = g.copy(), gam
    print(f"  beta={beta:<8.0e}  ||g - g_stat|| = {np.linalg.norm(g - g0):.3e}"
          f"   gamma - gamma_stat = {gam - gam0:+.10f}")

print()
print("B: quadratic drift mu(t) = 0.5*L_mu*t^2, bias vs tau_max")
L_mu = 2.0
for scale in [1.0, 0.5, 0.25, 0.125]:
    X, T, tn = make_set(n, delta)
    T = T * scale
    tn = tn * scale
    Xa = X + xc
    Fstat = np.array([rosen(x) for x in Xa])
    F = Fstat + 0.5 * L_mu * T ** 2
    c, g, gam, lam, P, Re, Ro = KP.solve_kkt_with_posterior(Xa, F, T, xc, tn, rho)
    c0, g0s, gam0s, *_ = KP.solve_kkt_with_posterior(Xa, Fstat, T, xc, tn, rho)
    tau = np.max(np.abs(T - tn))
    bias = abs(gam - gam0s - L_mu * tn)
    print(f"  tau_max={tau:6.3f}  |bias| = {bias:.4e}   bias/tau_max = {bias / tau:.4f}"
          f"   ||g - g_stat|| = {np.linalg.norm(g - g0s):.3e}")
