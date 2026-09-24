"""
verify_curvature_kernel.py
==========================
Monte-Carlo verification of the curvature covariance kernel
(Proposition 3.4): for H drawn from the Gaussian orthogonal ensemble
with E[H_ab H_cd] = tau^2 (delta_ac delta_bd + delta_ad delta_bc),
the covariance of y_i = 0.5 * d_i' H d_i equals tau^2 * A with
A_ij = 0.5 * (d_i' d_j)^2.
"""
import numpy as np

n = 8
m = 17
N = 200_000
tau = 1.0
rng = np.random.default_rng(0)

# Random sample geometry.
D = rng.normal(size=(m, n))
A = 0.5 * (D @ D.T) ** 2

# GOE draws: H = tau * (Z + Z') / sqrt(2), so Var(H_ab) = tau^2 (a != b),
# Var(H_aa) = 2 tau^2.
Y = np.empty((N, m))
B = 20_000
for lo in range(0, N, B):
    hi = min(lo + B, N)
    Z = rng.normal(size=(hi - lo, n, n))
    H = tau * (Z + np.transpose(Z, (0, 2, 1))) / np.sqrt(2.0)
    Y[lo:hi] = 0.5 * np.einsum('ia,kab,ib->ki', D, H, D)

Yc = Y - Y.mean(axis=0)
C_emp = (Yc.T @ Yc) / (N - 1)
C_th = tau ** 2 * A

dev = np.linalg.norm(C_emp - C_th, 'fro') / np.linalg.norm(C_th, 'fro')
ratios = np.diag(C_emp) / np.diag(C_th)
print(f"n={n}, m={m}, draws={N}")
print(f"relative Frobenius deviation ||C_emp - tau^2 A||_F / ||tau^2 A||_F = {dev:.2e}")
print(f"diagonal ratios C_emp_ii / (tau^2 A_ii): mean = {ratios.mean():.4f}, "
      f"std = {ratios.std():.4f}")
