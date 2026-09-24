"""
ARC_subproblem.py
=================
Adaptive cubic regularization (ARC) subproblem solver.

Solves
    min_s  g^T s + (1/2) s^T H s + (sigma/3) ||s||^3.

Method
------
Stationarity condition:
    g + H s + sigma ||s|| s = 0
Setting lambda := sigma ||s||, this becomes
    (H + lambda I) s = -g,   with  ||s(lambda)|| = lambda / sigma.

With eigendecomposition H = Q diag(mu) Q^T and gt := Q^T g,
    ||s(lambda)||^2 = sum_i gt_i^2 / (mu_i + lambda)^2,
which is strictly decreasing in lambda on (max(0, -mu_min), inf),
while lambda/sigma is strictly increasing.  The unique root is found
by safeguarded bisection.

Optional UCB term
-----------------
An uncertainty-aware variant adds  beta * sqrt(a(s)^T P a(s))  where
a(s) = [1; s; 0]. Since sqrt is concave, its tangent is a global upper
bound, so a majorize-minimize (MM) iteration reduces the problem to a
sequence of plain ARC solves with modified (g, H). Each MM step is a
valid upper bound, so the iteration monotonically decreases the true
objective.
"""

import numpy as np


def solve_arc_subproblem(g, H, sigma, tol=1e-10, max_iter=100):
    """
    Solve  min_s  g^T s + 0.5 s^T H s + (sigma/3) ||s||^3.

    Parameters
    ----------
    g : ndarray, shape (n,)
        Gradient.
    H : ndarray, shape (n, n)
        Symmetric Hessian (may be indefinite).
    sigma : float
        Cubic regularization parameter (> 0).
    tol : float
        Root-finding tolerance on the secular equation.
    max_iter : int
        Maximum root-finding iterations.

    Returns
    -------
    s : ndarray, shape (n,)
        Approximate global minimizer of the cubic model.
    pred : float
        Predicted reduction: -(g^T s + 0.5 s^T H s).
        Note this EXCLUDES the cubic term, matching the classical
        definition of predicted reduction used in the ratio test.
    """
    n = len(g)
    sigma = max(float(sigma), 1e-14)

    g = np.array(g, dtype=float)
    H = np.array(H, dtype=float)

    # Sanitize non-finite inputs, and solve the eigenproblem on a
    # unit-scaled copy (eigenvectors are scale-invariant; the eigenvalues
    # are rescaled afterwards).
    if not np.all(np.isfinite(g)):
        g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.all(np.isfinite(H)):
        H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)

    # Symmetrize for numerical safety
    H = 0.5 * (H + H.T)

    # ---- Scale-invariant solve -------------------------------------------
    # Scaling (g, H, sigma) -> (g/c, H/c, sigma/c) leaves the minimizer s
    # unchanged.  Normalize by the largest magnitude and solve; s needs no
    # unscaling, and Pred scales linearly with c.
    g_norm = float(np.linalg.norm(g)) if g.size else 0.0
    H_norm = float(np.linalg.norm(H, 'fro')) if H.size else 0.0
    scale = max(g_norm, H_norm, sigma)
    if scale > 1e3 or scale < 1e-3:
        # Only normalize when far from unit scale (avoids unnecessary ops)
        g_scaled = g / scale
        H_scaled = H / scale
        sigma_scaled = sigma / scale
    else:
        g_scaled, H_scaled, sigma_scaled = g, H, sigma
        scale = 1.0

    h_scale = float(np.max(np.abs(H_scaled))) if H_scaled.size else 0.0
    if not np.isfinite(h_scale) or h_scale <= 0.0:
        h_scale = 1.0
    try:
        mu, Q = np.linalg.eigh(H_scaled / h_scale)
        mu = mu * h_scale
    except np.linalg.LinAlgError:
        # Fallback: drop curvature information and take a regularized
        # gradient step.
        mu = np.zeros(n)
        Q = np.eye(n)

    gt = Q.T @ g_scaled

    g_scaled_norm = float(np.linalg.norm(g_scaled))
    if g_scaled_norm < 1e-300:
        # Zero gradient: only escape direction is negative curvature.
        if mu[0] < -1e-12:
            # Move along the most negative eigenvector.
            s_len = -mu[0] / sigma_scaled
            s = s_len * Q[:, 0]
            pred = -(float(g_scaled @ s) + 0.5 * float(s @ H_scaled @ s)) * scale
            return s, pred
        return np.zeros(n), 0.0

    mu_min = float(mu[0])
    lam_lower = max(0.0, -mu_min)

    def s_norm_at(lam):
        """||s(lambda)|| for the shifted system (H + lam I) s = -g."""
        denom = mu + lam
        # Guard against exact zero denominators (hard case).
        safe = np.where(np.abs(denom) < 1e-300, 1e-300, denom)
        return float(np.sqrt(np.sum((gt / safe) ** 2)))

    def phi(lam):
        """Secular function: ||s(lam)|| - lam/sigma. Decreasing in lam."""
        return s_norm_at(lam) - lam / sigma_scaled

    # ---- Bracket the root ----
    # Start just above lam_lower to avoid the singular boundary.
    lo = lam_lower + max(1e-12, 1e-12 * max(1.0, abs(mu_min)))
    # phi(lo) may be +inf (hard case); treat as positive.
    phi_lo = phi(lo)
    if not np.isfinite(phi_lo):
        phi_lo = np.inf

    if phi_lo <= 0.0:
        # Root is at (or below) the boundary: interior solution with
        # lambda = lam_lower. This includes the case g orthogonal to
        # the minimal eigenspace.
        lam = lo
    else:
        # Expand upper bound until phi turns negative.
        hi = max(lo * 2.0, lam_lower + sigma_scaled * s_norm_at(lam_lower + 1.0) + 1.0)
        for _ in range(200):
            if phi(hi) < 0.0:
                break
            hi *= 2.0
        else:
            hi = max(hi, 1e12)

        # ---- Safeguarded bisection (robust; Newton adds little here) ----
        lam = 0.5 * (lo + hi)
        for _ in range(max_iter):
            val = phi(lam)
            if abs(val) < tol * max(1.0, lam / sigma):
                break
            if val > 0.0:
                lo = lam
            else:
                hi = lam
            lam = 0.5 * (lo + hi)
            if hi - lo < tol * max(1e-16, lam):
                break

    # ---- Recover the step ----
    denom = mu + lam
    safe = np.where(np.abs(denom) < 1e-300, 1e-300, denom)
    s = -(Q @ (gt / safe))

    pred = -(float(g_scaled @ s) + 0.5 * float(s @ H_scaled @ s)) * scale
    return s, pred


def solve_arc_with_uncertainty(g, H, sigma, P_cc, P_cg, P_gg, beta,
                               mm_iters=3, tol=1e-10):
    """
    Solve the uncertainty-penalized cubic subproblem

        min_s  g^T s + 0.5 s^T H s + (sigma/3)||s||^3
               + beta * sqrt(P_cc + 2 P_cg^T s + s^T P_gg s)

    via majorize-minimize. Since sqrt is concave, the tangent at the
    current iterate is a global upper bound; minimizing the tangent
    surrogate is a plain ARC solve with modified (g, H).

    Parameters
    ----------
    g, H : model gradient and Hessian.
    sigma : cubic regularization parameter.
    P_cc : float. Posterior variance of c.
    P_cg : ndarray, shape (n,). Posterior covariance between c and g.
    P_gg : ndarray, shape (n, n). Posterior covariance of g.
    beta : float. Confidence multiplier (0 disables the term).
    mm_iters : int. Number of majorize-minimize iterations.

    Returns
    -------
    s : ndarray. Step.
    pred : float. Predicted reduction (quadratic part only).
    """
    n = len(g)
    if beta <= 0.0:
        return solve_arc_subproblem(g, H, sigma, tol=tol)

    def q_of(s):
        return max(P_cc + 2.0 * float(P_cg @ s) + float(s @ P_gg @ s), 1e-300)

    s = np.zeros(n)
    pred = 0.0
    for _ in range(max(1, mm_iters)):
        q0 = q_of(s)
        w = beta / (2.0 * np.sqrt(q0))
        # Tangent surrogate: beta*sqrt(q) <= const + w * q(s)
        # q(s) contributes 2*w*P_cg to gradient and 2*w*P_gg to Hessian.
        g_eff = g + 2.0 * w * P_cg
        H_eff = H + 2.0 * w * P_gg
        s_new, _ = solve_arc_subproblem(g_eff, H_eff, sigma, tol=tol)
        s = s_new

    # Report predicted reduction on the TRUE quadratic model, so the
    # ratio test remains comparable to the classical definition.
    pred = -(float(g @ s) + 0.5 * float(s @ H @ s))
    return s, pred


def build_hessian_from_multipliers(lam, diff_X):
    """
    Assemble the explicit Hessian G = sum_i lambda_i d_i d_i^T.

    The ARC solve uses an eigendecomposition, so G is formed explicitly.

    Parameters
    ----------
    lam : ndarray, shape (m,). KKT multipliers.
    diff_X : ndarray, shape (m, n). Displacements x_i - x_k.

    Returns
    -------
    G : ndarray, shape (n, n). Symmetric.
    """
    G = (diff_X * lam[:, None]).T @ diff_X
    return 0.5 * (G + G.T)
