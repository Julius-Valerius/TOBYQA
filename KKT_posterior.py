"""
KKT_posterior.py
===============
Time-augmented KKT interpolation system with posterior covariance
extraction.

Solves the soft-constrained system with kernel K = A + rho^{-1} I and
constraint block X_aug (rows: constant, displacements, time offsets)
for the parameters (c, g, gamma), and returns a normalized covariance
scaffold obtained from the Schur complement
S = X_aug K^{-1} X_aug^T:

    P_normalized = S^{-1} (X_aug K^{-2} X_aug^T) S^{-1},

which multiplied by a noise variance R gives the covariance of
(c, g, gamma).
"""

import numpy as np


def build_augmented_kkt_matrix_physical(X_pts, T_pts, x_k, t_now, rho):
    """
    Build the augmented KKT matrix in physical coordinates.

    Returns
    -------
    M : ndarray, shape (m+n+2, m+n+2)
    """
    m, n = X_pts.shape
    dim = m + n + 2
    
    diff_X = X_pts - x_k
    A = 0.5 * (np.dot(diff_X, diff_X.T)) ** 2
    W = np.eye(m) / rho
    
    X_aug = np.zeros((n + 2, m))
    X_aug[0, :] = 1.0
    X_aug[1:n + 1, :] = diff_X.T
    X_aug[n + 1, :] = T_pts - t_now
    
    M = np.zeros((dim, dim))
    M[:m, :m] = A + W
    M[:m, m:] = X_aug.T
    M[m:, :m] = X_aug
    
    return M


def solve_kkt_with_posterior(X_pts, F_pts, T_pts, x_k, t_now, rho,
                             noise_std=None):
    """
    Solve augmented KKT and return parameters + normalized covariance scaffold.

    The returned P_normalized can be scaled by different R values for different
    purposes:
      - Acceptance tolerance: R_eff * P_normalized  (absorbs model misfit)
      - Termination test:     R_obs * P_normalized  (pure observation noise)

    Parameters
    ----------
    X_pts, F_pts, T_pts : ndarray
        Interpolation set.
    x_k, t_now, rho : as before.
    noise_std : float or None
        Known observation noise std. If None, estimated from residuals.

    Returns
    -------
    c, g, gamma, lam : as before.
    P_normalized : ndarray, shape (n+2, n+2)
        Covariance scaffold (unit-noise): multiply by R to get actual covariance.
    R_eff : float
        Effective variance (residual-based, includes model misfit).
    R_obs : float
        Pure observation noise variance (= noise_std² if provided, else R_eff).
    """
    m, n = X_pts.shape

    diff_X = X_pts - x_k
    A = 0.5 * (np.dot(diff_X, diff_X.T)) ** 2
    W = np.eye(m) / rho
    K = A + W

    X_aug = np.zeros((n + 2, m))
    X_aug[0, :] = 1.0
    X_aug[1:n + 1, :] = diff_X.T
    X_aug[n + 1, :] = T_pts - t_now

    # ---- Solve against K -------------------------------------------------
    # The identity Pi X_aug = (X_aug K^-1 X_aug^T)^-1 (X_aug K^-1 X_aug^T)
    # = I holds only for the exact solve; an unconditional ridge on K or S
    # would break it.  The exact solve is therefore attempted first, and
    # regularization is applied only when LAPACK fails.
    degraded = False

    try:
        K_inv_F = np.linalg.solve(K, F_pts)
        K_inv_Xaug = np.linalg.solve(K, X_aug.T)
    except np.linalg.LinAlgError:
        degraded = True
        K_scale = float(np.max(np.abs(np.diag(K))))
        if not np.isfinite(K_scale) or K_scale <= 0.0:
            K_scale = 1.0
        K = K + (1e-12 * K_scale) * np.eye(m)
        try:
            K_inv_F = np.linalg.solve(K, F_pts)
            K_inv_Xaug = np.linalg.solve(K, X_aug.T)
        except np.linalg.LinAlgError:
            K_inv_F = np.linalg.lstsq(K, F_pts, rcond=None)[0]
            K_inv_Xaug = np.linalg.lstsq(K, X_aug.T, rcond=None)[0]

    S = X_aug @ K_inv_Xaug
    rhs = X_aug @ K_inv_F

    # Overflow in the products above can leave inf/NaN entries, which make
    # LAPACK fail outright ("SVD did not converge"). Scrub them.
    if not np.all(np.isfinite(S)):
        S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
        degraded = True
    if not np.all(np.isfinite(rhs)):
        rhs = np.nan_to_num(rhs, nan=0.0, posinf=0.0, neginf=0.0)
        degraded = True

    S_scale = float(np.max(np.abs(np.diag(S))))
    if not np.isfinite(S_scale) or S_scale <= 0.0:
        S_scale = 1.0

    # Exact path: unregularized solve on a unit-scaled system (scaling by
    # a positive constant preserves Pi X_aug = I to machine precision).
    params = None
    try:
        params = np.linalg.solve(S / S_scale, rhs / S_scale)
        if not np.all(np.isfinite(params)):
            params = None
    except np.linalg.LinAlgError:
        params = None

    if params is None:
        # Degraded path: minimum-norm least squares, then ridge as last resort.
        degraded = True
        S_reg = S + (1e-12 * S_scale) * np.eye(n + 2)
        try:
            params = np.linalg.lstsq(S_reg / S_scale, rhs / S_scale,
                                     rcond=None)[0]
        except np.linalg.LinAlgError:
            ridge = 1e-8 * np.eye(n + 2)
            try:
                params = np.linalg.solve(
                    (S_reg.T @ S_reg) / (S_scale ** 2) + ridge,
                    (S_reg.T @ rhs) / (S_scale ** 2)
                )
            except np.linalg.LinAlgError:
                params = np.zeros(n + 2)
    else:
        S_reg = S

    c, g, gamma = params[0], params[1:n+1], params[n+1]
    lam = K_inv_F - K_inv_Xaug @ params

    # Residual-based R (absorbs model misfit + noise)
    dof = max(m - n - 2, 1)
    R_eff = float(lam @ (K @ lam)) / dof
    R_eff = max(R_eff, 1e-16)

    # Known observation noise (if provided)
    if noise_std is not None:
        R_obs = max(float(noise_std) ** 2, 1e-16)
    else:
        R_obs = R_eff

    # Normalized covariance scaffold (sandwich form, no R multiplier)
    # B = X_aug K^{-2} X_aug^T is PSD by construction; the sandwich
    # S^{-1} B S^{-1} is therefore PSD in exact arithmetic. Use pinv so a
    # rank-deficient S degrades gracefully instead of producing garbage.
    B = K_inv_Xaug.T @ K_inv_Xaug
    try:
        S_inv = np.linalg.pinv(S_reg)
        P_normalized = S_inv @ B @ S_inv
        P_normalized = 0.5 * (P_normalized + P_normalized.T)
    except np.linalg.LinAlgError:
        # pinv runs an SVD which can fail to converge on badly scaled
        # inputs; fall back to a zero covariance.
        P_normalized = np.zeros((n + 2, n + 2))

    # Round-off can push diagonal entries marginally negative; clip so that
    # every variance read off this matrix is a valid (non-negative) number.
    diag = np.diag(P_normalized)
    if np.any(diag < 0.0) or not np.all(np.isfinite(P_normalized)):
        P_normalized = np.nan_to_num(P_normalized, nan=0.0,
                                     posinf=0.0, neginf=0.0)
        np.fill_diagonal(P_normalized, np.maximum(np.diag(P_normalized), 0.0))

    return c, g, gamma, lam, P_normalized, R_eff, R_obs


def posterior_std_at_offset(P_post, d, n):
    """
    Compute posterior standard deviation at a spatial offset d.
    
    The model at x_k + d is:
        Q(x_k + d) = c + g^T d + (quadratic terms)
    
    The linear part (c + g^T d) has variance:
        a^T P a,  where a = [1; d; 0] (no drift contribution at t_now)
    
    Parameters
    ----------
    P_post : ndarray, shape (n+2, n+2)
        Posterior covariance [c; g; gamma].
    d : ndarray, shape (n,)
        Spatial offset from x_k.
    n : int
        Problem dimension.
    
    Returns
    -------
    std : float
        Posterior standard deviation of the linear predictor.
    """
    a = np.zeros(n + 2)
    a[0] = 1.0
    a[1:n + 1] = d
    # a[n + 1] = 0.0  (no time offset at current time)
    
    var = a @ P_post @ a
    return float(np.sqrt(max(var, 0.0)))


def posterior_std_gamma(P_post, n):
    """
    Extract the posterior standard deviation of gamma.

    Returns
    -------
    std_gamma : float
    """
    var_gamma = P_post[n + 1, n + 1]
    return float(np.sqrt(max(var_gamma, 0.0)))
