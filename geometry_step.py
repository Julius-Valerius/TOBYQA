"""
geometry_step.py
================
Minimal NEWUOA-style geometry improvement step.

When the interpolation matrix conditioning becomes poor (large condition
number of S = X_aug K^{-1} X_aug^T), perform a geometry improvement step:
find the direction that maximally violates Lagrange-polynomial separation,
take a probing step in that direction, and replace the oldest point.

This prevents S from becoming near-singular, which would otherwise cause
the posterior covariance P_gg to explode and trigger premature termination.
"""

import numpy as np


def check_geometry_quality(P_norm, threshold=1e6):
    """
    Estimate conditioning via the normalized posterior covariance.
    
    P_norm is the unit-noise covariance scaffold. Its trace indicates the
    effective "uncertainty budget" of the recovered parameters. When S is
    near-singular, P_norm's diagonal entries explode.
    
    Returns
    -------
    needs_repair : bool
        True if geometry is degraded and a repair step is warranted.
    quality_metric : float
        Trace of P_norm (larger = worse conditioning).
    """
    tr = float(np.trace(P_norm))
    needs_repair = (tr > threshold)
    return needs_repair, tr


def compute_geometry_step_direction(P_norm, n):
    """
    Find the direction that maximally violates Lagrange separation.
    
    The eigenvector of P_gg (gradient block) corresponding to the largest
    eigenvalue indicates the direction with the greatest uncertainty.
    Moving a sample point in this direction tightens the interpolation.
    
    Returns
    -------
    d : ndarray, shape (n,)
        Unit direction for the geometry step (normalized eigenvector).
    """
    P_gg = np.array(P_norm[1:n + 1, 1:n + 1], dtype=float)

    # Guard against non-finite input, which would make the symmetric
    # eigensolver fail.
    if not np.all(np.isfinite(P_gg)):
        P_gg = np.nan_to_num(P_gg, nan=0.0, posinf=0.0, neginf=0.0)

    d = None
    scale = float(np.max(np.abs(P_gg))) if P_gg.size else 0.0
    if np.isfinite(scale) and scale > 0.0:
        try:
            # Rescale to unit magnitude so eigh sees a well-scaled matrix
            eigvals, eigvecs = np.linalg.eigh(P_gg / scale)
            d = eigvecs[:, int(np.argmax(eigvals))]
        except np.linalg.LinAlgError:
            d = None

    if d is None or not np.all(np.isfinite(d)):
        # Coordinate fallback: axis with the largest marginal variance.
        diag = np.abs(np.diag(P_gg))
        d = np.zeros(n)
        d[int(np.argmax(diag)) if np.any(diag > 0.0) else 0] = 1.0

    # Normalize
    nrm = float(np.linalg.norm(d))
    if not np.isfinite(nrm) or nrm < 1e-14:
        d = np.zeros(n)
        d[0] = 1.0
        nrm = 1.0

    return d / nrm


def geometry_improvement_step(obj_func, x_k, t_now, X_pts, F_pts, T_pts,
                               P_norm, n, step_length=0.5):
    """
    Execute a single geometry improvement step.
    
    Takes a probing step in the direction of maximum posterior uncertainty,
    queries the oracle, and replaces the oldest point (excluding center).
    
    Parameters
    ----------
    obj_func : callable
    x_k : ndarray, current center
    t_now : float
    X_pts, F_pts, T_pts : current interpolation set
    P_norm : posterior covariance scaffold
    n : problem dimension
    step_length : float, probe step size (relative to current spread)
    
    Returns
    -------
    X_pts, F_pts, T_pts : updated interpolation set (modified in-place)
    probe : dict with keys 'x', 'f', 't' -- the single oracle query made,
        so the caller can log it in the evaluation history.
    """
    # Find worst direction
    d = compute_geometry_step_direction(P_norm, n)

    # Scale the probe to the current sample spread so the step is
    # commensurate with the local geometry rather than a fixed length.
    spread = float(np.sqrt(np.max(np.sum((X_pts - x_k) ** 2, axis=1))))
    if not np.isfinite(spread) or spread < 1e-12:
        spread = 1.0
    radius = step_length * spread

    # Probe in that direction
    x_probe = x_k + radius * d
    t_probe = t_now + 1.0
    F_probe = float(obj_func(x_probe, t_probe))

    # Replace oldest non-center point
    dist_to_center = np.sum((X_pts - x_k) ** 2, axis=1)
    effective_time = T_pts.copy()
    effective_time[dist_to_center < 1e-12] = np.inf
    drop_idx = int(np.argmin(effective_time))

    X_pts[drop_idx] = x_probe
    F_pts[drop_idx] = F_probe
    T_pts[drop_idx] = t_probe

    probe = {'x': x_probe.copy(), 'f': F_probe, 't': t_probe}
    return X_pts, F_pts, T_pts, probe
