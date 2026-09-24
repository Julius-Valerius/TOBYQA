"""
TOBYQA: Time-augmented Optimization BY Quadratic Approximation.

Derivative-free minimization of a noisy, time-varying objective.  A
time-augmented interpolation system (KKT_posterior.py) recovers the
model constant c, the gradient g, and a scalar drift coefficient gamma
from one saddle-point solve; steps are computed by adaptive cubic
regularization (ARC_subproblem.py).  With use_zero_hessian=True (the
default) the model Hessian used in the step computation is zero, and
curvature enters only through the residual-covariance weighting of the
interpolation system.
"""


import numpy as np
import KKT_posterior
import ARC_subproblem


def tobyqa_optimize(obj_func, n_vars, noise_std, x_start,
                    use_zero_hessian=True,
                    initial_sigma=None, max_iter=1000,
                        rho_override=None,
                        gamma1=2.0, gamma2=1.2, k_tol=3.0,
                        beta_ucb=0.0,
                        use_geometry_step=False,
                        geom_threshold=1e6,
                        use_hessian_update=False,
                        use_affine_scaling=False,
                        affine_mode='dynamic',
                        hess_alpha=1.0,
                        arc_aniso=0.0,
                        sigma_law='basic',
                        step_bound=None,
                        verbose=0):
    """
    Minimize a noisy time-varying objective.

    Parameters
    ----------
    obj_func : callable, obj_func(x, t) -> float.
        Noisy time-varying objective.
    n_vars : int.
        Problem dimension.
    noise_std : float.
        A-priori noise standard deviation estimate.
    x_start : array_like, length n_vars.
        Initial point.
    use_zero_hessian : bool.
        If True (default), the step computation uses a zero model
        Hessian; the KKT multipliers are not converted into an explicit
        Hessian.  Ignored when use_hessian_update is True.
    initial_sigma : float or None.
        Initial cubic regularization parameter. If None, estimated
        from the axis-probe spread.
    max_iter : int.
        Maximum iterations.
    rho_override : float or None.
        KKT ridge parameter. If None, inferred from probe.
    gamma1, gamma2 : float.
        Sigma update rates: success -> sigma /= gamma1,
        failure -> sigma *= gamma2.
    k_tol : float.
        Acceptance tolerance multiplier:
        tolerance = k_tol * posterior_std.
    beta_ucb : float.
        UCB confidence multiplier (0 = disabled). Adds an uncertainty
        penalty to the subproblem.
    use_hessian_update : bool.
        If True, solve the KKT system for the model increment against
        the residuals r_i = F_i - Q_old(y_i), i.e. minimize
        ||H - H_old||_F; if False, fit from scratch (min ||H||_F).
    use_affine_scaling : bool.
        If True, solve the ARC subproblem in scaled coordinates
        s = S s_hat with S = diag(1/sqrt(|H_jj|)). The scaling affects
        only the subproblem, not the KKT system.
    verbose : int.
        0=silent, 1=summary, 2=per-iteration trace.

    Returns
    -------
    x_k : ndarray.
        Final incumbent.
    history : list of dict.
        Query history {'x', 'f', 't'}.
    """
    n = n_vars
    m = 2 * n + 1

    def _log(level, msg):
        if verbose >= level:
            print(msg)

    # ======================================================================
    # Phase 0: axis-probe initialization
    # ======================================================================
    x_k = np.array(x_start, dtype=float).copy()
    t_now = 0.0
    history = []

    X_pts = np.zeros((m, n))
    F_pts = np.zeros(m)
    T_pts = np.zeros(m)

    X_pts[0] = x_k
    T_pts[0] = t_now
    F_pts[0] = obj_func(X_pts[0], T_pts[0])

    # Initial probe radius
    delta_probe = 2.0

    idx = 1
    for i in range(n):
        t_now += 1.0
        x_step = x_k.copy()
        x_step[i] += delta_probe
        X_pts[idx] = x_step
        T_pts[idx] = t_now
        F_pts[idx] = obj_func(x_step, t_now)
        idx += 1

        t_now += 1.0
        x_step = x_k.copy()
        x_step[i] -= delta_probe
        X_pts[idx] = x_step
        T_pts[idx] = t_now
        F_pts[idx] = obj_func(x_step, t_now)
        idx += 1

    for i_init in range(m):
        history.append({
            'x': X_pts[i_init].copy(),
            'f': F_pts[i_init],
            't': T_pts[i_init],
        })

    F_old = F_pts[0]
    t_old_accept = T_pts[0]

    # ---- Hyperparameter inference from probe ----
    f_spread = float(np.max(F_pts) - np.min(F_pts))
    f_scale = max(f_spread, 1e-8)
    R = max(noise_std ** 2, (1e-6 * f_scale) ** 2, 1e-16)

    # Ridge parameter for the KKT system: K = A + rho^{-1} I with
    # rho = tau_hat^2 / R, where tau_hat is the probe value spread per
    # squared radius and R the observation-noise variance.  The clip
    # bounds are numerical guards.
    tau_hat = max(f_scale / (delta_probe ** 2), 1e-8)
    rho = float(np.clip(tau_hat ** 2 / R, 1e2, 1e14))
    if rho_override is not None:
        rho = float(rho_override)

    # Initial cubic regularization parameter:
    # sigma ~ f_spread / delta_probe^3.  Only a lower floor is applied.
    if initial_sigma is None:
        sigma = max(f_scale / (delta_probe ** 3), 1e-8)
    else:
        sigma = float(initial_sigma)

    sigma_0 = sigma  # Remember initial for convergence test

    # ---- EWMA ratio state (only used when sigma_law == 'ewma') -----------
    avg_ratio = 1.0

    # ---- Persistent model state (only used when use_hessian_update) ------
    # Q_old is purely spatial (no drift term), so gamma is re-estimated
    # from scratch every iteration.
    c_model = 0.0
    g_model = np.zeros(n)
    H_model = np.zeros((n, n))

    # Geometry quality baseline from the axis probe.  Used by the
    # optional repair step and by the termination test.
    _, _, _, _, P_norm_init, _, _ = KKT_posterior.solve_kkt_with_posterior(
        X_pts, F_pts, T_pts, x_k, t_now, rho, noise_std=noise_std
    )
    geom_baseline = float(np.trace(P_norm_init))
    if not np.isfinite(geom_baseline) or geom_baseline <= 0.0:
        geom_baseline = 1.0
    if use_geometry_step:
        # Trigger when geometry degrades to >5x the baseline
        geom_threshold = 5.0 * geom_baseline
        _log(1, f"Geometry baseline: tr(P_norm)={geom_baseline:.2f}, threshold={geom_threshold:.2f}")

    _log(2, f"=== TOBYQA-ARC initialization (n={n}) ===")
    _log(2, f"Noise std: {noise_std:.3e} → R={R:.3e}")
    _log(2, f"F-spread: {f_spread:.3f} → rho={rho:.2f}, sigma_0={sigma:.3f}")
    _log(2, f"{'Iter':<5} | {'Sigma':<10} | {'||g||':<10} | {'g_unc':<10} | {'Ared':<10} | {'Status'}")
    _log(2, "-" * 80)

    # ======================================================================
    # Main loop
    # ======================================================================
    for k in range(max_iter):
        t_now += 1.0

        # ------------------------------------------------------------------
        # 1. Solve KKT with posterior extraction
        # ------------------------------------------------------------------
        # Returns a normalized covariance scaffold and two variance values:
        #   R_eff: residual-based (used for the acceptance tolerance)
        #   R_obs: observation noise (used for the termination test)
        if use_hessian_update:
            # Right-hand side = residuals against the carried-over model,
            # so the solve yields the model increment (min ||H - H_old||_F).
            # Q_old is purely spatial; gamma is re-estimated each iteration.
            diff_X = X_pts - x_k
            Q_old = (c_model
                     + diff_X @ g_model
                     + 0.5 * np.einsum('ij,jk,ik->i', diff_X, H_model, diff_X))
            RHS_pts = F_pts - Q_old
            if not np.all(np.isfinite(RHS_pts)):
                # Carried-over model is non-finite: reset and fit from
                # scratch for this iteration.
                c_model = 0.0
                g_model = np.zeros(n)
                H_model = np.zeros((n, n))
                RHS_pts = F_pts
        else:
            RHS_pts = F_pts

        c, g, gamma, lam, P_norm, R_eff, R_obs = KKT_posterior.solve_kkt_with_posterior(
            X_pts, RHS_pts, T_pts, x_k, t_now, rho, noise_std=noise_std
        )

        # ------------------------------------------------------------------
        # 1b. Geometry improvement step (optional)
        # ------------------------------------------------------------------
        if use_geometry_step:
            import geometry_step  # optional module; required only here
            needs_repair, geom_metric = geometry_step.check_geometry_quality(
                P_norm, threshold=geom_threshold
            )
            if needs_repair:
                _log(2, f"    [Geometry repair: tr(P_norm)={geom_metric:.2e} > {geom_threshold:.0e}]")
                X_pts, F_pts, T_pts, probe = geometry_step.geometry_improvement_step(
                    obj_func, x_k, t_now, X_pts, F_pts, T_pts, P_norm, n
                )
                history.append(probe)
                t_now += 1.0
                # Re-solve with the updated point set.  The right-hand side
                # is rebuilt because one interpolation point was replaced.
                if use_hessian_update:
                    diff_X = X_pts - x_k
                    Q_old = (c_model
                             + diff_X @ g_model
                             + 0.5 * np.einsum('ij,jk,ik->i', diff_X,
                                               H_model, diff_X))
                    RHS_pts = F_pts - Q_old
                    if not np.all(np.isfinite(RHS_pts)):
                        RHS_pts = F_pts
                else:
                    RHS_pts = F_pts
                c, g, gamma, lam, P_norm, R_eff, R_obs = KKT_posterior.solve_kkt_with_posterior(
                    X_pts, RHS_pts, T_pts, x_k, t_now, rho, noise_std=noise_std
                )

        # ------------------------------------------------------------------
        # 2. Compute the trial step
        # ------------------------------------------------------------------
        diff_X = X_pts - x_k

        if (use_zero_hessian and not use_hessian_update
                and arc_aniso == 0.0 and beta_ucb == 0.0 and not use_affine_scaling):
            # Closed-form analytical ARC step for zero model Hessian:
            #   min_s  g^T s + (sigma/3) ||s||^3
            # has unique global minimizer  s = -g / sqrt(sigma * ||g||)
            # with Pred = ||g|| * ||s||.  Computed in O(n) operations.
            G = None
            g_norm = float(np.linalg.norm(g))
            if not np.isfinite(g_norm) or g_norm < 1e-14:
                s = np.zeros(n)
                Pred = 0.0
            else:
                s = -g / np.sqrt(sigma * g_norm)
                Pred = g_norm * float(np.linalg.norm(s))
        else:
            if use_zero_hessian and not use_hessian_update:
                # Zero model Hessian: no curvature estimate enters the step.
                G = np.zeros((n, n))
            else:
                G = ARC_subproblem.build_hessian_from_multipliers(lam, diff_X)

            # Optional scaling of the recovered curvature.
            if hess_alpha != 1.0:
                G = float(hess_alpha) * G

            if use_hessian_update:
                # (c, g, G) are the increment; the working model is old + new.
                c_model = c_model + c
                g_model = g_model + g
                H_model = H_model + G
                H_model = 0.5 * (H_model + H_model.T)
                if not (np.all(np.isfinite(g_model))
                        and np.all(np.isfinite(H_model))
                        and np.isfinite(c_model)):
                    c_model, g_model, H_model = c, g.copy(), G.copy()
                # The subproblem is driven by the accumulated model.
                g = g_model
                G = H_model

            if arc_aniso > 0.0:
                # ---- Anisotropic cubic regularization -------------------------
                # Replace (sigma/3)||s||^3 by (sigma/3)||s||_M^3 with
                # ||s||_M = sqrt(s' M s), M built from the posterior covariance
                # of the gradient.  M is normalized to unit determinant, so it
                # carries anisotropy only and no scale component; sigma stays in
                # the same metric across iterations.
                P_gg_a = R_obs * P_norm[1:n + 1, 1:n + 1]
                try:
                    w, V = np.linalg.eigh(0.5 * (P_gg_a + P_gg_a.T))
                    w = np.where(np.isfinite(w), np.maximum(w, 0.0), 0.0)
                    w_ref = float(np.median(w[w > 0])) if np.any(w > 0) else 0.0
                    if w_ref <= 0.0:
                        raise np.linalg.LinAlgError
                    # Eigenvalues of M: M_eig = 1 + aniso * w/w_ref
                    mvals = 1.0 + float(arc_aniso) * (w / w_ref)
                    # Normalize to unit determinant.
                    logs = np.log(mvals)
                    if not np.all(np.isfinite(logs)):
                        raise np.linalg.LinAlgError
                    mvals = mvals / float(np.exp(np.mean(logs)))
                    # Change of variables s = T u with T = V diag(mvals^-1/2) V',
                    # so that ||s||_M = ||u||.
                    t_half = V @ np.diag(mvals ** -0.5) @ V.T
                    g_a = t_half @ g
                    G_a = t_half @ G @ t_half
                    s_u, Pred = ARC_subproblem.solve_arc_subproblem(
                        g_a, 0.5 * (G_a + G_a.T), sigma)
                    s = t_half @ s_u
                    if not np.all(np.isfinite(s)):
                        raise np.linalg.LinAlgError
                except np.linalg.LinAlgError:
                    s, Pred = ARC_subproblem.solve_arc_subproblem(g, G, sigma)
            elif beta_ucb > 0.0:
                # Uncertainty-penalized subproblem; uses R_eff.
                P_post_eff = R_eff * P_norm
                P_cc = P_post_eff[0, 0]
                P_cg = P_post_eff[0, 1:n + 1]
                P_gg = P_post_eff[1:n + 1, 1:n + 1]
                s, Pred = ARC_subproblem.solve_arc_with_uncertainty(
                    g, G, sigma, P_cc, P_cg, P_gg, beta_ucb, mm_iters=3
                )
            elif use_affine_scaling:
                # Solve in scaled coordinates s = S s_hat with
                # S = diag(1/sqrt|H_jj|).  The scaling affects only the
                # subproblem; the KKT system above is untouched.
                h_diag = np.abs(np.diag(G))
                pos = h_diag[h_diag > 0.0]
                # Curvature-free directions get the median positive scale.
                fallback = float(np.median(pos)) if pos.size else 1.0
                h_diag = np.where(h_diag > 0.0, h_diag, fallback)
                S_diag = 1.0 / np.sqrt(h_diag)
                if not np.all(np.isfinite(S_diag)) or np.all(S_diag <= 0.0):
                    s, Pred = ARC_subproblem.solve_arc_subproblem(g, G, sigma)
                else:
                    if affine_mode == 'normalized':
                        # S normalized to unit median: it carries anisotropy
                        # only, and sigma is used unchanged.
                        S_diag = S_diag / float(np.median(S_diag))
                        sigma_hat = sigma
                    else:
                        # 'dynamic': sigma is transported by the cube of the
                        # median scale factor.
                        s_ref = float(np.median(S_diag))
                        sigma_hat = sigma * (s_ref ** 3)
                    g_hat = S_diag * g
                    G_hat = G * S_diag[:, None] * S_diag[None, :]
                    s_hat, Pred = ARC_subproblem.solve_arc_subproblem(
                        g_hat, G_hat, sigma_hat
                    )
                    s = S_diag * s_hat
                    # Pred is invariant under the change of variables.
            else:
                # Default: plain ARC subproblem.
                s, Pred = ARC_subproblem.solve_arc_subproblem(g, G, sigma)

        # ------------------------------------------------------------------
        # 2b. Step-length cap (optional)
        # ------------------------------------------------------------------
        # If step_bound is set, cap ||s|| at step_bound * sample_radius,
        # where sample_radius = max_i ||y_i - x_k||, and recompute Pred on
        # the truncated step.
        if step_bound is not None:
            s_norm = float(np.linalg.norm(s))
            sample_radius = float(np.max(np.linalg.norm(diff_X, axis=1)))
            cap = float(step_bound) * sample_radius
            if (np.isfinite(cap) and cap > 0.0 and np.isfinite(s_norm)
                    and s_norm > cap):
                s = s * (cap / s_norm)
                quad_term = 0.5 * float(s @ G @ s) if G is not None else 0.0
                Pred = -(float(g @ s) + quad_term)
            elif not np.isfinite(s_norm):
                s = np.zeros(n)
                Pred = 0.0

        x_new = x_k + s

        # ------------------------------------------------------------------
        # 3. Query and compute drift-compensated reduction
        # ------------------------------------------------------------------
        t_new = t_now
        F_new = obj_func(x_new, t_new)
        history.append({'x': x_new.copy(), 'f': F_new, 't': t_new})

        delta_t = t_new - t_old_accept
        Ared = (F_old - F_new) + gamma * delta_t

        # ------------------------------------------------------------------
        # 4. Acceptance tolerance
        # ------------------------------------------------------------------
        # Scales with the observation-noise variance R_obs:
        # Var(Ared) is dominated by the two independent noise samples in
        # F_old and F_new, each with variance R_obs.
        noise_tol = k_tol * float(np.sqrt(2.0 * R_obs))

        # ------------------------------------------------------------------
        # 5. Accept/reject and update sigma
        # ------------------------------------------------------------------
        if Pred > 1e-14:
            ratio = Ared / Pred
        else:
            ratio = 1.0 if Ared >= 0.0 else -1.0

        # Acceptance criterion: Ared >= -0.5 * noise_tol
        accept = (Ared >= -0.5 * noise_tol)

        if accept:
            if use_hessian_update:
                # Re-expand the carried model about the new centre: with
                # Q(x_k + d) = c + g'd + 0.5 d'Hd, substituting d -> s + d
                # gives
                #   c <- c + g's + 0.5 s'Hs
                #   g <- g + Hs
                #   H <- H            (unchanged)
                Hs = H_model @ s
                c_model = c_model + float(g_model @ s) + 0.5 * float(s @ Hs)
                g_model = g_model + Hs
                if not (np.isfinite(c_model) and np.all(np.isfinite(g_model))):
                    c_model = 0.0
                    g_model = np.zeros(n)
                    H_model = np.zeros((n, n))

            x_k = x_new.copy()
            F_old = F_new
            t_old_accept = t_new

            if sigma_law != 'ewma':
                # Successful step: reduce regularization if ratio > 0.75,
                # otherwise hold.
                if ratio > 0.75:
                    sigma /= gamma1

            sigma = max(sigma, 1e-12)  # Lower bound for numerical safety
        else:
            if sigma_law != 'ewma':
                # Rejected step: increase regularization.
                sigma *= gamma2
                sigma = min(sigma, 1e12)  # Overflow guard

        if sigma_law == 'ewma':
            # ---------------------------------------------------------------
            # EWMA sigma law (sigma_law == 'ewma')
            # ---------------------------------------------------------------
            # The ratio is smoothed with an EWMA, mapped to a radius
            # factor, and converted to a sigma factor via
            #     sigma_factor = radius_factor ** (1 / elasticity),
            # where elasticity = d log||s|| / d log sigma.
            _ELASTICITY = -0.705          # d log||s|| / d log sigma
            _BETA = 0.7                   # EWMA weight
            _ETA = 0.5                    # exponential slope
            _TARGET = 0.7                 # target ratio

            avg_ratio = _BETA * avg_ratio + (1.0 - _BETA) * float(
                np.clip(ratio, -2.0, 2.0))

            if avg_ratio >= _TARGET:
                radius_factor = float(np.clip(
                    np.exp(_ETA * (avg_ratio - _TARGET)), 1.1, 2.0))
            elif avg_ratio > 0.1:
                radius_factor = 1.0
            else:
                radius_factor = float(np.clip(
                    np.exp(_ETA * (avg_ratio - _TARGET)), 0.5, 0.9))

            sigma *= radius_factor ** (1.0 / _ELASTICITY)
            sigma = float(np.clip(sigma, 1e-12, 1e12))

        # Gradient statistics for the termination test.  Uses R_obs
        # (observation noise), not R_eff.
        grad_norm = float(np.linalg.norm(g))
        P_gg_obs = R_obs * P_norm[1:n+1, 1:n+1]
        grad_uncertainty = float(np.sqrt(np.trace(P_gg_obs)))

        # R_eff-scaled uncertainty, used for display only.
        P_gg_eff = R_eff * P_norm[1:n+1, 1:n+1]
        grad_unc_eff = float(np.sqrt(np.trace(P_gg_eff)))

        _log(2, f"{k+1:<5} | {sigma:<10.4g} | {grad_norm:<10.4g} | {grad_unc_eff:<10.4g} | "
                f"{Ared:<10.4f} | {'ACCEPT' if accept else 'REJECT'}")

        # ------------------------------------------------------------------
        # 6. Interpolation set update (time-priority replacement)
        # ------------------------------------------------------------------
        drop_idx = _get_point_to_drop(x_k, X_pts, T_pts)
        X_pts[drop_idx] = x_new
        F_pts[drop_idx] = F_new
        T_pts[drop_idx] = t_new

        # ------------------------------------------------------------------
        # 7. Termination: statistical stationarity test
        # ------------------------------------------------------------------
        # Stop when ||g|| < k_tol * std_obs(g), subject to two guards:
        #   * grad_uncertainty must be a finite positive number and
        #     grad_norm finite;
        #   * geometry gate: tr(P_norm) <= 5 * geom_baseline.  The test
        #     statistic is proportional to sqrt(tr(P_norm)), a geometric
        #     quantity, so the statistic is only used while the geometry
        #     is comparable to the axis-probe baseline.
        step_norm = float(np.linalg.norm(s))

        stat_ok = (np.isfinite(grad_uncertainty) and grad_uncertainty > 0.0
                   and np.isfinite(grad_norm))

        geom_now = float(np.trace(P_norm))
        geom_ok = (np.isfinite(geom_now)
                   and geom_now <= 5.0 * geom_baseline)

        if stat_ok and geom_ok and grad_norm < k_tol * grad_uncertainty:
            _log(1, f">>> ||g||={grad_norm:.2e} < {k_tol}*std_obs="
                    f"{k_tol * grad_uncertainty:.2e} with healthy geometry "
                    f"(tr={geom_now:.3g} <= {5.0 * geom_baseline:.3g}); "
                    f"statistically stationary. <<<")
            break

        # Stop when the step norm vanishes relative to ||x_k||.
        x_scale = max(float(np.linalg.norm(x_k)), 1.0)
        if np.isfinite(step_norm) and step_norm < 1e-14 * x_scale:
            _log(1, f">>> ||s||={step_norm:.2e} vanished relative to "
                    f"||x||={x_scale:.2e}; stagnated. <<<")
            break

        if k == max_iter - 1:
            _log(1, ">>> Max iterations reached. <<<")

    _log(1, "=== Optimization finished ===")
    _log(1, f"Final x_k: {x_k}")
    _log(1, f"Final F_old (with drift/noise): {F_old:.4f}")
    _log(1, f"Final sigma: {sigma:.4g}")

    return x_k, history


def _get_point_to_drop(x_k, X_pts, T_pts):
    """
    Time-priority point replacement: drop the chronologically oldest
    non-center point.
    """
    dist_to_center = np.sum((X_pts - x_k) ** 2, axis=1)
    effective_time = T_pts.copy()
    effective_time[dist_to_center < 1e-12] = np.inf
    return int(np.argmin(effective_time))


# Backwards-compatible alias.
tobyqa_arc_optimize = tobyqa_optimize
