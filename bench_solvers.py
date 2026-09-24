"""
bench_solvers.py
================
统一求解器包装层。

所有求解器共用同一接口::

    run(oracle, x0, dim, noise_std, budget) -> list[(eval_index, x)]

返回每次 oracle 调用所在点的序列（长度 = 实际消耗的求值次数）。
评分由 run_benchmark.py 在无噪声静态目标上重新计算。

预算强制
--------
每个求解器都被 BudgetedOracle 包裹：达到 budget 次调用后抛
_BudgetExhausted，包装层捕获并正常返回。返回值中的 n_evals 与
terminated_early 标志区分"自行停机"与"预算耗尽"。
"""

from __future__ import annotations
import numpy as np


class _BudgetExhausted(Exception):
    pass


class BudgetedOracle:
    """把 oracle 包成"计数 + 记录 + 到点抛异常"，并维护 t = 调用序号。"""

    __slots__ = ('f', 'budget', 'n', 'xs', '_t0')

    def __init__(self, oracle, budget):
        self.f = oracle
        self.budget = int(budget)
        self.n = 0
        self.xs = []

    def __call__(self, x, t=None):
        if self.n >= self.budget:
            raise _BudgetExhausted
        self.n += 1
        xx = np.array(x, float, copy=True)
        self.xs.append(xx)
        # 时间戳约定：t = 调用序号。忽略调用方传入的 t，
        # 使所有求解器处于同一时间轴。
        return self.f(xx, float(self.n))


# ============================================================================
# TOBYQA 及其变体
# ============================================================================
def _import_arc():
    import TOBYQA, ARC_subproblem, KKT_posterior
    return TOBYQA, ARC_subproblem, KKT_posterior


def _zero_hessian(lam, dX):
    n = dX.shape[1]
    return np.zeros((n, n))


def _linear_kkt_factory(orig):
    """KKT 求解器：K = I/rho，不含二次核 A（纯线性模型）。"""
    def solver(X_pts, F_pts, T_pts, x_k, t_now, rho, noise_std=None):
        m, n = X_pts.shape
        d = X_pts - x_k
        Xa = np.zeros((n + 2, m))
        Xa[0, :] = 1.0; Xa[1:n+1, :] = d.T; Xa[n+1, :] = T_pts - t_now
        S = rho*(Xa @ Xa.T); rhs = rho*(Xa @ F_pts)
        if not (np.all(np.isfinite(S)) and np.all(np.isfinite(rhs))):
            S = np.nan_to_num(S); rhs = np.nan_to_num(rhs)
        sc = float(np.max(np.abs(np.diag(S))))
        if not np.isfinite(sc) or sc <= 0: sc = 1.0
        try:
            params = np.linalg.solve(S/sc, rhs/sc)
            if not np.all(np.isfinite(params)): raise np.linalg.LinAlgError
        except np.linalg.LinAlgError:
            params = np.linalg.lstsq(S/sc, rhs/sc, rcond=None)[0]
        c, g, gamma = params[0], params[1:n+1], params[n+1]
        lam = rho*(F_pts - Xa.T @ params)
        R_eff = max(float(lam @ lam)/rho/max(m-n-2, 1), 1e-16)
        R_obs = max(float(noise_std)**2, 1e-16) if noise_std is not None else R_eff
        try:
            Sinv = np.linalg.pinv(S)
            P = Sinv @ (rho*rho*(Xa @ Xa.T)) @ Sinv
            P = 0.5*(P + P.T)
            if np.any(np.diag(P) < 0) or not np.all(np.isfinite(P)):
                P = np.nan_to_num(P); np.fill_diagonal(P, np.maximum(np.diag(P), 0.0))
        except np.linalg.LinAlgError:
            P = np.zeros((n+2, n+2))
        return c, g, gamma, lam, P, R_eff, R_obs
    return solver


def _nodrift_factory(orig):
    """KKT 求解器包装：将返回的 gamma 置 0（等价于去掉 X_aug 的时间行）。"""
    def solver(X, F, T, xk, tn, rho, noise_std=None):
        r = orig(X, F, T, xk, tn, rho, noise_std=noise_std)
        return (r[0], r[1], 0.0) + tuple(r[3:])
    return solver


def _run_arc(bo, x0, dim, noise_std, budget, variant):
    TOBYQA, ARC_sub, KKT_post = _import_arc()
    ok = KKT_post.solve_kkt_with_posterior

    def setK(fn):
        KKT_post.solve_kkt_with_posterior = fn
        TOBYQA.KKT_posterior.solve_kkt_with_posterior = fn

    kw = dict(obj_func=bo, n_vars=dim, noise_std=noise_std, x_start=x0,
              max_iter=budget, verbose=0, step_bound=100.0)
    try:
        if variant == 'ours':                 # 默认配置（零 Hessian）
            pass
        elif variant == 'ours-quadH':         # 插值 Hessian
            kw['use_zero_hessian'] = False
        elif variant == 'ours-nodrift':       # 去掉时间列
            setK(_nodrift_factory(ok))
        elif variant == 'ours-purelinear':    # 去掉二次核 A
            setK(_linear_kkt_factory(ok))
        elif variant == 'ours-hessupd':       # 增量式 Hessian 更新
            kw['use_zero_hessian'] = False
            kw['use_hessian_update'] = True
        elif variant == 'ours-nosigma':       # 冻结 sigma
            kw.update(gamma1=1.0, gamma2=1.0)
        elif variant.startswith('ours-h'):    # 曲率收缩：H <- alpha * H，alpha = NN/100
            kw['use_zero_hessian'] = False
            kw['hess_alpha'] = int(variant[len('ours-h'):]) / 100.0
        else:
            raise ValueError(variant)
        TOBYQA.tobyqa_optimize(**kw)
    except _BudgetExhausted:
        pass
    finally:
        setK(ok)


# ============================================================================
# 外部基线
# ============================================================================
def _run_pdfo(bo, x0, budget, method):
    """PDFO 的 NEWUOA/BOBYQA；BOBYQA 在 PDFO 不可用时回退 Py-BOBYQA。
    其余异常向上传播，由 run_solver 记录。"""
    try:
        from pdfo import pdfo as _pdfo
        _pdfo(lambda z: bo(z), x0, method=method,
              options={'maxfev': budget, 'quiet': True})
        return
    except _BudgetExhausted:
        return
    except ImportError as e:
        if method != 'bobyqa':
            raise RuntimeError(f'pdfo unavailable: {e}') from e
# BOBYQA fallback: Py-BOBYQA
    import pybobyqa
    try:
        pybobyqa.solve(lambda z: float(bo(z)), np.asarray(x0, float),
                       maxfun=budget, objfun_has_noise=True,
                       do_logging=False)
    except _BudgetExhausted:
        pass


def _run_pybobyqa(bo, x0, budget):
    import pybobyqa
    try:
        pybobyqa.solve(lambda z: float(bo(z)), np.asarray(x0, float),
                       maxfun=budget, objfun_has_noise=True,
                       do_logging=False)
    except _BudgetExhausted:
        pass


def _run_dfols(bo, x0, budget):
    import dfols
    try:
        dfols.solve(lambda z: np.array([bo(z)]), x0, maxfun=budget,
                    objfun_has_noise=True, print_progress=False)
    except _BudgetExhausted:
        pass


def _run_scipy(bo, x0, budget, method, eps=1e-4):
    from scipy.optimize import minimize
    try:
        opts = {'maxiter': budget*10}
        if method == 'BFGS':
            opts['eps'] = eps
        minimize(lambda z: bo(z), x0, method=method, options=opts)
    except _BudgetExhausted:
        pass


# ============================================================================
# 注册表
# ============================================================================
ARC_VARIANTS = ['ours', 'ours-quadH', 'ours-nodrift', 'ours-purelinear',
                'ours-hessupd', 'ours-nosigma']
BASELINES = ['newuoa', 'bobyqa', 'pybobyqa', 'dfols', 'neldermead', 'bfgs-fd']
ALL_SOLVERS = ARC_VARIANTS + BASELINES

# 曲率收缩诊断变体 H <- alpha*H。可用 --solvers 显式点名，
# 但不在 ALL_SOLVERS 中，以免改变默认全量实验的求解器集合与 f_L。
ALPHA_VARIANTS = ['ours-h25', 'ours-h50', 'ours-h75']


def run_solver(name, oracle, x0, dim, noise_std, budget):
    """
    返回 dict:
      xs             : list[ndarray]，每次 oracle 调用的点
      n_evals        : 实际消耗的求值次数
      terminated_early: True 表示求解器在预算耗尽前自行停机
      error          : 异常字符串或 None
    """
    bo = BudgetedOracle(oracle, budget)
    err = None
    try:
        if name in ARC_VARIANTS or name in ALPHA_VARIANTS:
            _run_arc(bo, x0, dim, noise_std, budget, name)
        elif name == 'newuoa':
            _run_pdfo(bo, x0, budget, 'newuoa')
        elif name == 'bobyqa':
            _run_pdfo(bo, x0, budget, 'bobyqa')
        elif name == 'pybobyqa':
            _run_pybobyqa(bo, x0, budget)
        elif name == 'dfols':
            _run_dfols(bo, x0, budget)
        elif name == 'neldermead':
            _run_scipy(bo, x0, budget, 'Nelder-Mead')
        elif name == 'bfgs-fd':
            _run_scipy(bo, x0, budget, 'BFGS')
        else:
            raise ValueError(f'unknown solver {name}')
    except _BudgetExhausted:
        pass
    except Exception as e:                     # noqa: BLE001
        err = f'{type(e).__name__}: {e}'[:200]
    return dict(xs=bo.xs, n_evals=bo.n,
                terminated_early=(bo.n < budget), error=err)
