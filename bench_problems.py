"""
bench_problems.py
=================
测试问题库与时变环境构造。

问题来源（由 run_benchmark.py 的 --problem-source 选择）
-------------------------------------------------------
1. ``pycutest``
   直接从 CUTEst 拉取可变维度的无约束光滑问题。

2. ``builtin``（无需 CUTEst 环境）
   本文件内置 56 个可缩放的 CUTEst 风格问题（DIXMAAN 族计 12 个），
   为同名问题的标准解析式实现；对维度有整除要求的问题
   （DIXMAAN 族、WOODS、POWELLSG 等）按截断规则适配到任意 n。

时变环境
--------
七种环境的定义见 ENVIRONMENTS。噪声采用相对尺度：
    sigma_eps = noise_rel * scale,   scale = max(|f(x0)|, 1)
漂移幅度同样以 scale 归一。

时间戳约定
----------
每次 oracle 调用使 t 递增 1。调用方（优化器）负责传入 t。
"""

from __future__ import annotations
import numpy as np

# ============================================================================
# 内置问题库
# ============================================================================
# 每个函数签名 f(x: np.ndarray) -> float，对任意 n >= 4 可用。
# 命名沿用 CUTEst。实现均为标准解析式；维度受限者按截断规则适配。

def _pairs(x):
    return x[:-1], x[1:]

# ---- Rosenbrock 族 --------------------------------------------------------
def genrose(x):
    n = len(x)
    return 1.0 + 100.0*np.sum((x[1:] - x[:-1]**2)**2) + np.sum((x[1:] - 1.0)**2)

def srosenbr(x):
    n = len(x)//2*2
    a, b = x[0:n:2], x[1:n:2]
    return np.sum(100.0*(b - a**2)**2 + (1.0 - a)**2)

def extrosnb(x):
    return (x[0] - 1.0)**2 + 100.0*np.sum((x[1:] - x[:-1]**2)**2)

def chnrosnb(x):
    al = np.array([1.25, 1.40, 2.40, 1.40, 1.75, 1.20, 2.25, 1.20, 1.00, 1.10,
                   1.50, 1.60, 1.25, 1.25, 1.20, 1.20, 1.40, 0.50, 0.50, 1.25,
                   1.80, 0.75, 1.25, 1.40, 1.60, 2.00, 1.00, 1.60, 1.25, 2.75,
                   1.25, 1.25, 1.25, 3.00, 1.50, 2.00, 1.25, 1.40, 1.80, 1.50,
                   2.20, 1.40, 1.50, 1.25, 2.00, 1.50, 1.25, 1.40, 0.60, 1.50])
    n = len(x)
    a = np.resize(al, n)[1:]
    return np.sum(16.0*a**2*(x[:-1] - x[1:]**2)**2 + (x[1:] - 1.0)**2)

def fletchcr(x):
    return 100.0*np.sum((x[1:] - x[:-1] + 1.0 - x[:-1]**2)**2)

def nondia(x):
    return (x[0] - 1.0)**2 + np.sum((100.0*x[0] - x[1:]**2)**2)

def liarwhd(x):
    return np.sum(4.0*(x**2 - x[0])**2) + np.sum((x - 1.0)**2)

# ---- 二次 / 四次族 --------------------------------------------------------
def dqdrtic(x):
    return np.sum(x[:-2]**2 + 100.0*x[1:-1]**2 + 100.0*x[2:]**2)

def dqrtic(x):
    j = np.arange(1, len(x)+1)
    return np.sum((x - j)**4)

def quartc(x):
    j = np.arange(1, len(x)+1)
    return np.sum((x - j)**4)

def tridia(x):
    n = len(x); j = np.arange(2, n+1)
    return (x[0] - 1.0)**2 + np.sum(j*(2.0*x[1:] - x[:-1])**2)

def dixon3dq(x):
    return (x[0] - 1.0)**2 + np.sum((x[:-1] - x[1:])**2) + (x[-1] - 1.0)**2

def power(x):
    j = np.arange(1, len(x)+1)
    return np.sum((j*x)**2)

def nondquar(x):
    return ((x[0] - x[1])**2 + np.sum((x[:-2] + x[1:-1] + x[-1])**4)
            + (x[-2] + x[-1])**2)

def tquartic(x):
    return (x[0] - 1.0)**2 + np.sum((x[0]**2 - x[1:]**2)**2)

def arwhead(x):
    return np.sum((x[:-1]**2 + x[-1]**2)**2 - 4.0*x[:-1] + 3.0)

def bdqrtic(x):
    n = len(x)
    if n < 5: return float(np.sum(x**2))
    i = np.arange(n-4)
    a = (-4.0*x[i] + 3.0)**2
    b = (x[i]**2 + 2.0*x[i+1]**2 + 3.0*x[i+2]**2 + 4.0*x[i+3]**2 + 5.0*x[n-1]**2)**2
    return np.sum(a + b)

def engval1(x):
    return np.sum((x[:-1]**2 + x[1:]**2)**2) + np.sum(-4.0*x[:-1] + 3.0)

def edensch(x):
    return 16.0 + np.sum((x[:-1] - 2.0)**4 + (x[:-1]*x[1:] - 2.0*x[1:])**2
                         + (x[1:] + 1.0)**2)

def cragglvy(x):
    n = len(x)//2*2
    if n < 4: return float(np.sum(x**2))
    i = np.arange(0, n-3, 2)
    t1 = (np.exp(np.clip(x[i], -50, 50)) - x[i+1])**4
    t2 = 100.0*(x[i+1] - x[i+2])**6
    t3 = np.tan(np.clip(x[i+2] - x[i+3], -1.5, 1.5))**4
    t4 = x[i]**8
    t5 = (x[i+3] - 1.0)**2
    return np.sum(t1 + t2 + t3 + t4 + t5)

def woods(x):
    n = len(x)//4*4
    if n < 4: return float(np.sum(x**2))
    i = np.arange(0, n, 4)
    return np.sum(100.0*(x[i+1] - x[i]**2)**2 + (1.0 - x[i])**2
                  + 90.0*(x[i+3] - x[i+2]**2)**2 + (1.0 - x[i+2])**2
                  + 10.0*(x[i+1] + x[i+3] - 2.0)**2 + 0.1*(x[i+1] - x[i+3])**2)

def chainwoo(x):
    n = (len(x)-2)//2*2
    if n < 4: return float(np.sum(x**2))
    i = np.arange(0, n-2, 2)
    return 1.0 + np.sum(100.0*(x[i+1] - x[i]**2)**2 + (1.0 - x[i])**2
                        + 90.0*(x[i+3] - x[i+2]**2)**2 + (1.0 - x[i+2])**2
                        + 10.0*(x[i+1] + x[i+3] - 2.0)**2 + 0.1*(x[i+1] - x[i+3])**2)

def powellsg(x):
    n = len(x)//4*4
    if n < 4: return float(np.sum(x**2))
    i = np.arange(0, n, 4)
    return np.sum((x[i] + 10.0*x[i+1])**2 + 5.0*(x[i+2] - x[i+3])**2
                  + (x[i+1] - 2.0*x[i+2])**4 + 10.0*(x[i] - x[i+3])**4)

# ---- 三角 / 指数族 --------------------------------------------------------
def cosine(x):
    return np.sum(np.cos(x[:-1]**2 - 0.5*x[1:]))

def scosine(x):
    n = len(x); p = 6.0
    s = np.exp(p*np.arange(n)/(n-1)) if n > 1 else np.ones(1)
    y = x*s
    return np.sum(np.cos(y[:-1]**2 - 0.5*y[1:]))

def sinquad(x):
    return ((x[0] - 1.0)**4
            + np.sum((np.sin(x[1:-1] - x[-1]) - x[0]**2 + x[1:-1]**2)**2)
            + (x[-1]**2 - x[0]**2)**2)

def schmvett(x):
    if len(x) < 3: return float(np.sum(x**2))
    a, b, c = x[:-2], x[1:-1], x[2:]
    t1 = -1.0/(1.0 + (a - b)**2)
    t2 = -np.sin(np.clip((np.pi*b + c)/2.0, -1e3, 1e3))
    t3 = -np.exp(np.clip(-((a + c)/np.maximum(np.abs(b), 1e-8) - 2.0)**2, -50, 50))
    return np.sum(t1 + t2 + t3)

def genhumps(x):
    z = 2.0
    a = np.sin(z*x[:-1])**2 * np.sin(z*x[1:])**2
    return np.sum(a + 0.05*(x[:-1]**2 + x[1:]**2))

def eg2(x):
    return np.sum(np.sin(np.clip(x[0] + x[:-1]**2 - 1.0, -1e3, 1e3))) \
           + 0.5*np.sin(np.clip(x[-1]**2, -1e3, 1e3))

def morebv(x):
    n = len(x); h = 1.0/(n + 1.0)
    t = h*np.arange(1, n+1)
    xx = np.concatenate(([0.0], x, [0.0]))
    r = 2.0*xx[1:-1] - xx[:-2] - xx[2:] + 0.5*h**2*(xx[1:-1] + t + 1.0)**3
    return np.sum(r**2)

def integreq(x):
    n = len(x); h = 1.0/(n + 1.0)
    t = h*np.arange(1, n+1)
    xx = np.concatenate(([0.0], x, [0.0]))
    s = np.zeros(n)
    for i in range(n):
        ti = t[i]
        lo = np.sum(t[:i+1]*(x[:i+1] + t[:i+1] + 1.0)**3)
        hi = np.sum((1.0 - t[i+1:])*(x[i+1:] + t[i+1:] + 1.0)**3)
        s[i] = x[i] + 0.5*h*((1.0 - ti)*lo + ti*hi)
    return float(np.sum(s**2))

def tointgss(x):
    n = len(x)
    if n < 3: return float(np.sum(x**2))
    d = x[2:]**2 + 0.1
    return np.sum((10.0/(n + 2.0) + x[2:]**2)*(2.0 - np.exp(np.clip(
        -((x[:-2] - x[1:-1])**2)/d, -50, 50))))

def curly(x, K=10):
    n = len(x)
    q = np.zeros(n)
    for i in range(n):
        q[i] = np.sum(x[i:min(i+K+1, n)])
    return float(np.sum(q*(q*(q**2 - 20.0) - 0.1)))

def curly10(x): return curly(x, 10)
def curly20(x): return curly(x, 20)
def curly30(x): return curly(x, 30)

# ---- Broyden 族 -----------------------------------------------------------
def broydn3d(x):
    n = len(x)
    xx = np.concatenate(([0.0], x, [0.0]))
    r = (3.0 - 2.0*xx[1:-1])*xx[1:-1] - xx[:-2] - 2.0*xx[2:] + 1.0
    return float(np.sum(r**2))

def broydn7d(x):
    n = len(x); h = n//2
    xx = np.concatenate(([0.0], x, [0.0]))
    r1 = np.abs((3.0 - 2.0*xx[1:-1])*xx[1:-1] - xx[:-2] - 2.0*xx[2:] + 1.0)**(7.0/3.0)
    r2 = np.abs(x[:h] + x[h:2*h] if 2*h <= n else x[:h] + x[h:h+h])**(7.0/3.0) \
         if h > 0 else np.zeros(0)
    return float(np.sum(r1) + np.sum(r2))

def brybnd(x):
    n = len(x); ml, mu = 5, 1
    s = 0.0
    for i in range(n):
        lo, hi = max(0, i-ml), min(n, i+mu+1)
        idx = [j for j in range(lo, hi) if j != i]
        s += (x[i]*(2.0 + 5.0*x[i]**2) + 1.0
              - np.sum(x[idx]*(1.0 + x[idx])))**2
    return float(s)

def sbrybnd(x):
    n = len(x); p = 6.0
    s = np.exp(p*np.arange(n)/(n-1)) if n > 1 else np.ones(1)
    return brybnd(x*s)

# ---- 稀疏族 ---------------------------------------------------------------
def sparsine(x):
    n = len(x); j = np.arange(1, n+1)
    s = np.zeros(n)
    for i in range(n):
        idx = [(2*i) % n, (3*i+1) % n, (5*i+2) % n, (7*i+3) % n, (11*i+4) % n]
        s[i] = np.sin(np.clip(np.sum(x[idx]), -1e3, 1e3))
    return float(0.5*np.sum(j*s**2))

def sparsqur(x):
    n = len(x); j = np.arange(1, n+1)
    s = np.zeros(n)
    for i in range(n):
        idx = [(2*i) % n, (3*i+1) % n, (5*i+2) % n, (7*i+3) % n, (11*i+4) % n]
        s[i] = np.sum(x[idx]**2)
    return float(0.25*np.sum(j*s**2))

# ---- 罚函数 / 病态族 ------------------------------------------------------
def penalty1(x):
    a = 1e-5
    return float(a*np.sum((x - 1.0)**2) + (np.sum(x**2) - 0.25)**2)

def penalty2(x):
    n = len(x); a = 1e-5
    y = np.exp(np.clip(np.arange(1, 2*n+1)/10.0, -50, 50)) + np.exp(
        np.clip(np.arange(0, 2*n)/10.0, -50, 50))
    ex = np.exp(np.clip(x/10.0, -50, 50))
    t1 = (x[0] - 0.2)**2
    t2 = a*np.sum((ex[1:] + ex[:-1] - y[1:n])**2)
    t3 = a*np.sum((ex[1:] - np.exp(-0.1))**2)
    t4 = (np.sum((n - np.arange(n))*x**2) - 1.0)**2
    return float(t1 + t2 + t3 + t4)

def vardim(x):
    n = len(x); j = np.arange(1, n+1)
    s = np.sum(j*(x - 1.0))
    return float(np.sum((x - 1.0)**2) + s**2 + s**4)

def trigon(x):
    n = len(x); j = np.arange(1, n+1)
    return float(np.sum((n - np.sum(np.cos(x)) + j*(1.0 - np.cos(x)) - np.sin(x))**2))

def arglina(x):
    n = len(x); m = 2*n
    A = np.full((m, n), -2.0/m)
    np.fill_diagonal(A, 1.0 - 2.0/m)
    r = A @ x - 1.0
    return float(np.sum(r**2))

def arglinb(x):
    n = len(x); m = 2*n
    i = np.arange(1, m+1)[:, None]; j = np.arange(1, n+1)[None, :]
    r = (i*j) @ x - 1.0
    return float(np.sum(r**2))

def arglinc(x):
    n = len(x); m = 2*n
    i = np.arange(2, m)[:, None]; j = np.arange(2, n+1)[None, :]
    r = ((i-1)*(j-1)) @ x[1:] - 1.0
    return float(2.0 + np.sum(r**2))

def freuroth(x):
    a = (-13.0 + x[:-1] + ((5.0 - x[1:])*x[1:] - 2.0)*x[1:])**2
    b = (-29.0 + x[:-1] + ((x[1:] + 1.0)*x[1:] - 14.0)*x[1:])**2
    return float(np.sum(a + b))

def brownal(x):
    n = len(x); s = np.sum(x)
    t1 = np.sum((x[:-1] + s - (n + 1.0))**2)
    t2 = (np.prod(np.clip(x, -10, 10)) - 1.0)**2
    return float(t1 + t2)

def noncvxu2(x):
    n = len(x)
    i = np.arange(n)
    a = x[i] + x[(2*i+1) % n] + x[(3*i+2) % n]
    return float(np.sum(a**2 + 4.0*np.cos(a)))

def fletcbv2(x):
    n = len(x); h = 1.0/(n + 1.0)
    xx = np.concatenate(([0.0], x, [0.0]))
    return float(0.5*np.sum((xx[1:] - xx[:-1])**2) - np.sum(x)*h**2
                 - np.sum(np.cos(x))*h**2)

def fletcbv3(x):
    n = len(x); h = 1.0/(n + 1.0); p = 1e-8
    xx = np.concatenate(([0.0], x, [0.0]))
    return float(0.5*p*np.sum((xx[1:] - xx[:-1])**2)
                 - p*np.sum(100.0*np.sin(x/100.0))*h**2)

# ---- DIXMAAN 族（12 个变体）----------------------------------------------
_DIXMAAN_PARAMS = {
    'a': (1.0, 0.0000, 0.125,  0.125,  (0, 0, 0, 0)),
    'b': (1.0, 0.0625, 0.0625, 0.0625, (0, 0, 0, 1)),
    'c': (1.0, 0.1250, 0.125,  0.125,  (0, 0, 0, 0)),
    'd': (1.0, 0.2600, 0.260,  0.260,  (0, 0, 0, 0)),
    'e': (1.0, 0.0000, 0.125,  0.125,  (1, 0, 0, 1)),
    'f': (1.0, 0.0625, 0.0625, 0.0625, (1, 0, 0, 1)),
    'g': (1.0, 0.1250, 0.125,  0.125,  (1, 0, 0, 1)),
    'h': (1.0, 0.2600, 0.260,  0.260,  (1, 0, 0, 1)),
    'i': (1.0, 0.0000, 0.125,  0.125,  (2, 0, 0, 2)),
    'j': (1.0, 0.0625, 0.0625, 0.0625, (2, 0, 0, 2)),
    'k': (1.0, 0.1250, 0.125,  0.125,  (2, 0, 0, 2)),
    'l': (1.0, 0.2600, 0.260,  0.260,  (2, 0, 0, 2)),
}

def _dixmaan(x, key):
    """DIXMAAN 族。原问题要求 n = 3m；此处取 m = n//3 并只用前 3m 个坐标
    （不足部分以二次项补足，保证对任意 n 有定义）。"""
    al, be, ga, de = _DIXMAAN_PARAMS[key][:4]
    k1, k2, k3, k4 = _DIXMAAN_PARAMS[key][4]
    n_full = len(x); m = n_full//3
    if m < 1:
        return float(1.0 + np.sum(x**2))
    n = 3*m; z = x[:n]; rest = x[n:]
    i = np.arange(1, n+1)/n
    s1 = al*np.sum(z**2 * i**k1)
    s2 = be*np.sum(z[:-1]**2 * (z[1:] + z[1:]**2)**2 * (i[:-1]**k2))
    s3 = ga*np.sum(z[:2*m]**2 * z[m:3*m]**4 * (i[:2*m]**k3))
    s4 = de*np.sum(z[:m] * z[2*m:3*m] * (i[:m]**k4))
    return float(1.0 + s1 + s2 + s3 + s4 + np.sum(rest**2))

for _k in _DIXMAAN_PARAMS:
    globals()['dixmaan' + _k] = (lambda kk: (lambda x: _dixmaan(x, kk)))(_k)


# ============================================================================
# 注册表：名称 -> (函数, 初值构造)
# ============================================================================
def _x0_ones(n):    return np.ones(n)
def _x0_neg1(n):    return -np.ones(n)
def _x0_half(n):    return np.full(n, 0.5)
def _x0_spread(n):  return np.linspace(-1.5, 1.2, n)
def _x0_two(n):     return np.full(n, 2.0)
def _x0_alt(n):     return np.where(np.arange(n) % 2 == 0, -1.2, 1.0)
def _x0_small(n):   return np.full(n, 1.0/n)
def _x0_idx(n):     return 1.0 - np.arange(1, n+1)/n

_BUILTIN = {
    # name                func          x0
    'genrose':      (genrose,      _x0_alt),
    'srosenbr':     (srosenbr,     _x0_alt),
    'extrosnb':     (extrosnb,     _x0_neg1),
    'chnrosnb':     (chnrosnb,     _x0_alt),
    'fletchcr':     (fletchcr,     _x0_zero := (lambda n: np.zeros(n))),
    'nondia':       (nondia,       _x0_neg1),
    'liarwhd':      (liarwhd,      _x0_two),
    'dqdrtic':      (dqdrtic,      _x0_spread),
    'dqrtic':       (dqrtic,       _x0_two),
    'quartc':       (quartc,       _x0_two),
    'tridia':       (tridia,       _x0_ones),
    'dixon3dq':     (dixon3dq,     _x0_neg1),
    'power':        (power,        _x0_ones),
    'nondquar':     (nondquar,     _x0_alt),
    'tquartic':     (tquartic,     _x0_half),
    'arwhead':      (arwhead,      _x0_ones),
    'bdqrtic':      (bdqrtic,      _x0_ones),
    'engval1':      (engval1,      _x0_two),
    'edensch':      (edensch,      _x0_zero),
    'cragglvy':     (cragglvy,     _x0_ones),
    'woods':        (woods,        _x0_alt),
    'chainwoo':     (chainwoo,     _x0_alt),
    'powellsg':     (powellsg,     _x0_spread),
    'cosine':       (cosine,       _x0_ones),
    'scosine':      (scosine,      _x0_ones),
    'sinquad':      (sinquad,      _x0_small),
    'schmvett':     (schmvett,     _x0_half),
    'genhumps':     (genhumps,     _x0_neg1),
    'eg2':          (eg2,          _x0_zero),
    'morebv':       (morebv,       _x0_half),
    'integreq':     (integreq,     _x0_idx),
    'tointgss':     (tointgss,     _x0_spread),
    'curly10':      (curly10,      _x0_small),
    'curly20':      (curly20,      _x0_small),
    'curly30':      (curly30,      _x0_small),
    'broydn3d':     (broydn3d,     _x0_neg1),
    'broydn7d':     (broydn7d,     _x0_neg1),
    'brybnd':       (brybnd,       _x0_neg1),
    'sbrybnd':      (sbrybnd,      _x0_neg1),
    'sparsine':     (sparsine,     _x0_half),
    'sparsqur':     (sparsqur,     _x0_half),
    'penalty1':     (penalty1,     _x0_idx),
    'penalty2':     (penalty2,     _x0_half),
    'vardim':       (vardim,       _x0_idx),
    'trigon':       (trigon,       _x0_small),
    'arglina':      (arglina,      _x0_ones),
    'arglinb':      (arglinb,      _x0_ones),
    'arglinc':      (arglinc,      _x0_ones),
    'freuroth':     (freuroth,     _x0_half),
    'brownal':      (brownal,      _x0_half),
    'noncvxu2':     (noncvxu2,     _x0_idx),
    'fletcbv2':     (fletcbv2,     _x0_small),
    'fletcbv3':     (fletcbv3,     _x0_small),
}
for _k in _DIXMAAN_PARAMS:
    _BUILTIN['dixmaan' + _k] = (globals()['dixmaan' + _k], _x0_two)

BUILTIN_NAMES = sorted(_BUILTIN.keys())


def get_problem(name, dim, source='builtin', pycutest_cache=None):
    """返回 (f, x0)。f 接受 ndarray 返回 float（无噪声、静态）。"""
    if source == 'pycutest':
        import pycutest
        p = pycutest.import_problem(name.upper(), sifParams={'N': dim})
        return (lambda z: float(p.obj(np.asarray(z, float)))), np.asarray(p.x0, float)
    f, x0f = _BUILTIN[name]
    def wrapped(z):
        try:
            v = float(f(np.asarray(z, float)))
        except (FloatingPointError, OverflowError, ValueError):
            return np.inf
        return v if np.isfinite(v) else np.inf
    return wrapped, np.asarray(x0f(dim), float)


def list_problems(source='builtin', pycutest_filter=None):
    if source == 'pycutest':
        import pycutest
        names = pycutest.find_problems(objective='other', constraints='unconstrained',
                                       regular=True, userN=True)
        return sorted(set(n.lower() for n in names))
    return list(BUILTIN_NAMES)


# ============================================================================
# 时变环境
# ============================================================================
ENVIRONMENTS = ['Static', 'LinearDrift', 'MultCoupling', 'PeriodicAdd',
                'PeriodicMult', 'ChaoticDrift', 'Hetero']

# 环境超参数
ENV_PARAMS = dict(
    beta_lin      = 0.01,    # LinearDrift:  v += beta_lin * scale * t
    mult_rate     = 1e-4,    # MultCoupling: v *= (1 + mult_rate * t)
    per_amp_add   = 0.05,    # PeriodicAdd:  v += per_amp_add * scale * sin(2 pi t / T)
    per_amp_mult  = 0.05,    # PeriodicMult: v *= (1 + per_amp_mult * sin(2 pi t / T))
    period_T      = 50.0,    # 周期（以 oracle 调用数计）
    chaos_amp     = 0.03,    # ChaoticDrift: v += chaos_amp * scale * lorenz_x(t)
    lorenz_sigma  = 10.0,
    lorenz_rho    = 28.0,
    lorenz_beta   = 8.0/3.0,
    lorenz_dt     = 0.01,    # 每次 oracle 调用推进的 Lorenz 时间步
    lorenz_x0     = (1.0, 1.0, 1.0),
    hetero_rate   = 0.01,    # Hetero: v += hetero_rate*scale*t*(1+tanh(||x-x0||))
)


class _Lorenz:
    """Lorenz 系统的 x 分量，按 oracle 调用数推进。确定性、可复现。"""
    __slots__ = ('s', 'r', 'b', 'dt', 'st', 'cache', 'tmax')

    def __init__(self, p):
        self.s, self.r, self.b = p['lorenz_sigma'], p['lorenz_rho'], p['lorenz_beta']
        self.dt = p['lorenz_dt']
        self.st = np.array(p['lorenz_x0'], float)
        self.cache = [self.st[0]]
        self.tmax = 0

    def x_at(self, t):
        t = int(max(0, min(t, 10**7)))
        while self.tmax < t:
            x, y, z = self.st
            dx = self.s*(y - x); dy = x*(self.r - z) - y; dz = x*y - self.b*z
            self.st = self.st + self.dt*np.array([dx, dy, dz])
            self.st = np.clip(self.st, -1e3, 1e3)
            self.cache.append(self.st[0]); self.tmax += 1
        return self.cache[t]


def make_oracle(f, x0, env, seed, noise_rel, params=None):
    """
    构造时变含噪 oracle。

    返回 (oracle, noise_std, scale)，其中 oracle(x, t) -> float。
    t 由调用方按"每次求值 +1"的约定传入。

    噪声与漂移幅度均以 scale = max(|f(x0)|, 1) 归一。
    """
    p = dict(ENV_PARAMS); p.update(params or {})
    f0 = f(x0)
    scale = max(abs(f0), 1.0) if np.isfinite(f0) else 1.0
    noise_std = noise_rel*scale
    rng = np.random.default_rng(np.uint64(0x9E3779B97F4A7C15) ^ np.uint64(seed))
    lor = _Lorenz(p) if env == 'ChaoticDrift' else None
    T = p['period_T']

    def oracle(x, t):
        x = np.asarray(x, float)
        v = f(x)
        if not np.isfinite(v):
            return np.inf
        if env == 'LinearDrift':
            v = v + p['beta_lin']*scale*t
        elif env == 'MultCoupling':
            v = v*(1.0 + p['mult_rate']*t)
        elif env == 'PeriodicAdd':
            v = v + p['per_amp_add']*scale*np.sin(2.0*np.pi*t/T)
        elif env == 'PeriodicMult':
            v = v*(1.0 + p['per_amp_mult']*np.sin(2.0*np.pi*t/T))
        elif env == 'ChaoticDrift':
            v = v + p['chaos_amp']*scale*lor.x_at(t)
        elif env == 'Hetero':
            d = float(np.linalg.norm(x - x0))
            v = v + p['hetero_rate']*scale*t*(1.0 + np.tanh(d))
        return float(v + rng.normal(0.0, noise_std))

    return oracle, noise_std, scale
