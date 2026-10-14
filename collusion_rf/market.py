"""Logit-demand Bertrand market: demand, profit, best response, Nash/monopoly benchmarks, consumer surplus.

Every function takes quality `a` (scalar or per-firm), cost `c` (scalar or per-firm) and `mu`;
the defaults are the symmetric values the regulator assumes.
"""
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .config import MARKET as M


def _attract(p, a, mu) -> tuple[np.ndarray, float]:
    """Logit attractions exp((a - p)/mu) and the outside-good term."""
    return np.exp((a - np.asarray(p, dtype=float)) / mu), float(np.exp(M.a0 / mu))


def demand(p: np.ndarray, a=M.a, mu: float = M.mu) -> np.ndarray:
    """Logit market shares for a price vector (shares sum to < 1 because of the outside good)."""
    e, out = _attract(p, a, mu)
    return e / (e.sum() + out)


def profit(p: np.ndarray, c, a=M.a, mu: float = M.mu) -> np.ndarray:
    """Per-firm profit (p_i - c_i) * q_i."""
    return (np.asarray(p, dtype=float) - c) * demand(p, a, mu)


def consumer_surplus(p: np.ndarray, a=M.a, mu: float = M.mu) -> float:
    """Per-period consumer surplus (market size 1)."""
    e, out = _attract(p, a, mu)
    return float(mu * np.log(e.sum() + out))


def best_response(p_prev: np.ndarray, c, a=M.a, mu: float = M.mu, iters: int = 30, tol: float = 1e-10) -> np.ndarray:
    """Best response of every firm at once, each holding its rivals at p_prev.

    Solves the logit FOC p_i = c_i + mu / (1 - q_i) with a Newton-accelerated fixed point (NOTES.md).
    """
    e, out = _attract(p_prev, a, mu)
    rest = e.sum() - e + out
    c = c + np.zeros_like(e)
    x = c + mu
    for _ in range(iters):
        ex = np.exp((a - x) / mu)
        q = ex / (ex + rest)
        new = x * q + c * (1 - q) + mu
        done = np.max(np.abs(new - x)) < tol
        x = new
        if done:
            break
    return x


def nash_prices(a, c: np.ndarray, mu: float = M.mu) -> np.ndarray:
    """Bertrand-Nash price vector for (possibly asymmetric) firms: iterate best responses from c + mu."""
    p = np.asarray(c, dtype=float) + mu
    for _ in range(500):
        new = best_response(p, c, a, mu)
        if np.max(np.abs(new - p)) < 1e-10:
            break
        p = new
    return new


def monopoly_prices(a, c: np.ndarray, mu: float = M.mu) -> np.ndarray:
    """Joint-profit-maximising prices: a common markup m solving m = mu / (outside share)."""
    c, out = np.asarray(c, dtype=float), float(np.exp(M.a0 / mu))
    m = brentq(lambda m: m - mu * (1 + np.exp((a - c - m) / mu).sum() / out), 0.0, 20.0, xtol=1e-12)
    return c + m


def nash_price(n: int, c: float) -> float:
    """Symmetric Bertrand-Nash price under the regulator's assumptions (a = 2, mu = 0.25)."""
    return float(nash_prices(M.a, np.full(n, c))[0])


def monopoly_price(n: int, c: float) -> float:
    """Symmetric joint-profit-maximising price."""
    res = minimize_scalar(lambda p: -n * (p - c) * demand(np.full(n, p))[0],
                          bounds=(c, c + 3), method="bounded", options={"xatol": 1e-8})
    return float(res.x)


class BenchmarkTable:
    """Symmetric p_N(c), p_M(c) per n on a cost grid: the regulator's view of the market."""

    def __init__(self, ns: tuple = (2, 3, 4, 5)):
        self.grid = np.linspace(M.c_lo, M.c_hi, M.grid)
        self.pn = {n: np.array([nash_price(n, c) for c in self.grid]) for n in ns}
        self.pm = {n: np.array([monopoly_price(n, c) for c in self.grid]) for n in ns}

    def p_n(self, n: int, c):
        """Nash price for n firms at cost c (scalar or array)."""
        return np.interp(c, self.grid, self.pn[n])

    def p_m(self, n: int, c):
        """Monopoly price for n firms at cost c (scalar or array)."""
        return np.interp(c, self.grid, self.pm[n])


@lru_cache(maxsize=1)
def table() -> BenchmarkTable:
    """Process-wide symmetric benchmark table (built once per process)."""
    return BenchmarkTable()


class RunBenchmarks:
    """True per-firm Nash and joint-monopoly prices of one run, solved on a grid of the common cost."""

    def __init__(self, a: np.ndarray, cost_mult: np.ndarray, mu: float, points: int = 17):
        self.grid = np.linspace(M.c_lo, M.c_hi, points)
        self.pn = np.array([nash_prices(a, c * cost_mult, mu) for c in self.grid])
        self.pm = np.array([monopoly_prices(a, c * cost_mult, mu) for c in self.grid])

    def at(self, c) -> tuple[np.ndarray, np.ndarray]:
        """(p_N, p_M) per firm at common cost c; c scalar -> (n,), c array (T,) -> (T, n)."""
        x = (np.asarray(c, dtype=float) - M.c_lo) / (M.c_hi - M.c_lo) * (len(self.grid) - 1)
        j = np.clip(np.floor(x).astype(int), 0, len(self.grid) - 2)
        w = (x - j)[..., None]
        return (1 - w) * self.pn[j] + w * self.pn[j + 1], (1 - w) * self.pm[j] + w * self.pm[j + 1]


def collusion_index(mean_price, p_n, p_m):
    """Calvano's Delta: 0 = competitive, 1 = monopoly. Reporting only, never a feature."""
    return (mean_price - p_n) / (p_m - p_n)
