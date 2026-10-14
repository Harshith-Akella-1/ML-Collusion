"""Pricing agents: CompetitiveFirm, TriggerColluder and the shared cartel state machine."""
import numpy as np

from .config import AGENT as A, MARKET as M
from .market import RunBenchmarks


def step_cost(s: float, sigma_c: float, rng: np.random.Generator) -> float:
    """One AR(1) step of the common cost c_t = 1 + s_t, clipped to [c_lo, c_hi]."""
    return float(np.clip(1 + M.rho * s + rng.normal(0, sigma_c), M.c_lo, M.c_hi))


class CompetitiveFirm:
    """Noisy, inert best-responder with occasional short promotions."""

    def __init__(self, i: int, sigma_p: float, rng: np.random.Generator):
        self.i, self.sigma_p, self.rng = i, sigma_p, rng
        self.lam = rng.uniform(*A.lam)
        self.promo_left, self.promo_mult = 0, 1.0
        self.forced_competitive_until: int = -1      # regulator override hook
        self.permanently_competitive: bool = False

    def _clip(self, p: float, c: float) -> float:
        return float(np.clip(p, c, c + M.price_cap))

    def competitive_price(self, p_prev: np.ndarray, br: np.ndarray, c: float) -> float:
        """Partial adjustment towards the best response, plus noise and promotions."""
        p = (1 - self.lam) * p_prev[self.i] + self.lam * br[self.i] + self.rng.normal(0, self.sigma_p)
        if self.promo_left == 0 and self.rng.random() < A.promo_prob:
            self.promo_left = int(self.rng.integers(1, 4))
            self.promo_mult = self.rng.uniform(*A.promo_depth)
        if self.promo_left > 0:
            self.promo_left -= 1
            p *= self.promo_mult
        return self._clip(p, c)

    def price(self, t: int, p_prev: np.ndarray, br: np.ndarray, c: float, cartel=None) -> float:
        """Price for step t given last period's prices, the best-response vector and the firm's own cost c."""
        return self.competitive_price(p_prev, br, c)


class Cartel:
    """Shared cartel state: Markov regime (active/inactive) and COLLUDE/PUNISH phases."""

    def __init__(self, members: list, bench: RunBenchmarks, kappa: float, delta: float, T: int, psi: float,
                 p_on: float = A.p_on, p_off: float = A.p_off):
        self.members, self.bench = np.asarray(members, dtype=int), bench
        self.kappa, self.delta, self.T, self.psi, self.p_on, self.p_off = kappa, delta, T, psi, p_on, p_off
        self.active, self.punish_left, self.grace = False, 0, 0
        self.p_n = self.p_c = np.zeros(bench.pn.shape[1])   # per-firm Nash price and collusive target

    @property
    def state(self) -> str:
        if not self.active:
            return "INACTIVE"
        return "PUNISH" if self.punish_left > 0 else "COLLUDE"

    def update(self, p_prev: np.ndarray, c: float, u: float) -> None:
        """Advance regime and phase for this step; u is a U[0,1] draw for the regime chain."""
        prev_pc, was_active = self.p_c, self.active
        self.p_n, p_m = self.bench.at(c)
        self.p_c = self.p_n + self.kappa * (p_m - self.p_n)
        self.active = (u >= self.p_off) if was_active else (u < self.p_on)
        if not self.active:
            self.punish_left = self.grace = 0          # going inactive resets punishment
        elif not was_active:
            self.grace = A.grace
        elif self.punish_left > 0:
            self.punish_left -= 1
            if self.punish_left == 0:
                self.grace = A.grace
        elif self.grace > 0:
            self.grace -= 1
        elif (p_prev < prev_pc * (1 - self.delta))[self.members].any():   # members watch members only
            self.punish_left = self.T


class TriggerColluder(CompetitiveFirm):
    """Cartel member running a forgiving trigger strategy."""

    def __init__(self, i: int, sigma_p: float, rng: np.random.Generator, cheat_prob: float = A.cheat_prob):
        super().__init__(i, sigma_p, rng)
        self.cheat_prob, self.cheat_left = cheat_prob, 0

    def price(self, t: int, p_prev: np.ndarray, br: np.ndarray, c: float, cartel: Cartel = None) -> float:
        """Collusive, punishment or competitive price depending on cartel state and overrides."""
        state = cartel.state
        if state == "INACTIVE" or self.permanently_competitive or t < self.forced_competitive_until:
            self.cheat_left = 0
            return self.competitive_price(p_prev, br, c)
        noise, p_c = self.rng.normal(0, self.sigma_p), cartel.p_c[self.i]
        if state == "PUNISH":
            self.cheat_left = 0
            return self._clip(cartel.p_n[self.i] * (1 - cartel.psi) + noise, c)
        if self.cheat_left == 0 and cartel.grace == 0 and self.rng.random() < self.cheat_prob:
            self.cheat_left = int(self.rng.integers(1, 4))
        if self.cheat_left > 0:
            self.cheat_left -= 1
            return self._clip(A.cheat_mult * p_c, c)
        return self._clip((1 - self.lam) * p_prev[self.i] + self.lam * p_c + noise, c)
