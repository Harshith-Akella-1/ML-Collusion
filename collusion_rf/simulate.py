"""Scenario sampler and one-run market simulator (with an optional regulator hook)."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .agents import Cartel, CompetitiveFirm, TriggerColluder, step_cost
from .config import AGENT as A, MARKET as M
from .market import RunBenchmarks, best_response, collusion_index, consumer_surplus, demand

# Scenario distributions: False = in-distribution, True = out-of-distribution (history in NOTES.md).
DIST = {
    False: dict(n=(3, 4), kappa=(0.3, 1.0), sigma_p=(0.005, 0.02), sigma_c=(0.002, 0.008), T=(5, 20)),
    True: dict(n=(5,), kappa=(0.1, 0.25), sigma_p=(0.02, 0.035), sigma_c=(0.008, 0.012), T=(20, 35)),
}


def sample_scenario(run_id: int, seed: int, ood: bool = False, cartel: bool | None = None) -> dict:
    """Draw one scenario from the ID/OOD distribution; `cartel` forces presence/absence of a cartel."""
    rng, d = np.random.default_rng([seed, run_id, 0]), DIST[ood]
    n = int(rng.choice(d["n"]))
    has = bool(rng.random() < 0.5)
    k = int(rng.integers(2, n + 1))
    members = sorted(int(i) for i in rng.choice(n, size=k, replace=False))
    return dict(
        n_firms=n, cartel=members if (has if cartel is None else cartel) else [],
        kappa=float(rng.uniform(*d["kappa"])), delta=float(rng.uniform(*A.delta)),
        T=int(rng.integers(d["T"][0], d["T"][1] + 1)), psi=float(rng.uniform(*A.psi)),
        sigma_p=float(rng.uniform(*d["sigma_p"])), sigma_c=float(rng.uniform(*d["sigma_c"])),
        cheat_prob=A.cheat_prob, p_on=A.p_on, p_off=A.p_off, ood=ood,
        # unobserved heterogeneity: firms know these, the regulator assumes a = 2, mu = 0.25, c_i = c_t
        a=rng.uniform(*M.a_range, n).tolist(), cost_mult=rng.uniform(*M.cost_mult, n).tolist(),
        mu=float(rng.uniform(*M.mu_range)))


@dataclass
class History:
    """Per-step arrays of one run; rows up to the current t are filled. Passed to the regulator."""
    n: int
    firms: list
    in_cartel: np.ndarray      # (n,) bool
    bench: RunBenchmarks       # true per-firm Nash / monopoly prices (reporting only)
    price: np.ndarray          # (H, n)
    quantity: np.ndarray       # (H, n)
    profit: np.ndarray         # (H, n)
    cost_true: np.ndarray      # (H,) common cost c_t (firm i pays c_t * cost_mult_i)
    cost_obs: np.ndarray       # (H,)
    cs: np.ndarray             # (H,)
    regime: np.ndarray         # (H,) bool, cartel regime active
    state: list                # cartel state per step


@dataclass
class RunResult:
    df: pd.DataFrame           # long format, one row per (t, firm)
    scenario: dict
    history: History


def simulate_run(run_id: int, scenario: dict, horizon: int = 500, seed: int = 42, regulator=None) -> RunResult:
    """Simulate one market. Environment (costs, regime) and firm noise use separate seeded streams."""
    sc, n = scenario, scenario["n_firms"]
    a, mult, mu = np.array(sc["a"]), np.array(sc["cost_mult"]), sc["mu"]
    env, frng = np.random.default_rng([seed, run_id, 1]), np.random.default_rng([seed, run_id, 2])
    bench = RunBenchmarks(a, mult, mu)
    cartel = Cartel(sc["cartel"], bench, sc["kappa"], sc["delta"], sc["T"], sc["psi"],
                    sc["p_on"], sc["p_off"]) if sc["cartel"] else None
    firms = [TriggerColluder(i, sc["sigma_p"], frng, sc["cheat_prob"]) if i in sc["cartel"]
             else CompetitiveFirm(i, sc["sigma_p"], frng) for i in range(n)]
    h = History(n, firms, np.isin(np.arange(n), sc["cartel"]), bench, *(np.zeros((horizon, n)) for _ in range(3)),
                *(np.zeros(horizon) for _ in range(3)), np.zeros(horizon, dtype=bool), [])
    c = 1.0
    p_prev = np.clip(bench.at(c)[0] + frng.normal(0, 0.005, n), c * mult, c * mult + M.price_cap)
    for t in range(horizon):
        # 1. costs and cartel regime
        c = step_cost(c - 1, sc["sigma_c"], env)
        h.cost_true[t], h.cost_obs[t] = c, c + env.normal(0, M.sigma_obs)
        u = env.random()
        if cartel is not None:
            cartel.update(p_prev, c, u)
        h.regime[t] = cartel is not None and cartel.active
        h.state.append(cartel.state if cartel is not None else "NONE")
        # 2. synchronous price setting from p_{t-1}, with each firm's true cost, quality and mu
        ci = c * mult
        br = best_response(p_prev, ci, a, mu)
        p = np.array([f.price(t, p_prev, br, ci[f.i], cartel) for f in firms])
        # 3. market outcome
        q = demand(p, a, mu)
        h.price[t], h.quantity[t], h.profit[t], h.cs[t] = p, q, (p - ci) * q, consumer_surplus(p, a, mu)
        # 4. regulator may set override flags for future steps
        if regulator is not None:
            regulator.step(t, h)
        p_prev = p
    rep, til = (lambda x: np.repeat(x, n)), (lambda x: np.tile(x, horizon))
    df = pd.DataFrame({
        "run_id": run_id, "t": rep(np.arange(horizon)), "firm": til(np.arange(n)),
        "price": h.price.ravel(), "quantity": h.quantity.ravel(), "profit": h.profit.ravel(),
        "cost_true": rep(h.cost_true), "cost_obs": rep(h.cost_obs), "cs": rep(h.cs),
        "in_cartel": til(h.in_cartel), "regime_active": rep(h.regime), "cartel_state": rep(np.array(h.state))})
    return RunResult(df, sc, h)


def mean_delta(h: History, t0: int = 100) -> float:
    """Mean collusion index over steps >= t0 against the run's true asymmetric Nash / monopoly prices."""
    pn, pm = h.bench.at(h.cost_true[t0:])
    return float(np.mean(collusion_index(h.price[t0:].mean(axis=1), pn.mean(axis=1), pm.mean(axis=1))))
