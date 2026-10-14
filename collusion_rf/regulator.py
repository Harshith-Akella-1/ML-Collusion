"""Online regulator (detect -> fine -> force competitive) and the paired policy comparison."""
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import COMP_ID0, POLICY_ID0, RunConfig
from .evaluate import _save, plot_prices
from .features import window_features, window_label
from .market import table
from .simulate import History, mean_delta, sample_scenario, simulate_run


class Regulator:
    """Checks every firm on the last W steps, flags after k consecutive suspicious checks, fines and remedies."""

    def __init__(self, model, tau: float, mode: Literal["ml", "oracle"], W: int = 50, check_every: int = 10,
                 k_consecutive: int = 3, fine_multiplier: float = 1.5, remedy_steps: int = 100):
        self.model, self.tau, self.mode, self.W, self.check_every = model, tau, mode, W, check_every
        self.k, self.fine_multiplier, self.remedy_steps = k_consecutive, fine_multiplier, remedy_steps
        self.streak, self.n_flags, self.flags = {}, {}, []

    def step(self, t: int, history: History) -> None:
        """Called once per simulation step; may set override flags on firms."""
        if t < self.W or t % self.check_every:
            return
        h, w = history, slice(t - self.W + 1, t + 1)
        P, C = h.price[w], h.cost_obs[w]
        truth, _ = window_label(h.in_cartel, h.regime[w])
        if self.mode == "oracle":
            suspicious = truth
        else:
            suspicious = self.model.predict_proba(window_features(P, C))[:, 1] >= self.tau
        p_hat_n = table().p_n(h.n, C)                         # competitive price as estimated from c_obs
        for i, firm in enumerate(h.firms):
            if firm.permanently_competitive or t < firm.forced_competitive_until:
                continue                                      # already under remedy: not re-examined
            self.streak[i] = self.streak.get(i, 0) + 1 if suspicious[i] else 0
            if self.streak[i] < self.k:
                continue
            self.streak[i], self.n_flags[i] = 0, self.n_flags.get(i, 0) + 1
            fine = self.fine_multiplier * float((np.maximum(0, P[:, i] - p_hat_n) * h.quantity[w, i]).sum())
            self.flags.append({"t": t, "firm": i, "fine": fine, "guilty": bool(truth[i])})
            firm.forced_competitive_until = t + self.remedy_steps
            firm.permanently_competitive = self.n_flags[i] >= 2


@lru_cache(maxsize=2)
def _load_model(path: str):
    """Model cached per worker process; single-threaded prediction (tiny batches)."""
    model = joblib.load(path)
    if "n_jobs" in model.get_params():
        model.set_params(n_jobs=1)
    return model


def policy_run(run_id: int, cfg: RunConfig, mode: str, cartel: bool, tau: float, with_history: bool = False):
    """One run under policy `mode` in {none, ml, oracle}; returns a summary row (and optionally the history)."""
    path = str(Path(cfg.out) / "models" / "best_model.joblib")
    reg = None if mode == "none" else Regulator(_load_model(path) if mode == "ml" else None, tau, mode)
    h = simulate_run(run_id, sample_scenario(run_id, cfg.seed, cartel=cartel), cfg.horizon, cfg.seed, reg).history
    flags = reg.flags if reg else []
    active = np.flatnonzero(h.regime)
    first = int(active[0]) if len(active) else None
    hits = [f["t"] for f in flags if h.in_cartel[f["firm"]] and first is not None and f["t"] >= first]
    row = {"run_id": run_id, "policy": mode if cartel else f"{mode}_competitive_runs",
           "delta": mean_delta(h), "cs": float(h.cs[100:].mean()),
           "guilty_fines": sum(f["fine"] for f in flags if f["guilty"]),
           "innocent_fines": sum(f["fine"] for f in flags if not f["guilty"]),
           "n_flags": len(flags), "activated": first is not None,
           "delay": hits[0] - first if hits else np.nan,
           "permanent": any(f.permanently_competitive for f, m in zip(h.firms, h.in_cartel) if m)}
    return (row, h, flags) if with_history else row


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean +- std per policy over runs, plus detection-delay and remedy statistics."""
    g = runs.groupby("policy", sort=False)
    out = g[["delta", "cs", "guilty_fines", "innocent_fines"]].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n_runs"] = g.size()
    out["median_delay"] = g["delay"].median()
    out["never_flagged"] = g.apply(lambda d: int((d["activated"] & d["delay"].isna()).sum()), include_groups=False)
    out["frac_permanent"] = g["permanent"].mean()
    return out.reset_index()


def penalise(cfg: RunConfig) -> pd.DataFrame:
    """Paired comparison none / ml / oracle on fresh cartel runs, plus ml on competitive-only runs."""
    import json
    tau = json.loads((Path(cfg.out) / "models" / "threshold.json").read_text())["tau"]
    jobs = [(POLICY_ID0 + r, mode, True) for mode in ("none", "ml", "oracle") for r in range(cfg.n_policy)]
    jobs += [(COMP_ID0 + r, "ml", False) for r in range(cfg.n_policy)]
    # 4 workers only: each holds its own copy of the model (16 forest copies ran out of memory)
    rows = Parallel(n_jobs=4)(delayed(policy_run)(rid, cfg, mode, cartel, tau) for rid, mode, cartel in jobs)
    runs = pd.DataFrame(rows)
    res = summarise(runs)
    rep = Path(cfg.out) / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    runs.to_csv(rep / "penalty_runs.csv", index=False)
    res.to_csv(rep / "penalty_results.csv", index=False)

    main = res[~res["policy"].str.contains("competitive")]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, col, title in ((axes[0], "delta", "Collusion index (lower = more competitive)"),
                           (axes[1], "cs", "Consumer surplus (higher = better)")):
        ax.bar(main["policy"], main[f"{col}_mean"], yerr=main[f"{col}_std"], capsize=4,
               color=["tab:gray", "tab:blue", "tab:green"])
        ax.set_title(title, fontsize=9)
    lo = (main["cs_mean"] - main["cs_std"]).min()
    axes[1].set_ylim(bottom=lo - 0.02)
    _save(fig, cfg, "policy_comparison.png")

    ml = runs[runs["policy"] == "ml"]
    example = int(ml.loc[ml["guilty_fines"].idxmax(), "run_id"])   # most heavily fined cartel run
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, sharey=True)
    for ax, mode in zip(axes, ("none", "ml")):
        _, h, flags = policy_run(example, cfg, mode, True, tau, with_history=True)
        plot_prices(ax, h, f"run {example}, policy = {mode}  (dotted: green = guilty flag, orange = innocent)", flags)
    _save(fig, cfg, "penalty_example.png")

    print("[penalise] policy                 delta(mean+-std)  CS(mean+-std)    guilty fines  innocent fines  "
          "median delay  never flagged  frac permanent")
    for r in res.itertuples():
        print(f"[penalise] {r.policy:<22} {r.delta_mean:6.3f} +- {r.delta_std:.3f}  {r.cs_mean:.4f} +- "
              f"{r.cs_std:.4f}  {r.guilty_fines_mean:10.3f}  {r.innocent_fines_mean:14.3f}  "
              f"{r.median_delay:12.0f}  {r.never_flagged:13d}  {r.frac_permanent:14.2f}")
    return res
