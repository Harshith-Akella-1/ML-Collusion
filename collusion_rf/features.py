"""Rolling windows and the 15 behavioural features (computed only from price and observed cost)."""
import numpy as np
import pandas as pd

from .config import FEAT as F

FEATURE_COLS: list[str] = [
    "markup_mean", "markup_std", "rel_price", "price_vol", "rigidity", "corr_mean", "corr_max",
    "lead_lag_max", "sync_up_rate", "punish_response", "n_undercut_events", "market_markup",
    "market_dispersion", "cost_passthrough", "markup_trend"]
META_COLS = ["run_id", "firm", "window_start", "n_firms", "kappa", "in_cartel", "frac_active", "label"]


def _xcorr(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """R[i, j] = corr(X[:, i], Y[:, j]); constant series -> 0; diagonal -> NaN (a firm is not its own rival)."""
    X, Y = X - X.mean(axis=0), Y - Y.mean(axis=0)
    d = np.sqrt((X ** 2).sum(axis=0))[:, None] * np.sqrt((Y ** 2).sum(axis=0))[None, :]
    R = np.divide(X.T @ Y, d, out=np.zeros_like(d), where=d > 1e-12)
    R[np.diag_indices_from(R)] = np.nan
    return R


def _slope(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """OLS slope of each column of y on x (cov/var); 0 if x is (almost) constant."""
    xc = x - x.mean()
    var = xc @ xc
    return xc @ (y - y.mean(axis=0)) / var if var > 1e-12 else np.zeros(y.shape[1])


def window_features(P: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Features for one window. P: (W, n) prices, C: (W,) observed cost. Returns (n, 15) in FEATURE_COLS order."""
    W, n = P.shape
    markup = (P - C[:, None]) / C[:, None]
    dP = np.diff(P, axis=0)
    rival_mean = (P.sum(axis=1, keepdims=True) - P) / (n - 1)
    R, L = _xcorr(dP, dP), _xcorr(dP[1:], dP[:-1])                 # same-step and lead-lag correlations
    up = dP > F.eps
    rival_up = (up.sum(axis=1, keepdims=True) - up) > 0
    # undercut events: a rival's price falls > 3% in one step; response = own change over the next 3 steps
    drop = dP / P[:-1] < -F.undercut
    events = drop.sum(axis=1, keepdims=True) - drop                # (W-1, n) rival drops landing at step s+1
    after = np.minimum(np.arange(1, W) + F.response_steps, W - 1)
    response = P[after] / P[1:] - 1
    n_events = events.sum(axis=0)
    ones = np.ones(n)
    out = np.column_stack([
        markup.mean(axis=0),
        markup.std(axis=0),
        (P / rival_mean).mean(axis=0),
        dP.std(axis=0) / P.mean(axis=0),
        (np.abs(dP) < F.eps).mean(axis=0),
        np.nanmean(R, axis=1),
        np.nanmax(R, axis=1),
        np.nanmax(L, axis=1),
        (up & rival_up).mean(axis=0),
        (events * response).sum(axis=0) / np.maximum(n_events, 1),
        n_events,
        markup.mean() * ones,
        (P.std(axis=1) / P.mean(axis=1)).mean() * ones,
        _slope(dP, np.diff(C)),
        _slope(markup, np.arange(W, dtype=float)),
    ])
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def window_label(in_cartel: np.ndarray, regime: np.ndarray) -> tuple[np.ndarray, float]:
    """Label per firm (cartel member and regime active >= 50% of the window) and the active fraction."""
    frac = float(np.mean(regime))
    return in_cartel & (frac >= 0.5), frac


def build_features(run_df: pd.DataFrame, kappa: float = -1.0) -> pd.DataFrame:
    """Feature table for one run: one row per (firm, window_start). kappa is metadata (-1 = competitive run)."""
    d = run_df.sort_values(["t", "firm"])
    n = int(d["firm"].nunique())
    H = len(d) // n
    P = d["price"].to_numpy().reshape(H, n)
    C = d["cost_obs"].to_numpy().reshape(H, n)[:, 0]
    regime = d["regime_active"].to_numpy().reshape(H, n)[:, 0]
    in_cartel = d["in_cartel"].to_numpy().reshape(H, n)[0].astype(bool)
    run_id, blocks = d["run_id"].iloc[0], []
    for s in range(F.burn_in, H - F.W + 1, F.stride):
        label, frac = window_label(in_cartel, regime[s:s + F.W])
        meta = np.column_stack([np.full(n, run_id), np.arange(n), np.full(n, s), np.full(n, n),
                                np.full(n, kappa), in_cartel, np.full(n, frac), label])
        blocks.append(np.hstack([meta, window_features(P[s:s + F.W], C[s:s + F.W])]))
    out = pd.DataFrame(np.vstack(blocks), columns=META_COLS + FEATURE_COLS)
    return out.astype({"run_id": int, "firm": int, "window_start": int, "n_firms": int,
                       "in_cartel": bool, "label": int})
