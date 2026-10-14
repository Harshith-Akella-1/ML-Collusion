import numpy as np

from collusion_rf.features import FEATURE_COLS, build_features, window_features
from collusion_rf.simulate import sample_scenario, simulate_run


def test_comoving_series_have_corr_one():
    rng = np.random.default_rng(0)
    p1 = 1.5 + np.cumsum(rng.normal(0, 0.01, 50))
    F = window_features(np.column_stack([p1, p1 + 0.2]), np.ones(50))
    assert abs(F[0, FEATURE_COLS.index("corr_max")] - 1) < 1e-9
    assert abs(F[1, FEATURE_COLS.index("corr_mean")] - 1) < 1e-9


def test_constant_series_give_zero_not_nan():
    F = window_features(np.full((50, 3), 1.4), np.ones(50))
    assert np.isfinite(F).all()
    assert F[0, FEATURE_COLS.index("corr_max")] == 0
    assert F[0, FEATURE_COLS.index("rigidity")] == 1


def test_undercut_events_and_response():
    P = np.full((50, 2), 2.0)
    P[20:, 1] = 1.8                      # firm 1 undercuts by 10% at t=20
    P[21:, 0] = 1.5                      # firm 0 answers by cutting 25% one step later
    F = window_features(P, np.ones(50))
    n_ev, resp = FEATURE_COLS.index("n_undercut_events"), FEATURE_COLS.index("punish_response")
    assert F[0, n_ev] == 1 and abs(F[0, resp] + 0.25) < 1e-12
    assert F[1, n_ev] == 1               # firm 0's own cut is an undercut event for firm 1


def test_build_features_shape_and_finiteness():
    assert len(FEATURE_COLS) == 15 and len(set(FEATURE_COLS)) == 15
    horizon, W = 500, 50
    for run_id, cartel in ((3, True), (4, False)):
        sc = sample_scenario(run_id, seed=42, cartel=cartel)
        df = build_features(simulate_run(run_id, sc, horizon, seed=42).df, sc["kappa"] if cartel else -1.0)
        assert set(FEATURE_COLS) <= set(df.columns)
        assert np.isfinite(df[FEATURE_COLS].to_numpy()).all()
        per_firm = df.groupby("firm").size()
        assert len(per_firm) == sc["n_firms"]
        assert (per_firm == (horizon - 50 - W) // 10 + 1).all()
        assert df["window_start"].min() == 50
        assert set(df["label"].unique()) <= {0, 1}
        assert not df.loc[~df["in_cartel"], "label"].any()
        assert ((df["label"] == 1) == (df["in_cartel"] & (df["frac_active"] >= 0.5))).all()
