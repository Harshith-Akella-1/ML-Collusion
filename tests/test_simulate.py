import numpy as np
import pandas as pd

from collusion_rf.simulate import mean_delta, sample_scenario, simulate_run


def _scenario(cartel, **kw):
    sc = dict(n_firms=3, cartel=cartel, kappa=0.8, delta=0.08, T=10, psi=0.02, sigma_p=0.005,
              sigma_c=0.004, cheat_prob=0.004, p_on=0.01, p_off=0.005, ood=False,
              a=[2.0, 2.0, 2.0], cost_mult=[1.0, 1.0, 1.0], mu=0.25)
    sc.update(kw)
    return sc


HETERO = dict(a=[1.85, 2.0, 2.2], cost_mult=[1.1, 0.95, 0.9], mu=0.3)


def test_delta_uses_true_asymmetric_benchmarks():
    comp = simulate_run(6, _scenario([], sigma_p=0.01, **HETERO), seed=7)
    cart = simulate_run(7, _scenario([0, 1, 2], cheat_prob=0.0, p_on=1.0, p_off=0.0, **HETERO), seed=7)
    assert -0.15 <= mean_delta(comp.history) <= 0.2
    assert 0.6 <= mean_delta(cart.history) <= 0.9


def test_sampled_heterogeneity_is_in_range():
    for rid in range(30):
        sc = sample_scenario(rid, seed=1)
        assert len(sc["a"]) == len(sc["cost_mult"]) == sc["n_firms"]
        assert all(1.8 <= x <= 2.2 for x in sc["a"]) and all(0.9 <= x <= 1.1 for x in sc["cost_mult"])
        assert 0.18 <= sc["mu"] <= 0.32 and 0.3 <= sc["kappa"] <= 1.0
    assert all(0.1 <= sample_scenario(r, 1, ood=True)["kappa"] <= 0.25 for r in range(30))


def test_competitive_run_has_delta_near_zero():
    res = simulate_run(1, _scenario([], sigma_p=0.01), seed=7)
    assert -0.15 <= mean_delta(res.history) <= 0.2
    assert not res.df["regime_active"].any() and not res.df["in_cartel"].any()


def test_full_cartel_forced_active_has_high_delta():
    res = simulate_run(2, _scenario([0, 1, 2], cheat_prob=0.0, p_on=1.0, p_off=0.0), seed=7)
    assert 0.6 <= mean_delta(res.history) <= 0.9
    assert res.df["regime_active"].all() and res.df["in_cartel"].all()


def test_same_seed_identical_different_seed_differs():
    sc = sample_scenario(5, seed=42, cartel=True)
    a, b, c = (simulate_run(5, sc, seed=s).df for s in (42, 42, 43))
    pd.testing.assert_frame_equal(a, b)
    assert not a["price"].equals(c["price"])


def test_dataframe_schema_and_bounds():
    sc = sample_scenario(11, seed=42, cartel=True)
    df = simulate_run(11, sc, horizon=200, seed=42).df
    assert list(df.columns) == ["run_id", "t", "firm", "price", "quantity", "profit", "cost_true",
                                "cost_obs", "cs", "in_cartel", "regime_active", "cartel_state"]
    assert len(df) == 200 * sc["n_firms"]
    assert df["cost_true"].between(0.8, 1.2).all()
    own_cost = df["cost_true"] * df["firm"].map(dict(enumerate(sc["cost_mult"])))
    assert (df["price"] >= own_cost).all() and (df["price"] <= own_cost + 3).all()
    assert np.allclose(df["profit"], (df["price"] - own_cost) * df["quantity"])
    assert set(df.loc[df["in_cartel"], "firm"]) == set(sc["cartel"])


def test_forced_competitive_colluder_stops_colluding():
    sc = _scenario([0, 1, 2], cheat_prob=0.0, p_on=1.0, p_off=0.0)

    class ForceAll:
        def step(self, t, history):
            if t == 0:
                for f in history.firms:
                    f.permanently_competitive = True

    res = simulate_run(3, sc, seed=7, regulator=ForceAll())
    assert -0.15 <= mean_delta(res.history) <= 0.2
