import numpy as np
import pytest

from collusion_rf.market import (RunBenchmarks, best_response, consumer_surplus, demand, monopoly_price,
                                 monopoly_prices, nash_price, nash_prices, profit, table)


def test_duopoly_benchmarks():
    assert nash_price(2, 1.0) == pytest.approx(1.473, abs=0.005)
    assert monopoly_price(2, 1.0) == pytest.approx(1.925, abs=0.01)


def test_triopoly_benchmarks():
    assert nash_price(3, 1.0) == pytest.approx(1.370, abs=0.005)
    assert monopoly_price(3, 1.0) == pytest.approx(2.00, abs=0.02)


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_nash_below_monopoly(n):
    for c in (0.8, 1.0, 1.2):
        assert nash_price(n, c) < monopoly_price(n, c)
        assert table().p_n(n, c) == pytest.approx(nash_price(n, c), abs=1e-6)
        assert table().p_m(n, c) == pytest.approx(monopoly_price(n, c), abs=1e-5)


def test_asymmetric_benchmarks():
    a, c, mu = np.array([1.8, 2.0, 2.2]), np.array([1.1, 1.0, 0.9]), 0.3
    pn, pm = nash_prices(a, c, mu), monopoly_prices(a, c, mu)
    assert pn == pytest.approx(c + mu / (1 - demand(pn, a, mu)), abs=1e-8)      # every firm's FOC holds
    assert (pn < pm).all()
    joint = profit(pm, c, a, mu).sum()
    rng = np.random.default_rng(0)
    assert all(profit(pm + rng.normal(0, 0.02, 3), c, a, mu).sum() < joint for _ in range(50))
    assert joint > profit(pn, c, a, mu).sum()
    # symmetric special case agrees with the scalar solvers
    assert nash_prices(2.0, np.full(3, 1.0)) == pytest.approx(np.full(3, nash_price(3, 1.0)), abs=1e-9)
    assert monopoly_prices(2.0, np.full(3, 1.0)) == pytest.approx(np.full(3, monopoly_price(3, 1.0)), abs=1e-5)
    # per-run grid interpolates the exact solution
    bench = RunBenchmarks(a, c, mu)
    bn, bm = bench.at(1.03)
    assert bn == pytest.approx(nash_prices(a, 1.03 * c, mu), abs=1e-4)
    assert bm == pytest.approx(monopoly_prices(a, 1.03 * c, mu), abs=1e-4)
    assert bench.at(np.array([0.9, 1.0, 1.1]))[0].shape == (3, 3)


def test_shares_sum_below_one():
    for p in ([1.2, 1.5], [1.4, 1.4, 1.4], [0.9, 2.5, 1.7, 1.3, 1.1]):
        q = demand(np.array(p))
        assert (q > 0).all() and q.sum() < 1


def test_consumer_surplus_falls_when_prices_rise():
    p = np.array([1.4, 1.5, 1.6])
    assert consumer_surplus(p + 0.1) < consumer_surplus(p)


def test_best_response_satisfies_foc_and_nash_is_fixed_point():
    p_prev, c = np.array([1.3, 1.6, 1.9]), 1.05
    br = best_response(p_prev, c)
    for i in range(3):
        p = p_prev.copy()
        p[i] = br[i]
        assert br[i] == pytest.approx(c + 0.25 / (1 - demand(p)[i]), abs=1e-8)
    pn = nash_price(3, c)
    assert best_response(np.full(3, pn), c) == pytest.approx(np.full(3, pn), abs=1e-8)
