"""All parameters in one place (SPEC sections 3-10)."""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class MarketConfig:
    a: float = 2.0            # product quality assumed by the regulator
    a0: float = 0.0           # outside-good quality
    mu: float = 0.25          # horizontal differentiation assumed by the regulator
    a_range: tuple = (1.8, 2.2)         # true a_i ~ U per firm per run (unobserved)
    cost_mult: tuple = (0.9, 1.1)       # true c_i = c_t * U per firm per run (unobserved)
    mu_range: tuple = (0.18, 0.32)      # true mu ~ U per run (unobserved)
    c_lo: float = 0.8         # cost bounds (also the benchmark grid)
    c_hi: float = 1.2
    grid: int = 81
    rho: float = 0.95         # AR(1) persistence of the common cost shock
    sigma_obs: float = 0.05   # regulator's cost observation noise
    price_cap: float = 3.0    # prices clipped to [c, c + price_cap]


@dataclass(frozen=True)
class AgentConfig:
    lam: tuple = (0.3, 0.8)           # price inertia lambda_i ~ U
    promo_prob: float = 0.02
    promo_depth: tuple = (0.85, 0.95)
    delta: tuple = (0.03, 0.08)       # cartel tolerance
    psi: tuple = (0.0, 0.05)          # punishment depth
    cheat_prob: float = 0.004
    cheat_mult: float = 0.95
    p_on: float = 0.01                # P(INACTIVE -> ACTIVE)
    p_off: float = 0.005              # P(ACTIVE -> INACTIVE)
    grace: int = 10                   # unmonitored ramp-up steps after entering COLLUDE (see NOTES.md)


@dataclass(frozen=True)
class FeatureConfig:
    W: int = 50
    stride: int = 10
    burn_in: int = 50
    eps: float = 0.002
    undercut: float = 0.03
    response_steps: int = 3


@dataclass(frozen=True)
class RunConfig:
    seed: int = 42
    horizon: int = 500
    n_id: int = 800           # run_ids 0..n_id-1
    n_ood: int = 200          # run_ids 10000..
    n_policy: int = 100       # run_ids 20000.. (cartel) and 30000.. (competitive)
    rf_trees: int = 300
    quick: bool = False
    out: str = "outputs"

    def as_quick(self) -> "RunConfig":
        """Smoke-test sizes (SPEC section 11)."""
        return replace(self, n_id=120, n_ood=30, n_policy=20, rf_trees=100, quick=True)


MARKET = MarketConfig()
AGENT = AgentConfig()
FEAT = FeatureConfig()
OOD_ID0, POLICY_ID0, COMP_ID0 = 10000, 20000, 30000
