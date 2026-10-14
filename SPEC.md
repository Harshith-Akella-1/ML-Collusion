# SPEC — Algorithmic Collusion Detection & Penalisation (single collusion type)

> **For Claude Code.** Every design decision is already made in this document. Your job is to implement it exactly, keep the package **≤ 1000 lines of Python** (tests excluded), and make every acceptance check in §12 pass. If something is ambiguous, choose the simplest option that matches the spirit of this spec and write the choice in `NOTES.md`.

---

## 0. One-paragraph summary

We simulate a repeated price-setting market (logit-demand Bertrand oligopoly, as in Calvano et al. 2020). Some markets contain a **cartel** of firms that run **one collusion type: a forgiving trigger strategy** — they hold prices above the competitive level and punish any cartel member who undercuts. Other firms price competitively. A **regulator** only sees prices and noisy cost estimates. We cut the price history into rolling windows, compute 15 behavioural features per *(firm, window)*, and train classifiers (logistic regression baseline, random forest, gradient boosting) to predict *"this firm is colluding in this window."* We evaluate with run-level splits and an out-of-distribution test set. Finally, we plug the best model into an **online regulator** that flags firms, fines them and forces flagged colluders to stop colluding, and we measure the change in prices and consumer surplus against "no regulator" and a perfect "oracle regulator."

---

## 1. Tech constraints

- Python 3.10+.
- Dependencies (put in `requirements.txt`): `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `joblib`, `pyarrow`, `pytest`.
- No deep learning, no notebooks, no RL.
- Type hints on public functions; one-line docstrings are enough.
- Every source of randomness uses `numpy.random.Generator` seeded from `(global_seed, run_id)` so results are fully reproducible.
- Package LOC budget: **≤ 1000** (count with `find collusion_rf -name "*.py" | xargs wc -l`). Tests do not count.

---

## 2. Repository layout and LOC budget

```
collusion-detection/
├── SPEC.md                 (this file)
├── NOTES.md                (your implementation notes / deviations)
├── README.md               (how to run, results table filled in after `all`)
├── requirements.txt
├── collusion_rf/
│   ├── __init__.py
│   ├── config.py           ~70   dataclasses for all parameters
│   ├── market.py           ~110  demand, profit, best response, Nash/monopoly solvers, consumer surplus
│   ├── agents.py           ~120  CompetitiveFirm, TriggerColluder, cartel regime process
│   ├── simulate.py         ~110  scenario sampler + one-run simulator (supports an optional regulator hook)
│   ├── features.py         ~130  windowing + 15 features
│   ├── dataset.py          ~70   generate many runs → feature tables, run-level splits
│   ├── train.py            ~110  3 models, GroupKFold tuning, threshold selection, save
│   ├── evaluate.py         ~110  metrics, subgroup analysis, figures
│   ├── regulator.py        ~110  online detection, sanctions, policy comparison
│   └── main.py             ~60   CLI
├── tests/
│   ├── test_market.py
│   ├── test_simulate.py
│   ├── test_features.py
│   └── test_dataset.py
└── outputs/                (created at runtime, git-ignored)
    ├── data/  models/  reports/  figures/
```

---

## 3. The market (`market.py`)

### 3.1 Logit demand

For `n` firms with prices `p` (vector), quality `a_i = 2.0`, outside-good quality `a_0 = 0.0`, horizontal differentiation `mu = 0.25`:

```
q_i(p) = exp((a_i - p_i)/mu) / ( Σ_j exp((a_j - p_j)/mu) + exp(a_0/mu) )
profit_i = (p_i - c) * q_i
```

Consumer surplus per period (market size normalised to 1):

```
CS(p) = mu * log( Σ_j exp((a_j - p_j)/mu) + exp(a_0/mu) )
```

### 3.2 Best response

Logit FOC gives the fixed-point `p_i = c + mu / (1 - q_i(p_i, p_-i))`.
Implement `best_response(p_prev: np.ndarray, c: float) -> np.ndarray` that returns the BR **for all firms at once**, each holding rivals at `p_prev`. Solve per firm by fixed-point iteration (start at `c + mu`, 30 iterations, vectorised over firms). Convergence tolerance 1e-8 is fine.

### 3.3 Benchmarks

- `nash_price(n, c)`: iterate the symmetric BR from `p = c + mu` until convergence.
- `monopoly_price(n, c)`: maximise joint symmetric profit `n * (p - c) * q(p)` with `scipy.optimize.minimize_scalar(bounds=(c, c+3), method="bounded")`.
- `BenchmarkTable`: precompute `p_N(c)` and `p_M(c)` for each `n ∈ {2,3,4,5}` on a grid `c ∈ [0.8, 1.2]` (81 points) and interpolate with `np.interp`. Agents and the regulator use this table (computing solvers every step is too slow).

### 3.4 Collusion index (Calvano's Δ)

```
Δ = (mean_price - p_N) / (p_M - p_N)
```
Δ ≈ 0 means competitive, Δ ≈ 1 means full monopoly pricing. Used for **reporting only, never as a feature** (the regulator does not know true costs).

**Sanity values (c = 1):** n=2 → p_N ≈ 1.473, p_M ≈ 1.925. n=3 → p_N ≈ 1.370, p_M ≈ 2.00.

---

## 4. Agents (`agents.py`)

### 4.1 Costs (shared by all firms)

AR(1) common cost shock: `c_t = 1 + s_t`, `s_t = 0.95 * s_{t-1} + N(0, σ_c)`, clipped so `c_t ∈ [0.8, 1.2]`. Firms know `c_t` exactly. The regulator observes `c_obs_t = c_t + N(0, σ_obs)` with `σ_obs = 0.05`.

### 4.2 CompetitiveFirm

Each step:
```
target = best_response(p_{t-1}, c_t)[i]
p_i,t  = (1 - λ_i) * p_i,t-1 + λ_i * target + N(0, σ_p)
```
- `λ_i ~ U[0.3, 0.8]` per firm per run (price inertia).
- **Promotions:** with probability 0.02 per step, the firm starts a promotion lasting 1–3 steps, during which its price is multiplied by `U[0.85, 0.95]`. (Creates undercut events in honest markets so "price drops" alone are not evidence.)
- Clip all prices to `[c_t, c_t + 3]`.

### 4.3 TriggerColluder (the single collusion type)

Parameters per run (shared by cartel members): collusion strength `κ`, tolerance `δ ~ U[0.03, 0.08]`, punishment length `T ~ randint[5, 20]`, punishment depth `ψ ~ U[0.0, 0.05]`, cheat probability `0.004` per step per member.

Collusive target: `p_C,t = p_N(c_t) + κ * (p_M(c_t) - p_N(c_t))` (from the benchmark table, using the run's `n`).

State machine per cartel (shared state, not per firm):

| State | Behaviour of each cartel member |
|---|---|
| `INACTIVE` (cartel regime off) | behaves exactly like a `CompetitiveFirm` |
| `COLLUDE` | `p = (1-λ) p_prev + λ p_C + N(0, σ_p)` |
| `PUNISH` (k steps left) | `p = p_N(c_t) * (1 - ψ) + N(0, σ_p)` |

Transitions:
- **Monitoring:** cartel members watch **only other cartel members**. If any member's previous price `< p_C,t-1 * (1 - δ)` while in `COLLUDE`, switch to `PUNISH` for `T` steps, then back to `COLLUDE`.
- **Cheating:** in `COLLUDE`, each member independently cheats with prob 0.004: prices at `0.95 * p_C` for 1–3 steps (this triggers punishment next step).
- **Regime switching:** a 2-state Markov chain decides whether the cartel is active. Start `INACTIVE`. `P(INACTIVE→ACTIVE) = 0.01`, `P(ACTIVE→INACTIVE) = 0.005` per step. Going inactive resets punishment.
- `λ_i` sampled as for competitive firms.

**Intended realism:** competitive outsiders in a partial cartel will best-respond to high cartel prices and therefore also price high ("umbrella pricing"). They are **not** colluding and must be labelled 0. This is the main hard negative.

### 4.4 Regulator override hook

Each firm object has `forced_competitive_until: int = -1`. If `t < forced_competitive_until`, a colluder behaves as a `CompetitiveFirm` regardless of cartel state (its cartel-mates will see it as a deviator and punish — that is intended). Also `permanently_competitive: bool = False`.

---

## 5. Scenarios and simulation (`simulate.py`)

### 5.1 Scenario distributions

| Parameter | In-distribution (ID) | Out-of-distribution (OOD) |
|---|---|---|
| `n_firms` | {3, 4} uniform | {5} |
| P(run has a cartel) | 0.5 | 0.5 |
| cartel size `k` | uniform over {2,…,n} | uniform over {2,…,n} |
| `κ` | U[0.4, 1.0] | U[0.15, 0.35] |
| `σ_p` | U[0.005, 0.02] | U[0.02, 0.035] |
| `σ_c` | U[0.002, 0.008] | U[0.008, 0.012] |
| `T` | randint[5, 20] | randint[20, 35] |

Cartel members are a random subset of firm indices.

### 5.2 One run

`simulate_run(run_id, scenario, horizon=500, seed, regulator=None) -> RunResult`

Initial prices: `p_N(c_0)` + small noise. Step order each `t`:
1. update cost `c_t`, `c_obs_t`; update cartel regime
2. every firm chooses `p_i,t` from `p_{t-1}` (synchronous moves)
3. compute `q`, profits, `CS_t`
4. if `regulator` is given, call `regulator.step(t, history)` (it may set override flags)

`RunResult` holds a long-format `DataFrame` with columns:
`run_id, t, firm, price, quantity, profit, cost_true, cost_obs, cs, in_cartel (bool), regime_active (bool), cartel_state (str)`
plus the scenario dict.

---

## 6. Features (`features.py`)

### 6.1 Windows

Window length `W = 50`, stride `10`, burn-in: ignore `t < 50`. Unit of prediction = **(run_id, firm, window_start)**.

### 6.2 Label

`label = 1` iff the firm is a cartel member **and** the cartel regime is active for ≥ 50% of the window's steps. Also store `frac_active` (float) for analysis.

### 6.3 The 15 features (computed only from `price` and `cost_obs`)

Notation: `Δp` = first difference, rivals = all other firms, ε = 0.002.

| # | Name | Definition |
|---|---|---|
| 1 | `markup_mean` | mean of `(p_i - c_obs)/c_obs` |
| 2 | `markup_std` | std of the same |
| 3 | `rel_price` | mean of `p_i / mean_{j≠i}(p_j)` |
| 4 | `price_vol` | `std(Δp_i) / mean(p_i)` |
| 5 | `rigidity` | fraction of steps with `|Δp_i| < ε` |
| 6 | `corr_mean` | mean over rivals of `corr(Δp_i, Δp_j)` |
| 7 | `corr_max` | max over rivals of the same |
| 8 | `lead_lag_max` | max over rivals of `corr(Δp_i[t], Δp_j[t-1])` |
| 9 | `sync_up_rate` | fraction of steps where `Δp_i > ε` and some rival has `Δp_j > ε` |
| 10 | `punish_response` | over "undercut events" (a rival's price falls >3% in one step), mean of firm i's relative price change over the next 3 steps; 0 if no events |
| 11 | `n_undercut_events` | number of such events in the window |
| 12 | `market_markup` | mean markup over all firms (market context) |
| 13 | `market_dispersion` | mean over t of `std_j(p_j)/mean_j(p_j)` |
| 14 | `cost_passthrough` | OLS slope of `Δp_i` on `Δc_obs` (`cov/var`; 0 if var≈0) |
| 15 | `markup_trend` | OLS slope of firm markup vs. t within the window |

Rules: any NaN correlation (constant series) → 0. Final table must contain **no NaN/inf**. Vectorise with numpy per window where reasonable; correctness beats speed.

`build_features(run_df) -> pd.DataFrame` with columns `run_id, firm, window_start, n_firms, kappa, in_cartel, frac_active, label, <15 features>`. (`kappa` is stored for analysis only; it is NOT a model input. `kappa = NaN` → store as -1 for competitive runs.)

Also export `FEATURE_COLS: list[str]` (exactly the 15 features) — the only columns models may see.

---

## 7. Dataset (`dataset.py`)

- ID: **800 runs** (run_ids 0–799). OOD: **200 runs** (run_ids 10000–10199).
- Use `joblib.Parallel(n_jobs=-1)` over runs. Target total generation time < 5 min on a laptop.
- **Split by run_id only:** ID runs → train 70% / val 15% / test 15% (stratified by "has cartel"). Never split rows of one run across sets.
- Save `outputs/data/features_id.parquet`, `features_ood.parquet`, `splits.json`.
- Expected positive rate ≈ 10–25% of rows. Print it.

---

## 8. Training (`train.py`)

Three models, all `sklearn`:

1. **Logistic regression** (baseline): `StandardScaler` + `LogisticRegression(class_weight="balanced", max_iter=2000)`.
2. **Random forest**: `RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=seed)`. Grid: `n_estimators ∈ {300}`, `max_depth ∈ {8, 16, None}`, `min_samples_leaf ∈ {1, 5, 20}`.
3. **Gradient boosting**: `HistGradientBoostingClassifier(class_weight="balanced")`. Grid: `learning_rate ∈ {0.05, 0.1}`, `max_depth ∈ {3, 6}`.

Tuning: `GridSearchCV` on the **train split** with `GroupKFold(n_splits=5)`, `groups=run_id`, scoring `average_precision`.

Model selection: pick the model with the highest **validation PR-AUC**.

Threshold: on validation predictions, choose τ that maximises **F0.5** (precision-weighted — wrongly accusing an innocent firm is worse than missing a guilty one). Save `outputs/models/best_model.joblib`, `outputs/models/threshold.json` (`{"model": name, "tau": float}`), and all three models.

---

## 9. Evaluation (`evaluate.py`)

On **ID test** and **OOD** sets, for each model report: ROC-AUC, PR-AUC, and for the best model at τ: precision, recall, F1, F0.5, confusion matrix. Save `outputs/reports/metrics.json` and print a table.

Subgroup analysis for the best model (ID test):
- recall by `κ` bucket: [0.4,0.6), [0.6,0.8), [0.8,1.0]
- **false positive rate for "umbrella" firms** (label 0, `in_cartel=False`, in a cartel run) vs. firms in fully competitive runs
- FPR for cartel members in inactive regime windows

Figures (`outputs/figures/`):
1. `pr_curves.png` — PR curves of the 3 models (ID test) + best model on OOD
2. `roc_curves.png`
3. `confusion_matrix.png` (best model, ID test)
4. `feature_importance.png` — **permutation importance** on ID test (PR-AUC scoring, 5 repeats)
5. `example_runs.png` — prices over time for one competitive run and one cartel run, shading active-cartel periods and punishment phases
6. `recall_by_kappa.png`

---

## 10. Penalisation (`regulator.py`)

### 10.1 Online regulator

```python
class Regulator:
    def __init__(self, model, tau, mode: Literal["ml", "oracle"], W=50, check_every=10,
                 k_consecutive=3, fine_multiplier=1.5, remedy_steps=100): ...
    def step(self, t, history) -> None: ...
```

Every `check_every` steps once `t ≥ W`:
1. Build the 15 features for each firm on the last `W` steps (reuse `features.py` single-window function).
2. `ml` mode: `prob = model.predict_proba(...)`; firm is "suspicious" if `prob ≥ τ`.
   `oracle` mode: suspicious iff the true label of that window is 1.
3. A firm is **flagged** after `k_consecutive` consecutive suspicious checks (then its counter resets).

On flag:
- **Fine** = `fine_multiplier × Σ_{last W steps} max(0, p_i - p̂_N) × q_i`, where `p̂_N` = Nash price from the benchmark table evaluated at `c_obs` (the regulator's estimate of the competitive price). Record whether the firm was truly colluding at that moment (guilty vs. innocent fine).
- **Remedy:** set `forced_competitive_until = t + remedy_steps`. On the firm's **second** flag set `permanently_competitive = True`.
- Innocent (non-cartel) firms are fined but their behaviour is unaffected (they were already competitive). This is the false-positive cost.

### 10.2 Policy comparison

Generate **100 fresh ID cartel runs** (run_ids 20000–20099, all with a cartel) and simulate each **three times with the identical seed**: `none`, `ml`, `oracle` (paired comparison). Also run **100 fresh competitive-only runs** under `ml` to measure fines on innocent markets.

Report per policy (mean ± std over runs, measured on steps ≥ 100):
- mean collusion index Δ (using true `p_N`, `p_M`)
- mean consumer surplus
- total fines collected from guilty firms / innocent firms
- detection delay: steps from first cartel activation to the first flag of a cartel member (runs never flagged → report count separately)
- fraction of runs where any cartel member is permanently competitive by t=500

Save `outputs/reports/penalty_results.csv` and figures:
7. `policy_comparison.png` — bars for Δ and CS per policy with error bars
8. `penalty_example.png` — one run's prices under `none` vs `ml`, with flag times marked

---

## 11. CLI (`main.py`)

```
python -m collusion_rf.main generate     # simulate + features + splits
python -m collusion_rf.main train
python -m collusion_rf.main evaluate
python -m collusion_rf.main penalise
python -m collusion_rf.main all          # everything in order
python -m collusion_rf.main all --quick  # 120 ID runs, 30 OOD runs, 20 policy runs (for smoke tests)
```

Global `--seed` (default 42). Each step prints a short summary.

### Expected console output (`all`, values approximate)

```
[generate] benchmarks: n=3 p_N=1.370 p_M=2.000 | n=4 p_N≈1.31 p_M≈2.07
[generate] ID runs: 800 (cartel runs: ~400)  rows: ~115,000  positive rate: ~0.17
[generate] OOD runs: 200  rows: ~41,000  positive rate: ~0.15
[generate] splits by run_id: train 560 / val 120 / test 120 runs  (no overlap ✔)
[train] logreg   val PR-AUC 0.6x
[train] rf       val PR-AUC 0.8x   best params {max_depth: 16, min_samples_leaf: 5}
[train] hgb      val PR-AUC 0.8x
[train] selected: rf   tau (F0.5-optimal) = 0.6x
[evaluate] ID test  : ROC-AUC 0.9x  PR-AUC 0.8x  precision 0.8x  recall 0.6x-0.7x
[evaluate] OOD      : ROC-AUC 0.7x-0.8x  PR-AUC 0.5x-0.6x
[evaluate] FPR umbrella firms 0.0x vs competitive-run firms 0.0x
[evaluate] figures written to outputs/figures/
[penalise] policy  Δ(mean)   CS(mean)  guilty fines  innocent fines  median delay
[penalise] none    ~0.45     lowest     0             0              -
[penalise] ml      ~0.2-0.3  higher     >0            small          ~60-120 steps
[penalise] oracle  ~0.1-0.2  highest    >0            0              ~30-50 steps
```

**How to react to results:** if ID test PR-AUC > 0.98, the task is too easy — raise `σ_p` range / lower `κ` range and note it in `NOTES.md`. If < 0.5, a feature is probably buggy — check §12 tests. The OOD drop is *expected* and is part of the story; do not tune on OOD.

---

## 12. Tests and acceptance criteria

`pytest -q` must pass:

- `test_market.py`
  - n=2, c=1: `p_N ≈ 1.473 ± 0.005`, `p_M ≈ 1.925 ± 0.01`
  - n=3, c=1: `p_N ≈ 1.370 ± 0.005`, `p_M ≈ 2.00 ± 0.02`
  - `p_N < p_M` for all n; demand shares sum to < 1; CS decreases when all prices rise
- `test_simulate.py`
  - fully competitive run (n=3): mean Δ over t ≥ 100 in `[-0.15, 0.2]`
  - full cartel, κ=0.8, cheat prob 0, regime forced active: mean Δ in `[0.6, 0.9]`
  - same seed → identical DataFrame; different seed → different
- `test_features.py`
  - two perfectly co-moving price series → `corr_max == 1` (± 1e-9)
  - output has the 15 `FEATURE_COLS`, no NaN/inf, expected number of windows `(horizon - 50 - W)//10 + 1` per firm
- `test_dataset.py`
  - no `run_id` overlap between train/val/test/OOD
  - `FEATURE_COLS` does not include `kappa`, `in_cartel`, `frac_active`, `label`, `run_id`

Additional acceptance:
- `python -m collusion_rf.main all --quick` finishes in < 2 minutes and produces all 8 figures.
- Package LOC ≤ 1000.
- `README.md` contains the final metrics table and the policy comparison table copied from the real run.

---

## 13. Implementation order (milestones)

1. `config.py`, `market.py` + `test_market.py` ✅ green
2. `agents.py`, `simulate.py` + `test_simulate.py`; plot one run to eyeball it
3. `features.py` + `test_features.py`
4. `dataset.py` + `test_dataset.py`; run `generate --quick`
5. `train.py`, `evaluate.py`; run `--quick` end to end
6. `regulator.py` + policy comparison
7. Full `all` run, fill README, check LOC

## 14. Out of scope (do not build)

Multiple collusion types, RL/Q-learning agents, deep models, limit-order-book markets, a web UI, hyperparameter search beyond §8, SHAP.
