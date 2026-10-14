# Algorithmic collusion detection & penalisation

Simulates a repeated logit-demand Bertrand market in which some firms run a **forgiving trigger
cartel**, trains classifiers to spot colluding firms from prices and noisy cost estimates only,
and plugs the best one into an **online regulator** that fines flagged firms and forces them to
price competitively. Full design: [SPEC.md](SPEC.md). Implementation choices, deviations and the
history of the scenario design: [NOTES.md](NOTES.md).

## Run it

```
pip install -r requirements.txt
pytest -q                                  # 25 tests

python -m collusion_rf.main all            # generate -> train -> evaluate -> penalise (~10 min)
python -m collusion_rf.main all --quick    # smoke test: 120 ID / 30 OOD / 20 policy runs, ~1 min
python -m collusion_rf.main generate       # or one stage at a time: generate | train | evaluate | penalise
```

Options: `--seed N` (default 42), `--out DIR` (default `outputs`). Everything is reproducible from
`(seed, run_id)`.

Outputs land in `outputs/`: `data/` (feature tables, splits), `models/` (three models, best model,
threshold), `reports/` (`metrics.json`, `penalty_results.csv`, `penalty_runs.csv`) and `figures/`
(8 PNGs).

## Layout

| File | Role |
|---|---|
| `collusion_rf/config.py` | all parameters |
| `collusion_rf/market.py` | demand, best response, symmetric and per-run asymmetric Nash / monopoly benchmarks, consumer surplus |
| `collusion_rf/agents.py` | `CompetitiveFirm`, `TriggerColluder`, cartel state machine |
| `collusion_rf/simulate.py` | scenario sampler, one-run simulator with regulator hook |
| `collusion_rf/features.py` | rolling windows, 15 features, labels |
| `collusion_rf/dataset.py` | parallel generation, run-level splits |
| `collusion_rf/train.py` | logistic regression, random forest, gradient boosting; F0.5 threshold |
| `collusion_rf/evaluate.py` | metrics, subgroup analysis, feature ablation, figures 1–6 |
| `collusion_rf/regulator.py` | online regulator, policy comparison, figures 7–8 |
| `collusion_rf/main.py` | CLI |

## Scenario design (revision 2)

| Parameter | In-distribution | Out-of-distribution |
|---|---|---|
| firms `n` | {3, 4} | {5} |
| collusion strength `κ` | U[0.3, 1.0] | U[0.1, 0.25] |
| price noise `σ_p` | U[0.005, 0.02] | U[0.02, 0.035] |
| cost volatility `σ_c` | U[0.002, 0.008] | U[0.008, 0.012] |
| punishment length `T` | 5–20 | 20–35 |
| product quality `a_i` (per firm) | U[1.8, 2.2] | U[1.8, 2.2] |
| firm cost `c_i = c_t × m_i` (per firm) | m ~ U[0.9, 1.1] | m ~ U[0.9, 1.1] |
| demand `mu` (per run) | U[0.18, 0.32] | U[0.18, 0.32] |

Quality, firm cost offsets and `mu` are **unobserved**: firms use the true values, the regulator
assumes `a = 2`, `mu = 0.25` and sees only the noisy common cost. The collusion index Δ is
reported against the run's true asymmetric Nash and joint-monopoly prices.

## Results (`python -m collusion_rf.main all`, seed 42)

All numbers are copied from the real run.

Data: 800 ID runs (400 with a cartel), 114,185 rows, positive rate 0.242; 200 OOD runs (97 with a
cartel), 41,000 rows, positive rate 0.206. Split by run: 560 train / 120 val / 120 test.

### Detection

| Model | Val PR-AUC | ID test ROC-AUC | ID test PR-AUC | OOD ROC-AUC | OOD PR-AUC |
|---|---|---|---|---|---|
| Logistic regression | 0.985 | 0.993 | 0.977 | 0.828 | 0.663 |
| **Random forest** (`max_depth=None`, `min_samples_leaf=5`) | **0.993** | 0.995 | 0.986 | 0.871 | 0.763 |
| Gradient boosting (`learning_rate=0.1`, `max_depth=3`) | 0.992 | 0.996 | 0.987 | 0.891 | 0.776 |

Selected model: random forest (highest validation PR-AUC), threshold τ = 0.737 (F0.5-optimal on validation).

| Random forest at τ = 0.737 | Precision | Recall | F1 | F0.5 | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|
| ID test | 0.962 | 0.884 | 0.921 | 0.945 | 12,936 | 142 | 466 | 3,553 |
| OOD | 1.000 | 0.028 | 0.054 | 0.125 | 32,540 | 0 | 8,224 | 236 |

Subgroups (random forest, ID test):

| Group | Rate | Rows |
|---|---|---|
| Recall, κ in [0.3, 0.4) | 0.498 | 574 |
| Recall, κ in [0.4, 0.6) | 0.917 | 1,303 |
| Recall, κ in [0.6, 0.8) | 0.967 | 1,332 |
| Recall, κ in [0.8, 1.0] | 0.968 | 810 |
| False-positive rate, umbrella firms (outsiders in cartel runs) | 0.001 | 2,050 |
| False-positive rate, umbrella firms while the cartel is active | 0.002 | 1,372 |
| False-positive rate, firms in fully competitive runs | 0.001 | 8,610 |
| False-positive rate, cartel members in inactive-regime windows | 0.055 | 2,418 |

Ablation — random forest retrained (same hyperparameters) without `markup_mean`, `markup_std`,
`rel_price` and `market_markup`:

| Features | ID test PR-AUC | OOD PR-AUC |
|---|---|---|
| All 15 | 0.986 | 0.763 |
| 11 (price-level features removed) | 0.880 | 0.676 |

- **Still above the spec's "too easy" line.** The tree models score ID-test PR-AUC 0.986–0.987, above the 0.98 threshold in SPEC §11. Heterogeneity alone does not hide a `κ ≥ 0.3` cartel; see NOTES.md for what would.
- **Detection leans on price levels.** `markup_mean` is by far the most important feature (permutation importance 0.27; next is `market_markup` at 0.04). Without the four price-level features, behavioural features alone still reach PR-AUC 0.88 in distribution.
- **Weak cartels are the misses.** Recall is 0.50 for κ in [0.3, 0.4) and at least 0.92 above 0.4.
- **Umbrella firms are almost never accused.** Most false positives are cartel members in windows where the cartel is mostly switched off.
- **Out of distribution the threshold fails.** OOD cartels (κ ≤ 0.25) raise prices less than unobserved heterogeneity does. Ranking quality drops to PR-AUC 0.76, and at the ID-tuned threshold the model flags almost nothing: recall 0.03, with no false positives.

### Penalisation

100 fresh ID cartel runs simulated under each policy with identical seeds, plus 100
competitive-only runs under the ML regulator. Mean ± std over runs, steps ≥ 100. Δ uses each run's
true asymmetric benchmarks.

| Policy | Collusion index Δ | Consumer surplus | Guilty fines | Innocent fines | Median delay (steps) | Never flagged | Runs with a permanently competitive member |
|---|---|---|---|---|---|---|---|
| none | 0.302 ± 0.166 | 0.799 ± 0.132 | 0 | 0 | – | 99 of 99 | 0% |
| ml | 0.077 ± 0.049 | 0.938 ± 0.091 | 23.34 ± 16.53 | 0.04 ± 0.43 | 52 | 4 of 99 | 80% |
| oracle | 0.031 ± 0.046 | 0.965 ± 0.092 | 19.90 ± 13.48 | 0 | 50 | 2 of 99 | 89% |
| ml, competitive-only runs | −0.020 ± 0.007 | 0.988 ± 0.093 | 0 | 0.23 ± 1.59 | – | – | – |

- The cartel activated in 99 of the 100 cartel runs; "never flagged" counts those 99.
- The ML regulator removes about 75% of the price elevation (Δ 0.302 → 0.077) and recovers about 84% of the consumer-surplus gain the oracle achieves.
- It is nearly as fast as the oracle (median 52 vs 50 steps).
- False accusations are rare but no longer zero: in the competitive-only markets, 2 of 100 runs had a firm fined (5 flags in total); in the cartel markets, 1 run had an innocent fine.

Figures: `pr_curves`, `roc_curves`, `confusion_matrix`, `feature_importance`, `example_runs`,
`recall_by_kappa`, `policy_comparison`, `penalty_example` (all `.png` in `outputs/figures/`).

Package size: 963 lines of Python (budget 1000). Full run ≈ 10.5 min on a 16-core laptop (≈ 7 min
of it the random-forest grid search); `--quick` ≈ 1 min.
