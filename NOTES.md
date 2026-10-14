# NOTES — implementation choices and deviations from SPEC.md

## Deviations (things that differ from the letter of the spec)

1. **Scenario design (revision 2, current).** The scenario design has gone through three versions:

   | Version | ID `κ` | OOD `κ` | ID `σ_p` | OOD `σ_p` | Heterogeneity | Best ID-test PR-AUC |
   |---|---|---|---|---|---|---|
   | SPEC §5.1 as written | U[0.4, 1.0] | U[0.15, 0.35] | U[0.005, 0.02] | U[0.02, 0.035] | none | 0.998 (quick run) |
   | Revision 1 (superseded) | U[0.05, 0.5] | U[0.15, 0.35] | U[0.01, 0.04] | U[0.04, 0.055] | none | 0.977 |
   | **Revision 2 (current)** | **U[0.3, 1.0]** | **U[0.1, 0.25]** | U[0.005, 0.02] | U[0.02, 0.035] | **yes** | 0.987 |

   - *SPEC as written* was too easy by the spec's own rule (PR-AUC > 0.98, §11): a cartel at
     `κ ≥ 0.4` lifts the markup so far that `markup_mean` alone separates the classes.
   - *Revision 1* followed §11 literally (lower `κ`, higher `σ_p`). It worked, but left cartels
     so weak that the unregulated collusion index was only 0.10, and OOD `κ` sat inside the ID range.
   - *Revision 2* (requested change) restores strong cartels and the spec's `σ_p` ranges, puts
     OOD `κ` strictly below the ID range, and makes the task hard through **unobserved
     heterogeneity** instead of weak collusion:
     - product quality `a_i ~ U[1.8, 2.2]` per firm, fixed for the run;
     - firm cost `c_i,t = c_t × m_i`, `m_i ~ U[0.9, 1.1]` per firm, fixed for the run;
     - demand `mu ~ U[0.18, 0.32]` per run.

     Firms use their true `a_i`, `c_i,t`, `mu` in best responses, demand, profit and the price
     clip `[c_i,t, c_i,t + 3]`. The regulator still assumes symmetric `a = 2`, `mu = 0.25` and
     sees only the common `c_obs` (features, and `p̂_N` in the fine, are unchanged).
   - **True benchmarks are solved per run** (`market.RunBenchmarks`): the asymmetric Nash vector
     (simultaneous best-response iteration) and the joint-monopoly vector, on a 17-point grid of
     the common cost, linearly interpolated per step (max interpolation error < 1e-4, tested).
     The joint-monopoly solution uses the logit multiproduct result that all products carry one
     common markup `m` solving `m = mu / s_0(m)` (`scipy.optimize.brentq`); it reproduces the
     symmetric `minimize_scalar` value to 1e-5 (tested).
   - The cartel's target is per firm: `p_C,i = p_N,i + κ (p_M,i − p_N,i)`; punishment is
     `p_N,i (1 − ψ)`; a member deviates when its own price is below `p_C,i (1 − δ)`. `p_M` is the
     joint monopoly of *all* firms, also for partial cartels (as in the original spec).
   - **Δ reporting** uses these true benchmarks:
     `Δ_t = (mean_i p_i,t − mean_i p_N,i,t) / (mean_i p_M,i,t − mean_i p_N,i,t)`.
     The symmetric `BenchmarkTable` is now used only by the regulator.
   - `cost_true` in the run DataFrame is the common `c_t`; `profit` uses the firm's own cost.
   - Recall-by-κ buckets: [0.3, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.0] (the spec's three plus
     the new low tail).
   - The ranges of revision 2 were specified up front and not tuned further after seeing results.

   **Ablation (added with revision 2).** `evaluate.py` retrains the selected model (cloned, same
   hyperparameters, same train split) without the four price-level features `markup_mean`,
   `markup_std`, `rel_price`, `market_markup` and reports ID-test and OOD PR-AUC
   (`metrics.json → ablation`). Result of the full run (random forest): ID-test PR-AUC
   0.986 → 0.880, OOD PR-AUC 0.763 → 0.676.

   **Result of revision 2, and an open issue.** ID-test PR-AUC is 0.977 (logreg) / 0.986 (rf) /
   0.987 (hgb). The two tree models are therefore still **above the spec's 0.98 "too easy" line**,
   even with heterogeneity: per-run differences in `a_i`, `c_i` and `mu` move the competitive
   markup by less than a `κ ≥ 0.3` cartel does. I did not retune, because the ranges of this
   revision were given explicitly. What heterogeneity did change:
   - weak cartels are now genuinely hard (recall 0.50 for κ in [0.3, 0.4) vs ≥ 0.92 above 0.4);
   - OOD is much harder (PR-AUC 0.76; at the ID-tuned threshold recall is 0.03 at precision 1.0),
     because OOD cartels (κ ≤ 0.25) lift prices less than heterogeneity alone does;
   - the ML regulator now fines some innocent firms in competitive-only markets (2 of 100 runs).

   To get under 0.98 one would widen the heterogeneity ranges or extend ID `κ` below 0.3.

2. **Best response uses a Newton step instead of the plain fixed point.**
   The plain iteration `p ← c + μ/(1 − q)` has slope `−q/(1−q)`, which is ≈ −0.9 at the n = 2
   Nash point and < −1 from the prescribed start `c + μ`, so 30 iterations do not converge for
   duopoly. `best_response` solves the same FOC with the Newton update
   `p ← p·q + c·(1 − q) + μ` (same start, ≤ 30 iterations, early exit at 1e-10). It converges
   monotonically for every n and reproduces the sanity values (n=2: 1.473 / 1.925, n=3: 1.370 / 2.000).

3. **`build_features(run_df, kappa=-1.0)`** takes `kappa` as an optional second argument, since
   the run DataFrame of §5.2 has no `kappa` column.

## Choices where the spec was silent or ambiguous

- **Ramp-up grace period (10 steps).** Taken literally, monitoring would fire on the very first
  collusive step (prices start at the Nash level, far below `p_C·(1 − δ)`), and again right after
  every punishment, so the cartel could never form. After entering `COLLUDE` (from `INACTIVE` or
  from `PUNISH`) monitoring and cheating are suspended for `grace = 10` steps while prices climb
  with inertia `λ`. A member that is forced competitive by the regulator still gets punished
  repeatedly (punish → 10-step ramp → punish …), as §4.4 intends.
- **Cheating is not always punished.** A cheat prices at `0.95·p_C`; with `δ ~ U[0.03, 0.08]` this
  is only below the trigger line when `δ < 0.05`. Larger-`δ` cartels tolerate it (the "forgiving"
  part). Implemented literally.
- **Promotions** multiply the firm's posted price for 1–3 steps; because the next price starts
  from `p_{t-1}` (literal §4.2 formula) the price recovers gradually afterwards.
- **Random streams.** Each run uses three generators seeded from `(seed, run_id, k)`: scenario,
  environment (costs, observation noise, regime chain) and firm noise. The cost path and the
  cartel's on/off regime are therefore *identical* across the `none` / `ml` / `oracle` policies
  of the paired comparison; only firm-level noise diverges after an intervention.
- **Undercut features.** An event is one (rival, step) pair with a one-step price fall > 3%.
  `n_undercut_events` counts them; `punish_response` is the event-weighted mean of the firm's
  relative price change from the event step to 3 steps later (truncated at the window end).
- **`frac_active`** is the share of the window with the cartel regime on, stored for every firm of
  the run (also outsiders), so umbrella windows can be isolated.
- **Umbrella FPR** is reported twice: `fpr_umbrella` (spec definition: outsiders in any window of a
  cartel run) and `fpr_umbrella_active` (same, restricted to windows where the cartel is active —
  the real hard negative).
- **Regulator.**
  - Checks run when `t ≥ W` and `t % check_every == 0`, on steps `t−W+1 … t`.
  - A flag is "guilty" iff the true label of that window is 1 (same definition as the oracle's
    suspicion), so the oracle collects zero innocent fines by construction.
  - A firm under remedy (`t < forced_competitive_until`, or permanently competitive) is not
    re-examined. Without this the oracle would re-flag a firm 30 steps into its own remedy and
    `remedy_steps` would be meaningless. The regulator cannot tell guilty from innocent, so this
    also caps an innocent firm at two fines.
  - Detection delay = first flag of a cartel member at or after the first activation, minus the
    activation step. Runs whose cartel activated but was never flagged are counted separately.
  - Competitive-only policy runs use run_ids 30000+.
- **`--quick`** also uses 100 instead of 300 trees for the random forest, to stay under 2 minutes
  (measured: 59 s with revision 2).
- **Policy comparison runs on 4 workers**, not all cores. Each worker loads its own copy of the
  model; with 16 workers and a random forest selected, the first revision-2 run died with a
  `MemoryError` in `penalise` on a 16 GB machine that was also busy with other jobs. The stage
  was rerun on its own (`python -m collusion_rf.main penalise`, 111 s); `generate`, `train` and
  `evaluate` outputs are from the same code and seed, so all reported numbers are consistent.
- **Runtime.** Solving per-run benchmarks and vector-valued best responses makes a run about 3×
  slower than before (≈ 0.16 s). Full `all` ≈ 10.5 min, of which ≈ 7 min is the forest grid search.
- **Extra outputs:** `outputs/reports/penalty_runs.csv` (one row per run and policy) and an
  `--out` CLI flag (default `outputs`).
- `conftest.py` at the repo root only exists so that a bare `pytest -q` can import the package.
