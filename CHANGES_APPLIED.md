# Design-Audit Changes — Applied

Companion to `DESIGN_AUDIT.md`. Every numbered item from that report is resolved
below, with the evidence that it works. `make test` is 67 tests, all passing;
`ruff check .` is back to the repo's pre-existing baseline of 10 warnings (all in
files this pass did not touch).

---

## 1. What is new in the pipeline

```
scripts/00_calibrate.py     NEW  the SS8 gate -- run before any training
scripts/01_generate_data.py      now rejects out-of-range trials, saves clean rates
scripts/02_train.py              test split stratified by posterior decile
scripts/03_analyze.py       REWRITTEN  every SS7 analysis + the SS9 control checks
scripts/04_figures.py            14 model figures, up from 7
```

`make calibrate CONFIG=...` runs stage 0 alone. `run_all.sh` runs it first and
**stops on FAIL** — verified: `run_all.sh --config configs/default.yaml` exits 1
at stage 0 and never writes a dataset.

New modules: `analysis/calibration.py` (the gate's statistics),
`analysis/units.py` (congruent/opposite, balance, lesion, RF shift),
`analysis/behavior.py` (Körding bias curves).

---

## 2. The audit items, one by one

### #1 `configs/pcommon1.yaml` was a stale copy — FIXED
Now `p_common: 1.0`, derived from `configs/flagship.yaml` with nothing else
changed. Verified on a trained network: `max |target − fused| = 1.1e-07`, i.e.
the targets reduce exactly to the always-integrate solution, and σ_out is now a
real measurement (0.627 deg on the means, 0.055 deg² on the variances).

### #2 p_common = 0 control — CONFIG + ASSERTION ADDED
`configs/pcommon0.yaml`, and stage 3 now measures the hand output's visual bias
whenever `p_common == 0`. **Result: slope = +0.0002.** No pipeline leak.

### #3 Satellites — ADDED
`pcommon03`, `pcommon07`, `pcommon028`, each differing from `flagship.yaml` only
in the Bernoulli constant.

### #4 Brute-force certification — NOW IN THE SUITE
`tests/test_design_guards.py::test_targets_match_brute_force_integration`
integrates the full posterior over (C, source, eye) on 601×601 grids and checks
w, both means and both variances per trial. Tolerances 1e-6 / 1e-5 / 1e-4.

### #5 Range containment — IMPLEMENTED
`sample_trials(..., contain=(lo, hi))` rejects and redraws through the identical
code path, so the rule cannot become a C cue; `containment_bounds()` derives the
bounds from the encoding (visual field minus 2·rf_width). The realized rate is
stored per dataset and reported by stage 0 and stage 3. **Flagship: 0.9%.**
Tests cover the bound, the C-symmetry of the surviving distribution, and the
recorded rate.

### #6 Poisson validity — CHECK ADDED, AND THE FAILURE FIXED
The gate ML-decodes the visual population at both ends of the σ range. The
statistic is now the **paired** ratio Var(x̂ − x)/σ², because the unpaired
Var(x̂)/σ² − 1 carries ~±1.8% of its own sampling noise at n = 6000 and was
overstating the problem.

| config | inflation |
|---|---|
| default (K=10) | **54%** FAIL |
| realistic (K=15) | **38%** FAIL |
| matlab_match (K=50) | **26%** FAIL |
| **flagship** | **3%** PASS |

The push–pull hand/eye codes were within 2% at every gain and needed no change —
the gaussian visual code was the only binding constraint.

### #7 §7.1 headline analyses — ALL THREE ADDED
`position_regression` (slope/intercept with SEs and 95% CIs, no division),
`sigma_out` + `sigma_w` (per-trial readability, σ_out taken from the control run
via `--control pcommon1`), `joint_fusion_weight` and `weight_consistency`.

### #8 §7.2 reliability test — ADDED
`reliability_within_disparity` bins |disparity| and regresses w_implied on
w_optimal within each bin. Bins where w_optimal barely varies are **dropped, not
downweighted** (they returned slopes like 61 ± 27) and listed under `skipped`.

### #9 §7.3 variance hump — ADDED
`variance_signature` bins the network's Var outputs by analytical posterior and
returns, per bin, the network output, the analytical mixture variance, the
between-component term alone, and the best fixed-weight prediction (which has no
hump by construction).

### #10 §7.4 model comparison — EXTENDED TO FIVE COMPARATORS
`model_comparison` adds full integration, full segregation and the best
fixed-weight model to averaging/selection/matching, binned by posterior decile.
The per-bin statistic is a **least-squares implied weight**, not a mean signed
bias — the latter cancels within a bin because Δ is signed, which is why the
first version of this figure looked flat. `fixed` nests integration and
segregation, so a parameter-free strategy wins ties within 0.5%.

### #11 §7.5 unit analyses — IMPLEMENTED
`congruency` (correlation of each MSL unit's visual- and proprioceptive-sweep
tuning), `balance`, `lesion` / `lesion_comparison`, `rf_shift` (preferred
spatial position vs eye position, plus gain fields). `todo.py` is now down to
two genuinely undesigned analyses.

### #12 §7.6 behavioral signatures — ADDED
`bias_vs_disparity` (Fig. 2e analog, with the Bayes-optimal overlay) and
`conditioned_bias` (Fig. 3b–c, split on the network's own causal judgment).
Both drop bins holding fewer than 30 trials — without that filter the
conditioned panel swung by ±3 deg on a handful of trials.

### #13 §8 calibration — NEW PIPELINE STAGE, AND A CALIBRATED CONFIG
`scripts/00_calibrate.py` implements the posterior histogram, the joint (w, Δ)
distribution, reliability coverage, containment, the Poisson check and the
anti-confound audit, with PASS/WARN/FAIL per design criterion.

The §8.3 coverage floor was **recalibrated**: the first version measured the
within-bin *std* and failed configs at 0.10, but the §7.2 test leverages the
within-bin *range*. Simulating a Bayes-optimal network showed it recovers slope
1.00 ± 0.003 at std 0.063 / range 0.37 — amply powered — so the check now uses
the range with a 0.15 floor, and says so in the code.

### #14 §8.4 stratified test set — IMPLEMENTED
`split_indices(..., stratify=post_c1)` splits within posterior decile bins;
training uses it automatically. Splits stay disjoint and complete.

### #15 §8.5 noiseless activations — SAVED
`encode(..., return_clean=True)`; datasets now carry `X_clean`.

### #16 §9.4 anti-confound audit — AUTOMATED
Per-channel AUCs for predicting C from |x_eye|, |x_vis|, |x_prop| and each σ²,
plus the moment match across C. **Flagship: worst AUC deviation 0.009 from 0.5.**

### #17 `p_common` → `post_c1` — RENAMED
The dataset key holding the trial-wise posterior is now `post_c1`; the config
key `p_common` remains the prior. Old datasets are remapped on load, so nothing
already generated breaks.

---

## 3. The calibrated flagship config

`configs/flagship.yaml`, derived from `matlab_match.yaml` by the calibration
pass (#6 and #13 done together, since they interact):

| | matlab_match | flagship | why |
|---|---|---|---|
| sensory σ² | as published | **× 0.30** | 56% → 25% intermediate posterior mass (§8.1) |
| `gain_K` | 50 | **180** | Poisson inflation 26% → 3% (§4) |
| `rf_width` | 10 | **6** | same |
| `n_vis` | 50 | **76** | same; also equalises group sizes |
| `visual_field` | ±75 | **±85** | containment 4.6% → 0.9% (§3) |

Tightening the noise was chosen over widening σ_P because it leaves the retinal
spread — and therefore containment — untouched.

**Gate result: eight PASS, zero WARN, zero FAIL.**

---

## 4. Verification run (50k trials, flagship + both controls + twin)

Read these as evidence the machinery works, not as final science — one seed, one
architecture.

| check | result | reading |
|---|---|---|
| Read-out R² | mu_vis 0.992, mu_prop 0.994, var_vis 0.876, var_prop 0.939 | means clear the §11 bar; variances are the harder channel |
| **Position regression** | slope **0.857** [0.831, 0.882], intercept −0.010 | intercept optimal; slope significantly below 1 — the network under-weights fusion slightly |
| §7.2 reliability | slope **0.572 ± 0.073**, CI [0.43, 0.72] | rules out a pure disparity heuristic (0); also below Bayes (1) |
| §7.3 variance hump | network **+3.40** vs analytical +4.55 deg² | present, slightly compressed |
| §7.4 comparison | averaging 1.79 < selection 2.04 < fixed 2.20 < segregation 2.24 ≪ integration 13.75 | **averaging wins** |
| p(C=1\|x) from MSL | **0.958** (SIL 0.482) | built by the network |
| same, always-fuse twin | **0.166** | emergent, not imposed |
| congruent / opposite | 17 / 24 units, index clearly bimodal | subpopulations emerge |
| balance vs posterior | r = **0.50** | the Rideaux mechanism carries it |
| MSL lesion | intact 1.48 → no-congruent 10.21, no-opposite 7.55 | both subpopulations load-bearing |
| RF shift gain | median **0.01** | MSL codes position spatially, not retinally |
| p_common=0 leak | **+0.0002** | clean |
| p_common=1 reduction | **1.1e-07** | exact |

The two sub-optimalities (regression slope 0.86, reliability slope 0.57) are
consistent with each other and are now attributable to the network rather than
the pipeline — every §10 guard is verified. Per the design, they are reportable
findings.

One number to treat with care: hand-vs-visual weight consistency comes out at
r = 0.125 with mean |difference| 0.141. The σ_w filter keeps high-|Δ| trials,
which are overwhelmingly confident-segregation trials, so w_optimal barely
varies across the surviving set and the correlation has almost nothing to
correlate. The mean absolute difference is the informative statistic there.

---

## 5. Where to look

`VERIFICATION.html` (repo root) is a self-contained page tying every item above to
its figure — open it in a browser, no server needed. The figures themselves are in
`results/flagship/figures/model/` and `results/calibration/flagship/figures/`.

## 6. Fix after first user run (2026-08-23)

`make all` crashed in stage 3 with `KeyError: 'residual_std'`. Cause: the guard
in `run_all.sh` tested only that `results/pcommon1/` **existed**, and a directory
left over from an earlier run satisfied it &mdash; so stage 3 loaded a
`metrics.json` written before `residual_std` was added and died on the missing
key. Two changes:

- `run_all.sh` now requires `results/pcommon1/metrics.json` to exist *and* to
  contain `residual_std` before passing `--control`.
- `03_analyze.py` never dies on the control any more. A missing run, a stale
  `metrics.json`, or an output-count mismatch each print a named warning and fall
  back to the run's own residuals; a control whose `p_common` is not 1.0 warns but
  proceeds. The fallback is conservative rather than wrong &mdash; the flagship's own
  residuals also carry any causal-inference misweighting, so &sigma;_w comes out too
  large, never too small.

If you hit this: `python scripts/03_analyze.py --run pcommon1` once to refresh
that control's metrics, then re-run. Or just `make all` again &mdash; it now sorts
itself out.

## 7. Housekeeping

`scripts/audit/` (the four exploratory scripts from the audit) has been removed
from this working copy — every check they performed now lives in
`analysis/calibration.py` or `tests/test_design_guards.py`. **Delete that folder
from your repo**; I could not remove it from your disk without a permission
prompt, and it will otherwise sit there as dead code that fails lint.
