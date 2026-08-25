make clean-runs   # Design-Document Audit of `causal_integration`

**Audited against:** *Causal Inference Network Across Reference Frames — Complete Design Document* (Körding-extension of Farahmandi, Abedi Khoozani & Blohm 2025).
**Date:** 2026-08-23.
**Method:** every source file in `src/`, `scripts/`, `tests/`, `configs/` was read against the design section by section, and four numerical experiments were run against the actual code (now committed under `scripts/audit/`): a brute-force grid-integration certification of the analytical targets (§5's mandatory test), dataset-calibration statistics for all four configs (§8), the Poisson validity check (§4), and the same-seed satellite-invariance check (§9.3).

---

## A. Verdict at a glance

| Design section | Status |
|---|---|
| §1 Emergent (4 outputs, no p(C) unit), fixed prior per network | **Satisfied** |
| §2 Task definition (both reports in spatial coordinates, RFT under both C) | **Satisfied** |
| §3 Generative model: sampling order, identical C=2 priors, C-independence, sign convention | **Satisfied** (verified empirically) |
| §3 Range containment / rejection | **MISSING — change #5** |
| §4 Input encoding (tuning, gain-only reliability, independent Poisson) | **Satisfied** |
| §4 Poisson validity check | **MISSING, and currently FAILING — change #6** |
| §5 Likelihoods, log-space posterior, prior in all four conditionals, mixture variance, noisy eye | **Satisfied** — but see decision item D1 (eye-prior transform differs from the doc's formula, and the *code* is the exact ideal observer) |
| §5 Mandatory brute-force unit test | **Missing from the test suite — change #4** (run externally: PASSES to ~1e-13) |
| §6 Architecture, 4 outputs, MSE, loss balancing, ≥50k, seeds | **Satisfied** (softplus on variance columns is a minor doc deviation — D2) |
| §7.1 Implicit weight recovery | **Partial — changes #7** (position-domain regression, σ_w filter, joint two-output solve all missing) |
| §7.2 Bayesian-vs-heuristic test | **Partial — change #8** |
| §7.3 Variance-hump analysis | **MISSING — change #9** |
| §7.4 Binned model comparison | **Partial — change #10** |
| §7.5 MSL analyses | **Partial — change #11** (posterior decoding ✔; congruent/opposite, lesion, RF-shift are placeholders) |
| §7.6 Behavioral signatures (Körding Fig. 2e / 3b–c) | **MISSING — change #12** |
| §8 Pre-training calibration | **MISSING — change #13**; current configs violate the §8.1 posterior-histogram target |
| §8.4 Stratified test set | **MISSING — change #14** |
| §8.5 Per-trial saves | Nearly complete — noiseless activations not saved (**change #15**) |
| §9.1 p_common = 1 control | **Config was WRONG (fixed in this commit) — change #1** |
| §9.2 p_common = 0 control | **MISSING (config added in this commit; analysis assertion still to write) — change #2** |
| §9.3 Satellites | **Configs added in this commit — change #3**; same-seed invariance verified ✔ |
| §9.4 Anti-confound audit | **MISSING as automated check — change #16** (spot-checked manually: passes) |
| §9.5 Context-input variant | Not present (explicitly optional — no action) |

---

## B. What is already consistent (verified, no change needed)

**§3 Generative model.** `sample_trials` follows the strict sampling order: `C ~ Bernoulli(p_common)`; one shared source under C=1; two *independent* sources under C=2, both drawn from the identical prior `N(mu0, sigma0_sq)`; eye and all three σ² drawn identically regardless of C. p_common is consumed exactly once (the Bernoulli threshold); the observer reads the same config constant. There is no jitter between the sources under C=1. Empirically (40k trials, every config): realized C=1 fraction 0.498; eye and σ² conditional distributions match across C to sampling error (e.g. default: eye|C=1 = +0.05 ± 5.02 vs eye|C=2 = −0.04 ± 4.97).

**§3 Sign convention.** One convention, documented in the `generative.py` header and used consistently: `retinal = s_vis − eye`; recovery `x̃ = x_vis + ê`. `tests/test_vision_is_retinal` guards it.

**§4 Encoding.** Gaussian tuning with uniform centers for vision; linear push–pull with random slopes/intercepts for proprioceptive hand and eye; reliability enters *only* through gain `K/σ²`; no auxiliary σ inputs; independent Poisson draws per trial and channel from a single RNG stream (no shared seeds across populations or C branches).

**§5 Analytical targets.** `log_bayes_factor` is exactly Körding's L1/L2 (with general μ_P), computed in log space; the posterior is `expit(log_bf + logit(prior))` — the numerically stable equivalent of log-sum-exp (guard #7 satisfied). The prior term appears in **all four** conditional estimates (guard #5). The variance targets are the **full mixture variance** via the law of total variance (guard #6), with the exceedance property tested (`test_mixture_variance_exceeds_its_components_when_they_disagree`). The observer uses the **noisy** eye measurement, never the true θ_e (guard #8), and `test_observer_never_touches_the_true_sources` enforces the no-leak property by perturbing the hidden truth.

**§5 certification (run externally, must be added to the suite — see change #4).** For 100 random trials I computed the posterior over (C, s, e) by brute-force numerical integration on 801-point grids and compared to the code's closed forms:

```
p(C=1|x):  max|diff| = 2.5e-15      mu_prop: 1.1e-14    var_prop: 2.3e-13
                                    mu_vis : 1.4e-14    var_vis : 1.7e-13
```

The code's targets are the **exact** ideal observer to numerical precision. (The design doc's own plain-sum formula is not — see decision D1.)

**§6 Network/training.** input → SIL(64, sigmoid) → MSL(64, sigmoid) → 4-unit readout; no p(C) or C-classification unit (guard #2); MSE against the four analytical targets with per-column 1/Var loss balancing (the design's "consider loss balancing" — implemented and property-tested); 50k trials; seeds fixed in config, config + split stored in every checkpoint; train/val/test disjoint (tested); early stopping on validation.

**§7.5 (the part that exists).** Layer-wise ridge decoding targets the dataset key `p_common`, which stores the **trial-wise posterior p(C=1|x)** — the well-posed quantity per §9.1/guard #1 — not the prior. The emergent-vs-imposed comparison against the always-fuse twin is implemented.

**§9.3 mechanism.** Verified: with the same seed, changing only `p_common` leaves eye, all σ², both sources, and all three measurements bit-identical on every trial whose C did not flip (all latent streams are drawn unconditionally before `np.where(C==1, …)` selects). Cross-prior comparisons are therefore uncontaminated by construction — the satellite configs added in this commit exploit this.

---

## C. Required changes, in order of severity

### #1 — `configs/pcommon1.yaml` had `p_common: 0.5` — **fixed in this commit**
The file was a byte-identical copy of `configs/matlab_match.yaml` (same content, same mtime) that was never edited. Any "p_common = 1 control" run made from it was actually a second flagship run, and any σ_out estimated from it (for §7.1) would be wrong. It now sets `p_common: 1.0` and nothing else differs from `matlab_match.yaml`. **If your flagship base is a different config, re-derive it from that base — the rule is: only the Bernoulli constant may differ.**

### #2 — p_common = 0 control (§9.2): config added; the zero-bias assertion still needs code
`configs/pcommon0.yaml` is added (this commit). Still to implement in `scripts/03_analyze.py` (or a dedicated control script): for a network trained on it, regress the hand output's error on the visual disparity and **assert zero visual bias** — any residual bias is a pipeline leak. Also add a target-reduction check: at p_common = 0 the four targets must equal the unimodal-plus-prior solutions exactly (they do analytically in `model_average`; assert it on generated data).

### #3 — Satellite configs (§9.3): added
`configs/pcommon03.yaml`, `configs/pcommon07.yaml`, `configs/pcommon028.yaml` (this commit) — each differs from `matlab_match.yaml` only in the Bernoulli constant, per the design's uncontamination rule. The 0.28 satellite is the direct comparison to Körding's human fit.

### #4 — Add the mandatory §5 brute-force unit test to the suite
The design calls it "certifies this whole section" and it does not exist in `tests/`. `scripts/audit/brute_force_check.py` (this commit) is the working implementation and passes to ~1e-13; port it into `tests/test_generative.py` with ~20 trials and coarser grids so it runs in seconds, asserting max|diff| < 1e-9 on all five quantities.

### #5 — Range containment (§3, guard #9): not implemented anywhere
No rejection or truncation exists in `sample_trials`/`make_dataset`. Measured fraction of visual measurements `x_vis` falling outside `visual_field` minus the 2·rf_width margin (40k trials):

| config | outside field entirely | outside field − 2σ_rf margin |
|---|---|---|
| default | 0.03% | **3.3%** |
| realistic | 0.06% | 1.2% |
| matlab_match | **0.65%** | **4.5%** |
| equal_n | 0.00% | 0.00% |

For those trials the input carries less information than the target assumes nominal σ². Required: rejection-resample trials whose `x_vis` (and analogously the proprioceptive-hand and eye measurements against their reliably encoded spans) falls outside the margin, inside `sample_trials`; log the rejection rate; assert it is < a few % and statistically identical for C=1 and C=2 (guard #14); note in the dataset metadata that truncation slightly deforms the effective prior. `matlab_match` is over the "few %" line — either widen `visual_field` or accept and document the ~4.5% rejection.

### #6 — Poisson validity check (§4, guard #10): missing, and the current gains FAIL it
ML-decoding the visual measurement from the Poisson counts (unimodal, position fixed, 4k trials/condition) and comparing the empirical estimator variance to nominal σ²:

| config | K | inflation of effective variance over nominal σ² |
|---|---|---|
| default | 10 | **+52–54%** (at both ends of the σ range) |
| realistic | 15 | **+35–36%** |
| matlab_match | 50 | **+26–28%** |

Because gain = K/σ², the decoder noise scales with σ², so the miscalibration is *multiplicative and uniform across the whole σ range* — the network's true visual uncertainty is ≈1.3–1.5× what the targets assume, everywhere, including the causal-ambiguity regime the design worries about. Required: (a) add this decode-check as a calibration script/test (`scripts/audit/poisson_check.py`, this commit, is the implementation); (b) raise `gain_K` (and/or widen tuning coverage) until the inflation is a few percent, or explicitly recalibrate the observer's σ² to the *effective* (decoded) variances so targets and inputs agree. This must be settled **before** the flagship training run — it biases every optimality comparison in §7. The analogous check for the push–pull hand/eye codes should be added too.

### #7 — §7.1 headline analyses missing
`analysis/causal.py` implements the per-trial ratio `w_implied` with a fixed 1-deg `min_separation` filter and slope/R² against p(C=1|x). Missing, all three explicitly specified:
- **Position-domain regression (the headline statistic):** regress `(ŝ_net − ŝ_C2)` on `w_analytical·Δ` across all trials, per output; report slope, intercept, and CIs (Bayes-optimal: slope 1, intercept 0). No division, every trial enters with natural leverage.
- **σ_w-based filtering:** estimate σ_out from the p_common = 1 control network's residuals (now possible — change #1), set per-trial σ_w ≈ σ_out/|Δ|, and filter scatter plots by a stated criterion (e.g. σ_w < 0.1) instead of (or alongside) the raw 1-deg cut.
- **Joint two-output w:** each trial gives two equations (hand, visual) with one common w — solve jointly (weighted least squares over the two Δ's) and report hand-vs-visual consistency on high-|Δ| trials as the internal-coherence test of model averaging.

### #8 — §7.2 matched-disparity reliability test missing
`by_reliability` draws w-vs-disparity curves per reliability level (a good visualization) but there is no *test*: bin trials by |disparity|, and within bins test whether w_implied tracks the reliability-driven variation in w_analytical (e.g. within-bin regression of w_implied on w_analytical, or partial correlation controlling disparity). This is the discriminator between Bayes and a disparity heuristic; without it §7.2 has no statistic. Depends on the §8.3 coverage check (change #13).

### #9 — §7.3 variance-hump analysis missing entirely
Nothing in `analysis/` or `viz/` touches the network's Var outputs beyond scatter/R². Add: network Var_hand and Var_vis binned against the analytical posterior (and against disparity), overlaid with the analytical mixture variance and its between-component term w(1−w)(ŝ_C1 − ŝ_C2)²; the prediction is a hump at intermediate ambiguity that no fixed-weight model can produce. This analysis carries the ambiguous zone where §7.1 is blind — it is a secondary headline, not a nice-to-have.

### #10 — §7.4 binned model comparison incomplete
`strategy_fit` compares averaging / selection / probability-matching by global RMSE. The design specifies **five** comparators binned by **analytical-posterior deciles**: (a) model averaging, (b) full integration (p≡1), (c) full segregation (p≡0), (d) model selection (hard w at 0.5), (e) best fixed-weight model (fit the single w that best explains the network, then compare). Add (b), (c), (e); add decile binning so the discrimination happens in the intermediate zone (averaging → smooth sigmoid, selection → step); keep matching as a bonus comparator if you like.

### #11 — §7.5 remaining MSL analyses are placeholders
`analysis/todo.py` stubs: congruent/opposite classification (correlate each MSL unit's response to visual vs proprioceptive position sweeps, Rideaux congruency-index logic), the congruent−opposite activity-balance vs decoded/analytical posterior test, optional lesion of the two subpopulations, gain-field and RF-shift analyses for continuity with the published paper. These are the secondary claims of §1 — implement before the flagship write-up.

### #12 — §7.6 behavioral signatures missing
No bias-vs-disparity curve (Körding Fig. 2e analog: bias of each report toward the other cue as a function of disparity, network vs analytical), and no conditioning on the inferred cause (w_implied or decoded posterior > 0.5) to look for the negative-bias/truncation effect of their Fig. 3b–c. Add both to `analysis/` + `viz/results.py`.

### #13 — §8 pre-training calibration: missing as a pipeline stage, and current configs miss the §8.1 target
There is no `00_calibrate` stage. Measured on 40k trials:

| config | posterior mass in (0.2, 0.8) | mass < 0.05 or > 0.95 |
|---|---|---|
| design target | **≈ 25%** | substantial |
| default | 50% | 27% |
| realistic | 53% | 26% |
| matlab_match | 56% | 25% |
| equal_n | 77% | 10% |

Every config is on the "all-intermediate" side of the design rule ("all-intermediate → widen σ_P or tighten sensory noise") by 2–3×; `equal_n` severely so. Note the interaction with change #6: shrinking sensory noise or raising σ_P changes the gain calibration too — do #6 and #13 as one calibration pass. Required: a `scripts/00_calibrate.py` run **before** training that produces (1) the posterior histogram with the 25%-intermediate check, (2) per-output |Δ| histograms against expected σ_out and the joint (w, Δ) distribution (current |Δ| medians: 1.5–5.6 deg depending on config/output — recorded in `scripts/audit/calibration_check.py` output), (3) the within-disparity-bin reliability-coverage check for §7.2, and hard-fails or warns per the design's criteria.

### #14 — §8.4 stratified test set missing
`split_indices` is a uniform shuffle. Stratify the **test** split by analytical-posterior decile (equal counts per decile) so §7.4 has equal power across bins; train/val can stay random. No leakage exists today (splits are disjoint and checkpoint-stored) — only the stratification is missing.

### #15 — §8.5: save the noiseless activations
The dataset stores C, true sources, eye, all σ², all measurements, x̃_v, all conditional estimates and variances, w, the four targets, and the Poisson counts (X) — everything in the §8.5 list **except the pre-Poisson noiseless activations**. Have `encode_groups` optionally return/store the clean rates (they are computed anyway) so encoding diagnostics don't need re-simulation.

### #16 — §9.4 anti-confound audit: automate it
I spot-checked the conditional distributions of eye and σ² given C (they match — Part B), but the design requires it as an audit on every generated set, plus decoders: predict C from θ_e alone, and from single-modality input at matched positions — neither may beat the analytically expected level. Add to the calibration stage (#13) or as a test.

### #17 — Rename the dataset key `p_common`
`d["p_common"]` holds the trial-wise **posterior** p(C=1|x) while `cfg["generative"]["p_common"]` is the **prior**. The quantity decoded is the right one, but the name is exactly the confusion guard #1 warns about ("decode p(C=1|x), never p_common"), and it *will* mislead someone (or a reviewer reading `metrics.json`, where the key is `p_common_decoding_r2`). Rename to e.g. `post_c1` throughout (`generative.py`, `dataset.py`, `03_analyze.py`, `04_figures.py`, viz, tests).

---

## D. Decision items — where the design document itself should move

**D1 — §5 eye transform: the code is right and the doc's formula is not the exact ideal observer.** The doc mandates `x̃_v = x_v + x_e` with `σ_ṽ² = σ_v² + σ_e²`. The code instead shrinks the eye measurement toward its prior (`ê = k·x_e + (1−k)·μ_e`, `σ_ṽ² = σ_v² + var_eye`) — the sufficient statistic, since x_v and x_e are correlated through e. Brute-force integration over (C, s, e) shows the code matches the exact posterior to ~1e-13, while the doc's plain-sum targets deviate from it by up to **0.076 in p(C=1|x)** and **1.6 deg / 1.2 deg²** in means/variances (rms 0.13–0.39 deg on means). Recommendation: **amend the design doc** to the sufficient-statistic form (the "ideal observer has only what the network has" rule is still honored — ê uses only the noisy x_e plus the true generative prior, which the network can also internalize from training statistics), and lock it in with the unit test of change #4. If instead you want the literal-Körding plain-sum observer (a deliberately non-exact reference), that requires splitting `eye_sigma_sq`'s two roles (sampling width vs observer prior) with a new config key, because passing `inf` today would break the sampler.

**D2 — §6 "exactly 4 linear units".** The readout is linear but the two variance columns pass through a softplus for positivity. This is standard and harmless; amend the doc to "4-unit linear readout with a positivity link on the variance channels" rather than removing the softplus (removing it invites negative-variance predictions that poison §7.3).

**D3 — Optimizer.** The doc says "resilient backprop (or as before)"; the code deliberately uses Adam and warns against Rprop with mini-batches (`utils/config.py`). Keep Adam; note it in the doc.

**D4 — Two distinct always-fuse controls exist.** The repo's `head: fused` twin (2 outputs, trained on fused targets) is *not* the design's p_common = 1 control (4 outputs, w ≡ 1). Both are useful — the twin for the emergent-vs-imposed decoding test, the pcommon1 network for §9.1 continuity and σ_out. Keep both, and use only the pcommon1 network for σ_out in §7.1.

---

## E. Delivered in this commit

- `configs/pcommon1.yaml` — **fixed** (was a stale copy of matlab_match with p_common 0.5; now 1.0, nothing else changed).
- `configs/pcommon0.yaml` — new §9.2 control.
- `configs/pcommon03.yaml`, `configs/pcommon07.yaml`, `configs/pcommon028.yaml` — new §9.3 satellites.
- `scripts/audit/brute_force_check.py` — §5 mandatory certification (basis for change #4).
- `scripts/audit/calibration_check.py` — §8.1/§3-containment/§9.4 statistics per config.
- `scripts/audit/poisson_check.py` — §4 validity check (currently failing at all three gain settings).
- `scripts/audit/satellite_check.py` — §9.3 same-seed invariance proof.
- `DESIGN_AUDIT.md` — this report.

All controls/satellites are derived from `configs/matlab_match.yaml` (the config you are evidently iterating on — it carries the newest timestamps). If the flagship base changes, regenerate them from that base with only the Bernoulli constant differing.

**Suggested order of work:** #6 + #13 together (one gain/σ_P calibration pass, since they interact), then #5, then #14/#15, regenerate data, then #1/#2 control runs, then the analysis suite #7–#12, with #4/#16/#17 folded in along the way.
