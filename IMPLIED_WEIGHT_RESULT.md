# Implied weight: what sets the spikes, what the sweeps say, and how to make the weight readable

*Part I, 2026-09-20 (configuration); Part II, 2026-09-21 (readability); Part III, 2026-09-28 (the variance read explained, the variance/mean hybrid tested and, in §14, applied as option (a)); Part IV, 2026-09-29 (the cleanup: the hybrid is the pipeline's weight, everything else retired), below. Investigation of the question "in what configuration can the weights implied from the network be brought closer to the analytical solution / into [0, 1]", following the exp2_128_mean result (4.3 % of per-trial ratios outside [−1, 2], position-regression slope 0.99) and the supervisor's options A (balanced weight distribution), B (systematic range testing) and C (fixed realistic variances). Nothing in the pipeline was changed; every new analysis lives in `src/cmsi/experiments/` and `scripts/06_implied_weight.py` (`compare`, `design`) and `scripts/09_sweep.py`.*

## 1. The one identity that decides everything

On a network that is Bayes-optimal up to a position error *e* on a trial, the per-trial ratio is exactly

    w = (network − seg) / Δ = p + e / Δ          Δ = fused − seg,  p = analytical posterior

so the ratio leaves [−1, 2] when |e/Δ| is large, and the expected fraction outside is

    E[ Φ(−(1 + p)|Δ| / σ_e) + Φ(−(2 − p)|Δ| / σ_e) ]        (Gaussian e of sd σ_e, `implied_weight.predicted_outside`)

Two things enter, and only two: how the trials distribute |Δ| (fixed by the configuration before any training) and how large the network's error *e* is (own residual sd on the test split: σ_out plus any misweighting). Small |Δ| trials are the confident-fusion trials — at p → 1 the two hypotheses coincide by definition — so **no configuration puts every per-trial ratio into [0, 1]**; the fraction outside can only be pushed down by making σ_e small relative to the noise scale that sets |Δ|. To get below ~1 % outside one would need σ_e ≈ 0.1°, which is the decoding floor of the spike code itself (0.10–0.15°, see §3). The least-squares per-bin weight and the position regression do not divide by Δ and are the estimators to report; the ratio histogram is a diagnostic, and its expected value is now computed alongside it.

## 2. Where exp2_128_mean's improvement came from (existing runs, test splits, vis read)

| run | hidden | σ0² | eye² | ranges | σ_out | own sd | \|Δ\|<1° | median \|Δ\| | outside | predicted from σ_out | slope | intermediate posterior |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flagship | 64 | 425 | 325 | wide | 0.628 | 1.797 | 16 % | 3.9° | 9.5 % | 6.0 % | 0.856 | 25 % |
| exp2 | 64 | 425 | 325 | narrow | 0.822 | 1.384 | 26 % | 2.5° | 13.3 % | 12.1 % | 0.887 | 15 % |
| exp2_128units | 128 | 425 | 325 | narrow | 0.651 | 1.105 | 26 % | 2.5° | 11.0 % | 9.7 % | 0.859 | 15 % |
| exp2_128_handlimit | 128 | 169 | 81 | narrow | 0.418 | 0.858 | 27 % | 2.2° | 5.9 % | 6.2 % | 0.979 | 31 % |
| exp2_128_eyelimit | 128 | 100 | 49 | narrow | 0.337 | 0.578 | 29 % | 2.0° | 5.4 % | 5.8 % | 0.925 | 48 % |
| exp2_128_mean | 128 | 100 | 81 | narrow | 0.335 | 0.458 | 29 % | 2.0° | 4.3 % | 5.5 % | 0.995 | 49 % |

("wide" = [1.2, 6.6]/[1.8, 9.6]/[3.3, 13.2]; "narrow" = [1, 1.2]/[4, 4.5]/[3.7, 4.2].)

- **The trials became less readable, not more.** Narrow, low-noise ranges shrink Δ on the fusion trials (sd of Δ_vis on C = 1 trials 2.8° → 1.55°): the share of trials with |Δ| < 1° rose from 16 % to 29 % and the median |Δ| halved. At the *same* error, exp2_128_mean's trials give more spikes than the flagship's (design screening: at σ_e = 0.335°, 5.7 % vs 3.2 % outside).
- **All of the improvement is the network.** Own residual sd fell 1.80 → 0.46: the flagship's systematic under-fusion (slope 0.86; own sd 2.9 × its σ_out) is gone (slope 0.995; own sd 1.4 × σ_out). Counterfactual with the identity above: flagship trials + exp2_128_mean's error → 4.5 % outside; exp2_128_mean trials + flagship's error → 26 %.
- **Which change did it:** along your ladder, 128 units alone lowered σ_out (0.82 → 0.65) but left the slope at 0.86–0.89; the slope only moved when the spread of hand positions and eye positions shrank (σ0² 425 → 169 → 100, eye² 325 → 81): 0.98–0.995. The network's failure on the flagship is an approximation problem over a large input domain relative to the sensory noise, not a capacity problem in the narrow sense (§3, S2).
- **The fraction outside is now what an optimal network with this σ_out would show** (4.3 % observed vs 5.5 % predicted from σ_out; the prediction over-shoots because the error is smaller on small-|Δ| trials). There is nothing left to gain on the ratio from optimality at this configuration; only σ_out (relative to the noise scale) or the |Δ| distribution can move it further.

## 3. New sweeps (this session)

Run with the delivered `09_sweep.py` / `07_architecture.py` on a CPU copy of the exact pipeline code, same datasets (same seeds), 50,000 trials, 300 epochs with early stopping. The reference points reproduce your runs: flagship 64 → 9.5 % / slope 0.857 (yours 9.5 % / 0.856); exp2_128_mean → 4.7 % / 0.987 (yours 4.3 % / 0.995). One seed per variant; a seed replicate of two variants moved the outside fraction by ≤ 0.3 pp, the vis slope by ≤ 0.01, own sd by ≤ 0.03 (the prop slope by up to 0.045) — differences smaller than that are noise. Every variant has its own control. Vis read unless stated; "outside" = per-trial ratios outside [−1, 2]; "inter" = posterior mass in 0.2–0.8 (the calibration gate wants 15–40 %).

**S1 — σ0² at exp2 ranges, eye² 81, 128 units** (`--param sigma0_sq --values 100 169 425 --config exp2_128_mean`)

| σ0² | σ_out | own sd | outside vis / prop | slope vis / prop | inter |
|---|---|---|---|---|---|
| 100 | 0.345 | 0.47 | 4.7 / 5.2 | 0.987 / 0.959 | 49 % |
| 169 | 0.384 | 0.90 | 5.7 / 6.8 | 0.986 / 0.978 | 31 % |
| 425 | 0.658 | 0.85 | 8.1 / 9.9 | 0.965 / 1.003 | 15 % |

**S5 — eye² at σ0² 100** (`--param eye_sigma_sq --values 49 325`, 81 from S1): 49 → 5.1 % / 0.965; 81 → 4.7 % / 0.987; 325 → 6.2 % / 0.926. The eye prior width matters as much as the hand prior: it sets the retinal domain the visual population has to cover.

**S2 — hidden units on the flagship generative model** (`07 --hidden 64 128 256`): 64 → 9.5 % / 0.857 / own sd 1.79; 128 → 7.2 % / 0.908 / 1.32; 256 → 6.1 % / 0.890 / 1.15 (σ_out 0.63 → 0.54 → 0.53). Capacity helps the flagship a little and stalls; the wide domain is the obstacle.

**S6 — flagship (wide) ranges, eye² 81, 128 units, σ0² 100 vs 425**: 100 → σ_out 0.299, own sd 0.64, **3.7 % outside (the best vis readability of everything trained, 49 % of trials pass the σ_w criterion)**, slope 0.964, but 61 % intermediate posterior mass and the prop read at 7.5 %; 425 → 6.3 % / 0.903 / 24 %.

**S7 — width of the noise ranges at σ0² 169, eye² 81, 128 units**: narrow → 5.7 % / 0.986 / inter 31 %; medium [1, 3]/[3, 6]/[3.3, 6] → 6.3 % / 0.954 / 37 %; wide (flagship ranges) → 4.7 % / 0.944 / 50 %. Wider reliability ranges make the weighting harder to learn (slope 0.99 → 0.95) and the task more ambiguous, but enlarge Δ on the vis read.

**S3 — visual noise alone at σ0² 100 (prop [4, 4.5])**: [1, 1.2] → 4.7 % / 0.987; [3, 3.6] → 4.9 % / 0.969; [6, 7.2] → 4.0 % / 0.978 on the vis read, while the prop read worsens 5.2 → 7.2 %. Δ_vis on fusion trials scales as σ²_vis,body / √(σ²_vis,body + σ²_prop): the noisier cue's read is the readable one, and the two reads trade off — the disparity is split between them.

**S4 — Option A, training side** (`--balance 0.5`, trials thinned to a flat posterior histogram by an acceptance that depends on the measurements only, evaluated on the *natural* test split): outside 4.0 % vs 4.7 %, slope 0.982 vs 0.987, own sd 0.53 vs 0.47; `--balance 0.7`: 4.5 % / 0.963 / 0.52. Within seed noise. The evaluation-side reading — re-weighting the existing test trials to a flat posterior (`outside_if_flat_posterior`, no retraining) — gives the same-sized effect for free: 3.3–4.7 % across all variants, 5.7 % for the flagship. Balancing the *training* distribution does not make the network more optimal.

Two things about Option A as literally stated ("adjust variances across mean combinations"): if the variances are made to depend on the true sources or on C, the anti-confound audit (SS9.4) is violated and the observer's targets are wrong; if they depend on the measurements only (as `balance_posterior` does, through the posterior), the targets stay exactly right, but the single-channel AUCs of the audit still move a little (|x_prop| AUC 0.57 at keep 0.5, 0.535 at keep 0.7; the 0.05 rule would flag the former) because they are marginal statistics of a selected distribution. A first version of this run had the acceptance draw on the same random stream as the sampler's C draw (seed reuse), which silently rejected trials by their cause; it was caught by the natural-split evaluation, fixed (derived seed), guarded by a test (`test_balance_posterior_accepts_by_posterior_bin_and_not_by_cause`) and re-run — the numbers above are from the corrected run.

**Option C** (fixed realistic variances) is `08_fixed_variance.py` (+ `--set sigma0_sq=…`) and was not re-run: with reliability constant the network's task is simpler and the readability question reduces to σ_out vs the one Δ scale; nothing in the sweeps suggests it would improve the ratio beyond what narrow ranges already do, and it removes the reliability analyses.

## 4. What it adds up to

1. **Optimality (the LS curve on the identity, slope ≈ 1) is governed by the input domain relative to the sensory noise** — σ0² and eye_sigma_sq — and secondarily by the width of the reliability ranges. Units help σ_out but not the slope on the wide domain. With σ0² ≤ 169, eye² 81, 128 units and narrow ranges the network is optimal to within 0.01–0.02 up to p ≈ 0.85 (the top posterior bin at p ≈ 0.94 reads 0.75–0.80 on the narrow- and wide-range 128-unit configurations, 0.63 with the medium ranges and 0.50 on the flagship, because that bin *is* the small-Δ bin).
2. **The per-trial ratio's spikes are at their floor for the configuration**: 4–5 % outside is what an optimal network with σ_out ≈ 0.3–0.35° shows on these trials, and σ_out is still 3× the decoding floor (0.10–0.15°). Further gains would need σ_out → 0.15° (a much better network: more units did not do it) or a Δ distribution with fewer near-zero trials, which means noisier cues on the read you care about (S3, S6) — at the price of posterior sharpness.
3. **The trade-off the sweeps expose**: shrinking σ0² to 100 makes the causal task ambiguous — 49–61 % of trials at intermediate posterior (the gate's window is 15–40 %), the posterior at zero disparity capped near 0.78 — because two independent sources of a 10° prior are often as close as one shared source. σ0² = 169 with eye² 81 sits inside the gate's window (31 % narrow, 37 % medium ranges) at slope 0.95–0.99.
4. **Candidates to take through the pipeline** (`06 train`, then `06 sigma` / `04_figures`), in order of how well they balance optimality against a task that still has confident trials:
   - `sigma0_sq=169 eye_sigma_sq=81 hidden=[128,128]` with the narrow ranges — your `exp2_128_handlimit`: slope 0.98–0.99, 5.7–5.9 % outside, 31 % intermediate. Best optimality inside the gate's window; the narrow ranges make figure 05's reliability levels nearly identical, which is the cost.
   - the same with medium ranges `[1,3]/[3,6]/[3.3,6]`: slope 0.95–0.97, 6.3 %, 37 % intermediate, and reliability varies enough for the reliability analyses.
   - if the vis read is the one that matters, widen the *visual* range relative to prop (S3: [6, 7.2] gave 4.0 % on mu_vis at slope 0.98) — and read the weight from the noisier cue's output, since that is where Δ is.
   - three seeds of whichever you pick before it goes in the manuscript.
5. **For the manuscript**: report the LS / position-regression weights, and next to the ratio histogram give the fraction expected from the identity (now in every `metrics.json` as `readability.outside_predicted_sigma_out`), so a reader sees the spikes are the estimator's, not the network's.

## 5. What was added to the codebase

- `weight_analysis` now carries a `readability` block per read (own sd, own sd on |Δ| < 1° and > 4° trials, share of |Δ| < 1°, observed / predicted / flat-posterior fractions outside, Δ and error by true cause). Every experiment inherits it; `07`/`08`/`09` draw it as figure `E_readability_by_variant`.
- `06_implied_weight.py compare --runs a b c`: trained pipeline runs side by side (figures A–F: readability, noise + decoding floor + slope, |Δ|/own sd, posterior, LS weight vs posterior and vs disparity).
- `06_implied_weight.py design`: a configuration before training — posterior histogram, |Δ| per read, expected fraction outside vs error sd (solid as drawn, dashed flat-posterior), fraction passing the σ_w criterion, the decoding floor; several configs or `--param/--values` on one figure; `--mark RUN` places trained runs' errors on the curves; `--balance` reports the thinned set and its audit. Already run on your machine: `results/experiments/implied_weight/design_ladder/` (flagship, exp2, exp2_128_mean).
- `09_sweep.py train --param KEY --values … [--set …] [--balance KEEP]` / `compare`: one key over several values, own datasets and control per variant, cached with the config and the balance in the signature; `cmsi.experiments.sweep`, `cmsi.experiments.design` (`predicted_outside`, `flat_posterior_weights`, `readability`, `readability_curves`, `decoding_floor`, `balance_posterior`, `design_analysis`, figures).
- Tests: 102 (7 new: the identity against a simulation, flat weights, the readability block against its arrays, balancing keeps the fraction / flattens / is blind to C given the posterior bin, design screening, sweep labels). README (experiments section, layout, outputs) and GUIDE updated.
- Results of every sweep above: `implied_weight_sweeps_2026-09-20.tar.gz` (metrics, configs, arrays.npz for `09 compare`, compare figures, per-variant 16 / sigma_05 / sigma_07, logs, the summary table). Unpack in the repo root and `09 compare --name s0_exp2_128 s0_169_medium s0_169_wide` etc. work on them.

## 6. Caveats

- One seed per variant (replicates for two of them); the σ0² = 169 own sd (0.90 vs 0.47 at 100) is larger than seed noise but comes from one run.
- The Gaussian prediction over-predicts the observed outside fraction by 30–60 % on every run because the error is smaller where |Δ| is small; use it for ranking configurations and as an upper bound, not as a point estimate.
- The decoding floor assumes each trial's gain is known to the decoder; it is a lower bound on σ_out.
- The sweeps were trained on CPU in a cloud copy of the repository (same code, same seeds, different hardware); early stopping landed at epochs 100–260, as in your runs.


---

# Part II — increasing readability (2026-09-21)

The question after Part I: is there any way to make the per-trial weight more readable than the configuration sweeps allow? Yes — two ways that need no new configuration at all, and a training recipe that halves the remaining spikes. All of it is in the codebase now (`weight_analysis` → `per_trial`, `sigma` figure 08, the tables of `compare`/07/08/09) and was verified on your own runs (numpy re-analysis of `analysis.npz` on the Mac) and on the cloud sweeps.

## 7. Read the weight from the variance outputs

The ratio divides by Δ, so it is blind exactly where the weight is highest. The network's *variance* outputs carry the same weight without that blind spot. A model-averaging observer's variance is the mixture variance (Eq. 10),

    v(w) = w·var_fus + (1 − w)·var_seg + w(1 − w)·Δ²

and a variance output can be solved for the weight the network used — a quadratic (`implied_weight.variance_weight`). Its sensitivity `|dv/dw| = |Δ²(1 − 2w) − c|`, with `c = var_seg − var_fus` the variance fusion saves, does **not** vanish at Δ = 0: there it equals c (≈ 2.4 deg² in exp2_128_mean, ≈ 8 deg² in the flagship). With the control's σ_out on the variance outputs (0.014 / 0.055 deg²) the per-trial uncertainty of this read is ~0.005 in w on 98 % of trials. When Δ² > c there are two roots; the position read — precise on exactly those large-|Δ| trials — picks the one. Nothing is filtered: every trial gets a weight.

What it gives, on the test splits (vis read for the ratio; "sd" is the sd of the per-trial weight minus the analytical posterior):

| run | estimator | outside [−1, 2] | sd, all trials | sd, \|Δ\| < 1° | sd, \|Δ\| > 4° | corr with posterior | σ_w < 0.1 |
|---|---|---|---|---|---|---|---|
| exp2_128_mean | ratio, mu_vis | 4.3 % | 24 | 44 | 0.08 | 0.03 | 34 % |
| | both position outputs (joint) | 1.4 % | 0.46 | 0.83 | 0.06 | 0.59 | 42 % |
| | **both variance outputs** | **0.0 %** | **0.072** | **0.050** | **0.032** | **0.98** | **99 %** |
| | all four combined | 0.0 % | 0.073 | 0.050 | 0.032 | 0.98 | 100 % |
| flagship | ratio, mu_vis | 9.5 % | 55 | 135 | 0.23 | 0.00 | 39 % |
| | joint | 5.6 % | 2.6 | 6.3 | 0.19 | 0.14 | 41 % |
| | **variance** | **0.0 %** | **0.175** | **0.112** | **0.18** | **0.90** | **98 %** |
| exp2_128_handlimit | ratio / joint / variance | 5.9 / 2.4 / 0.0 % | 3.4 / 0.68 / 0.073 | 6.5 / 1.2 / 0.054 | | 0.10 / 0.48 / 0.98 | |
| lownoise_both | ratio / joint / variance | 15.6 / 9.0 / 0.0 % | 107 / 2.4 / 0.196 | 224 / 4.9 / 0.097 | | 0.02 / 0.17 / 0.89 | |

Binned by posterior, the variance read sits on the identity for the near-optimal networks (exp2_128_mean: 0.01 / 0.56 / 0.84 / 0.92 at p ≈ 0.05 / 0.5 / 0.85 / 0.95, where the least-squares position read gives 0.84 in the top bin because that bin *is* the small-Δ bin), and it is not a copy of the posterior: on the flagship it reads 0.25 at p = 0.15 and 0.86 at p = 0.94 — the network's own over-fusion at low posterior and under-fusion at high posterior, trial by trial.

**Coherence.** Both reads measure the same weight only if the network mixes its position and variance outputs with one w. On the trials where the position read is precise (σ_w < 0.1, 41 % of trials) the two agree on every run to within 0.01 on average (|variance − position| ≤ 0.008 on exp2_128_mean, flagship, exp2, handlimit and lownoise_both), and bin by bin up to p ≈ 0.5–0.6 on all of them. Above that the near-optimal networks stay together (exp2_128_mean: 0.763/0.719 at p ≈ 0.76, 0.846/0.858 at 0.84) while the **flagship diverges**: its position read stays at 0.38–0.55 from p = 0.65 up while its variance read climbs from 0.59 to 0.86 — the flagship reports fusion-level uncertainty on confident-fusion trials but moves its estimate only about half the way. That is a diagnostic of a suboptimal network that neither read shows alone, and it is what figure `sigma/08` panel D draws.

**The empirical precision of the variance read** is the run's own variance error on small-Δ trials divided by c: 0.095/2.4 ≈ 0.04 in w for exp2_128_mean, 0.93/7.9 ≈ 0.12 for the flagship — ten to a thousand times better than the ratio on the same trials, and every trial is readable. The control-based σ_w (0.005) understates it, which is why the coherence block reports the measured spread.

**Also cheap: read both position outputs together.** The two position outputs share one weight and the disparity is split between them, so the least-squares read across both (`analysis.joint_fusion_weight`, sd σ_out/√(Δ_vis² + Δ_prop²)) has 2–3× fewer spikes than either ratio (exp2_128_mean 4.3 → 1.4 %, flagship 9.5 → 5.6 %, handlimit 5.9 → 2.4 %). It is the position-based estimator to use per trial; the ratio from one output has no advantage over it.

## 8. The network's error: training levers (exp2_128_mean base, cloud runs, one seed)

| change | σ_out | own sd | own/σ_out | ratio outside | joint outside | variance sd | slope vis / prop |
|---|---|---|---|---|---|---|---|
| baseline (sigmoid, 128, gain_K 180, lr 1e-3, 50k) | 0.345 | 0.47 | 1.37 | 4.7 % | 1.6 % | 0.066 | 0.987 / 0.959 |
| activation relu | 0.397 | 0.40 | 1.01 | 3.8 % | 1.3 % | 0.061 | 1.001 / 0.971 |
| activation tanh | 0.554 | 0.51 | 0.92 | 5.1 % | 1.9 % | 0.070 | 0.941 / 0.961 |
| gain_K 360 | 0.265 | 0.44 | 1.68 | 3.5 % | 1.0 % | 0.056 | 0.981 / 0.977 |
| gain_K 720 | 0.208 | 0.44 | 2.10 | 3.0 % | 0.9 % | 0.053 | 0.968 / 0.988 |
| hidden [256, 256] | 0.301 | 0.42 | 1.41 | 3.8 % | 1.1 % | 0.068 | 0.984 / 0.969 |
| lr 3e-4, 600 epochs, patience 40 | 0.380 | 0.39 | 1.02 | 4.1 % | 1.2 % | 0.076 | 0.980 / 0.989 |
| n_trials 100,000 | 0.314 | 0.38 | 1.21 | 4.2 % | 1.1 % | 0.057 | 0.992 / 0.995 |
| **all of the above together** (relu, 256, gain 720, lr 3e-4, 100k) | **0.180** | **0.19** | **1.07** | **2.1 %** | **0.3 %** | **0.032** | **0.999 / 0.985** |
| the same recipe on the wide ranges, σ0² 169, eye² 81 | 0.250 | 0.39 | 1.55 | 2.3 % | 0.5 % | 0.052 | 0.962 / 0.936 |

Each lever alone is worth 0.5–1.5 points of spikes; together they halve them (4.7 → 2.1 %) and bring the network's own error down to its control's (own sd 0.19 = 1.07 × σ_out), with 53 % of trials passing the σ_w criterion instead of 34 %. Higher gain_K lowers σ_out (the spike-count noise does limit the control) but not the causal network's own error; relu, lower lr and more data lower the own error (the approximation). On the wide-range σ0² = 169 configuration the recipe gives 2.3 % outside at slope 0.96 with 58 % of trials readable — the flagship-style generative model made readable.

**One warning that comes with relu.** The six outermost visual units (RF centres at the edge of the ±85° field) never fire in 100,000 trials, so the input standardiser clamps their std to 1e-6 (`Net.fit_standardizer`); a single spike from such a unit in a held-out trial then becomes an input of 10⁶, which a sigmoid saturates harmlessly but a relu passes through linearly. In one of the two recipe runs the control's *validation* loss was ~8 × 10⁸ at every epoch for exactly this reason (its test-split σ_out came out fine, by luck of the split). Before relu is used for anything that matters, the clamp should become `clamp_min(1.0)` (one spike) or silent units should be dropped — a one-line change in `models/network.py` I have not made, because it is the pipeline.

## 9. What to do, in order

1. **Report the variance read** (`06 sigma --run <run>` now prints the per-trial table and writes `sigma/08_weight_from_variance_all_trials`; `metrics.json` carries it under `per_trial`). It answers the readability question on every trial without filtering, and its coherence with the position read is a new, sharp test of model averaging: one weight for both outputs (Eqs. 9 and 10), which the near-optimal networks pass and the flagship fails above p ≈ 0.6.
2. **Use the joint position read** wherever a per-trial position-based weight is shown; keep the single-output ratio only for the figure that explains why it spikes.
3. **Retrain with the recipe** (relu after the standardiser fix, [256, 256], gain_K 720, lr 3e-4, 600 epochs / patience 40, 100k trials) — on σ0² = 169 / eye² 81 with medium or wide ranges if the reliability analyses matter, on the exp2 ranges if optimality matters most. Three seeds.
4. Keep Part I's configuration conclusions: the domain (σ0², eye²) sets optimality; ranges trade Δ against posterior sharpness; balancing (Option A) is an evaluation-side re-weighting, not a training change.

## 10. Added to the codebase (Part II)

- `implied_weight.variance_weight`, `per_trial_weights` (ratio per output, joint, variance, combined; per-estimator summaries; coherence table), `fig_weight_from_variance` (sigma figure 08), `print_per_trial`; `weight_analysis` returns them under `per_trial` (None when a dataset lacks the variance components).
- `06 sigma` prints the table; `06 compare`, `07/08/09 compare` tables carry the joint-read spike fraction and the variance-read sd.
- Tests: 104 (2 new: the quadratic inverts the mixture variance exactly, roots and flags; a coherent synthetic observer is read to within 0.03 on every trial by the variance outputs while the ratio is not).
- Cloud sweeps of §8 in `results/experiments/sweep/{act,gain,units,lr,ntrials,recipe}_exp2_128` and `recipe_169_wide`, every variant re-analysed with the per-trial block and figure 08.


## 11. Pipeline integration (2026-09-25)

The variance read is now part of the main pipeline, beside the position read (both stay in every run):

- `analysis.variance_weight` (the quadratic inversion), `analysis.mixture_variance`, `analysis.variance_regression` (variance-domain headline: `var_out − var_seg` on `v_opt − var_seg`) and `analysis.weight_regression` (a per-trial read on the posterior) live in `cmsi/analysis/causal.py`; the experiment module re-exports them.
- Stage 3 stores `fusion_weight_var_vis/prop` in `analysis.npz` and `variance_regression_vis/prop`, `weight_regression_var_vis/prop`, `implied_weight_var_vs_post[_prop]` in `metrics.json`, and prints both regressions.
- Stage 4 writes `figures/model/04v_fusion_weight_variance_read` (figure 04 with the variance read, position reads faded) and `08v_variance_regression` (A variance-domain regression, B the read on the posterior) — `viz.results.fusion_weight_variance_curve`, `variance_regression_figure`.
- `model.std_floor` (default 1e-6, the historical value) floors the input standardiser; relu configs set it to 1.0. `analysis.compare` no longer crashes when the σ_w filter keeps no trial.
- `configs/optimal_sweep.yaml`: the recipe the sweeps converged on (σ0² 100, eye² 81, narrow ranges, relu 256×256 with `std_floor: 1.0`, gain_K 720, lr 3e-4, 600 epochs / patience 40, 100k trials), with the wide-range σ0² = 169 alternative in its header. Run: `python scripts/06_implied_weight.py train --config configs/optimal_sweep.yaml --name optimal_sweep`.
- Tests: 108. Existing runs get the new figures after `03_analyze.py --run <run> --control <run>_pcommon1` and `04_figures.py --run <run> --only model`.


---

# Part III — the variance read explained, and the "variance where Δ² ≤ c, mean where Δ² > c" proposal tested (2026-09-28)

*Section 12 is the explanation (the equation, the roots, the four colours of `sigma/08` panel A). Section 13 is the investigation of the proposal, done outside the code — `results/experiments/implied_weight/hybrid_investigation.py`, its printed output in `hybrid_investigation_output.txt`, figures under `<run>/figures/hybrid_investigation/` for exp2_128_mean and flagship. Nothing in `variance_weight` or the pipeline was changed; §13.4 says what each choice would change.*

## 12. How the weight is read from a variance output

### 12.1 The equation

Start with the visual channel. The network's `var_vis` output was trained on the variance of the model-averaging observer's visual-position posterior, the mixture variance (Eq. 10):

    v(w) = w·F + (1 − w)·S + w(1 − w)·Δ²

F = `fused_var` (variance of the fused estimate), S = `seg_vis_var` (variance of the segregated visual estimate), Δ = `fused_mu − seg_vis_mu` (how far the two estimates lie apart on this trial), and w the weight on the fused estimate — the same w that mixes the means, `mu = w·fused + (1 − w)·seg`. F, S and Δ are known on every trial (they are the analytical quantities the dataset stores); v̂ is the network's `var_vis` on that trial. The one unknown is w. Rearranged:

    Δ²·w² − (Δ² − c)·w − (S − v̂) = 0,        c = S − F > 0

c is the variance fusion saves: about 2.4 deg² on the median trial of exp2_128_mean, about 7.9 deg² on the flagship's. In the code (`analysis.variance_weight`): `a = Δ²`, `b = a − c`, the discriminant `disc = b² + 4a(S − v̂)`, roots `r± = (b ± √disc)/(2a)`.

### 12.2 The shape of v(w), and why there are three cases

v(w) is a parabola in w that opens downward (the w² coefficient is −Δ²), with v(0) = S, v(1) = F and its peak at

    w* = b/(2a) = ½ − c/(2Δ²).

Everything follows from where that peak sits relative to [0, 1]:

- **Δ² ≤ c (small disparity, including Δ ≈ 0): the peak is at or left of w = 0.** On [0, 1] the parabola only falls, from S to F, so every v̂ between F and S corresponds to exactly one weight. `r_minus` lies left of the peak, i.e. at or below 0, and can never be a weight; `r_plus` is the root on the right and is taken (flag 0). At Δ = 0 exactly the parabola is the straight line v = S − c·w and w = (S − v̂)/c (the code takes this limit when Δ² is numerically zero). The sensitivity |dv/dw| = |Δ²(1 − 2w) − c| equals c at Δ = 0: the read is at its sharpest exactly where the ratio (sensitivity |Δ|) is blind. 42 % (exp2_128_mean) / 39 % (flagship) of the test trials are in this case.
- **Δ² > c (large disparity): the peak is inside (0, ½).** Now v(w) first rises from S (w = 0) to its peak and then falls to F (w = 1), so a v̂ between S and the peak is produced by two weights — `r_minus` on the rising side (a near-segregation weight) and `r_plus` on the falling side (a larger weight). The variance output alone cannot tell them apart; that is the only role of the `hint`: the root nearer to the hint is taken (flag 1; with no hint, `r_plus`). The hint does not need to be accurate — it only has to fall on the correct side of the midpoint between the two roots, which are a median 0.94 apart on exp2_128_mean (10th percentile 0.41) and 0.98 (0.58) on the flagship. A hint with an sd of 0.1–0.3 in w picks the right root on 92–98 % of these trials (§13.2). 56 % / 58 % of trials are in this case, and on 63–67 % of them the chosen root is `r_minus` (the segregation trials), so the no-hint default would be wrong more often than right: in this regime the hint is essential.
- **disc < 0: v̂ is above the peak.** The network reported a larger variance than any mixture of this trial's components can have (noise). There is no real root; the code returns the peak position w* — the weight whose variance is closest to v̂ — and flag 2. 1.9 % / 2.3 % of trials; on exp2_128_mean these read 0.35 on average against a posterior of 0.43 (sd 0.12).

A worked example with F = 2.2, S = 4.6 (c = 2.4): Δ = 1 (Δ² = 1 < c), v̂ = 3.0 → the unique root 0.746. Δ = 3 (Δ² = 9 > c), v̂ = 5.0 → roots 0.667 and 0.067; a hint of 0.6 (anything above the midpoint 0.367) selects 0.667, a hint of 0.1 selects 0.067. Δ = 3, v̂ = 6.5 → the peak variance is only 6.1, so flag 2 and w = w* = 0.367.

Nothing is clipped or dropped: a v̂ slightly above S on a segregation trial gives a slightly negative w (12 % of exp2_128_mean's trials, 16 % of the flagship's, all but 0.1 % / 0.9 % of them within −0.05), a v̂ slightly below F gives w slightly above 1 (0.5 % / 1.5 %). These are the per-trial noise of the read, and they are what the "outside [−1, 2]" fraction (0.0 %) and the sd figures include.

### 12.3 What the four colours of `sigma/08` panel A are

Every one of them is computed on every trial; none selects or drops trials, and "both" always means *combined by precision*, never *chosen*:

- **red — ratio from `mu_vis`**: `w = (mu_vis − seg_vis_mu)/Δ_v`, the estimator of figure 05; nominal sd σ_out(mu_vis)/|Δ_v|.
- **blue — joint read, both position outputs**: the least-squares w that best explains the two position outputs at once, `w = (Δ_v·r_v + Δ_p·r_p)/(Δ_v² + Δ_p²)` with r_v, r_p the two residuals `mu − seg`; sd σ_out/√(Δ_v² + Δ_p²). This is the `hint` the variance read uses.
- **orange — variance read, both variance outputs**: the root from `var_vis` (§12.2, with the joint read as hint) and the root from `var_prop` (the same equation with the prop components), each with a nominal sd σ_out(var)/sensitivity, combined by inverse-variance weighting (`_precision_combine`; sds floored at 1e-3). Panels B and C of the figure and the coherence test use this one.
- **black — all four combined**: the same inverse-variance combination over the two position ratios and the two variance roots. Because the variance roots' nominal sd (median 0.004) is far smaller than the ratios' on all but the largest-|Δ| trials, black is nearly identical to orange (sd 0.073 vs 0.072, 0.179 vs 0.175); it only says that adding the position reads changes nothing.

In the pipeline, `04v_fusion_weight_variance_read` draws the `var_vis` root and the `var_prop` root separately (bold), with the two position reads faded, and `08v` panel B regresses the `var_vis` root on the posterior; `metrics.json` → `per_trial.estimators` holds the summaries of all four, `per_trial.variance_root_flags` the counts of flags 0/1/2.

## 13. The proposal: variance root where Δ² ≤ c, the mean output where Δ² > c

### 13.1 Two ways to read it

The rule has a unique answer where the variance read has one root, and turns to `mu_vis` where it has two. There are two ways to use `mu_vis` there, and both were run:

- **(a) replace**: where Δ² > c, the weight *is* the ratio `(mu_vis − seg_vis_mu)/Δ_v`; the variance root is not used on those trials.
- **(b) pick**: where Δ² > c, the two variance roots stay, and the `mu_vis` ratio (instead of the joint read across both position outputs) decides which of them — the current function with `hint = ratio_vis` instead of `hint = joint`.

Both make the read a *pure visual-channel read* (`var_vis` + `mu_vis`, nothing from the prop outputs), which the current code is not (its hint uses `mu_prop` too). For completeness, (a′) = (a) with the joint read instead of the single ratio was also run.

### 13.2 Results (test splits, every trial, nothing filtered)

| run | estimator | outside [−1, 2] | sd vs posterior, all | Δ² ≤ c | Δ² > c | corr | weight-regression slope [95 % CI] | w at p ≈ 0.05 / 0.5 / 0.85 / 0.95 |
|---|---|---|---|---|---|---|---|---|
| exp2_128_mean | ratio from mu_vis | 4.3 % | 24 | 37 | 0.154 | 0.03 | 2.00 [0.44, 3.55] | 0.02 / 0.55 / 2.41 / 1.66 |
| | joint position read | 1.4 % | 0.46 | 0.70 | 0.131 | 0.59 | 0.973 [0.943, 1.003] | 0.02 / 0.56 / 0.84 / 0.93 |
| | **current variance read, `var_vis`** | **0.0 %** | **0.073** | **0.038** | **0.091** | **0.979** | **0.991 [0.987, 0.996]** | 0.01 / 0.56 / 0.84 / 0.92 |
| | (a) replace by the ratio where Δ² > c | 0.0 % | 0.119 | 0.038 | 0.154 | 0.945 | 0.980 [0.972, 0.988] | 0.02 / 0.55 / 0.84 / 0.90 |
| | (b) ratio picks the root | 0.0 % | 0.076 | 0.038 | 0.094 | 0.977 | 0.988 [0.984, 0.993] | 0.01 / 0.55 / 0.84 / 0.92 |
| | (a′) replace by the joint read | 0.0 % | 0.102 | 0.038 | 0.131 | 0.959 | 0.978 [0.972, 0.985] | 0.02 / 0.56 / 0.84 / 0.90 |
| flagship | ratio from mu_vis | 9.5 % | 55 | 87 | 0.304 | 0.00 | 0.58 [−2.6, 3.8] | 0.01 / 0.53 / 0.22 / 1.14 |
| | joint position read | 5.6 % | 2.6 | 4.1 | 0.220 | 0.14 | 0.954 [0.804, 1.103] | 0.02 / 0.51 / 0.94 / 0.69 |
| | **current variance read, `var_vis`** | **0.0 %** | **0.179** | **0.140** | **0.200** | **0.897** | **0.920 [0.910, 0.931]** | 0.02 / 0.52 / 0.81 / 0.86 |
| | (a) replace by the ratio where Δ² > c | 0.2 % | 0.252 | 0.140 | 0.304 | 0.815 | 0.900 [0.886, 0.915] | 0.01 / 0.53 / 0.80 / 0.81 |
| | (b) ratio picks the root | 0.0 % | 0.204 | 0.140 | 0.237 | 0.868 | 0.901 [0.890, 0.913] | 0.02 / 0.49 / 0.80 / 0.84 |
| | (a′) replace by the joint read | 0.1 % | 0.193 | 0.140 | 0.220 | 0.881 | 0.906 [0.895, 0.917] | 0.02 / 0.51 / 0.80 / 0.84 |

(The Δ² ≤ c column is identical across the variance-based rows by construction: the proposal changes nothing there.)

Precision in the two-root regime only, by |Δ_v| band (sd of weight − posterior):

| run | \|Δ_v\| band | n | variance root (joint hint) | mu_vis ratio | joint read |
|---|---|---|---|---|---|
| exp2_128_mean | 1–2° | 611 | 0.151 | 0.295 | 0.281 |
| | 2–3° | 935 | 0.120 | 0.166 | 0.116 |
| | 3–5° | 907 | 0.091 | 0.125 | 0.088 |
| | 5–10° | 1200 | 0.023 | 0.079 | 0.062 |
| | > 10° | 667 | 0.000 | 0.045 | 0.038 |
| flagship | 2–3° | 255 | 0.269 | 0.650 | 0.343 |
| | 3–5° | 1000 | 0.269 | 0.391 | 0.288 |
| | 5–10° | 1014 | 0.261 | 0.306 | 0.263 |
| | > 10° | 2257 | 0.072 | 0.118 | 0.107 |

Root choice on the two-root trials (which root is nearer the analytical posterior, and which one each hint selects):

| run | hint | picks the nearer root | sd vs posterior when it does | when it does not |
|---|---|---|---|---|
| exp2_128_mean | joint read (current) | 97.8 % | 0.051 | 0.49 |
| | mu_vis ratio (b) | 96.5 % | 0.051 | 0.42 |
| flagship | joint read (current) | 93.4 % | 0.087 | 0.67 |
| | mu_vis ratio (b) | 91.8 % | 0.086 | 0.74 |

### 13.3 What this says

1. **Replacing the root by the mean-based ratio (a) costs precision on every large-disparity trial, on both runs.** The ratio's error is σ_out/|Δ| and shrinks only as 1/|Δ|; the variance root's error is σ_out(var)/|Δ²(1 − 2w) − c| and shrinks as 1/Δ² once the trial is clearly segregated (w → 0). In every |Δ| band of the two-root regime the variance root is the more precise of the two: by a factor 1.4–2.4 at 2–5°, and on exp2_128_mean by 3× and more beyond 5° (0.023 vs 0.079; 0.000 vs 0.045 beyond 10°); on the flagship the two converge beyond 5° (0.26 vs 0.31, 0.07 vs 0.12) because there the flagship's own incoherence, not the read's noise, dominates both. The premise "at large disparity the mean output gives w" is right — the ratio *is* readable there — but the variance output gives it better on the same trials. Overall sd 0.073 → 0.119 (exp2_128_mean) and 0.179 → 0.252 (flagship); slope 0.991 → 0.980 and 0.920 → 0.900; the flagship also gets back 0.2 % of ratios outside [−1, 2]. The binned fusion-weight curve barely moves (figure B: the ratio's noise averages out inside a disparity bin), so the difference is invisible in a figure-04-style plot and visible in the per-trial spread (figures A, C, D).
2. **Letting the mu_vis ratio pick the root (b) costs almost nothing** — sd 0.073 → 0.076 and slope 0.991 → 0.988 on exp2_128_mean, 0.179 → 0.204 and 0.920 → 0.901 on the flagship — because the choice only needs the hint on the right side of the midpoint between the roots, and the single ratio manages that on 96.5 % / 91.8 % of two-root trials against the joint read's 97.8 % / 93.4 %. This is the version that keeps the rule as stated ("solve the visual variance; where two answers remain, let the visual mean decide") and makes the read a one-channel read.
3. **Where the variance read's remaining error comes from.** On the two-root trials the read is within 0.05 (exp2_128_mean) / 0.09 (flagship) of the posterior whenever the hint picks the nearer root, and off by ~0.5 when it does not; the 2–8 % of wrong picks account for most of the sd on those trials. So in the Δ² > c regime the lever is the *quality of the root choice*, not the estimator: a better hint (the joint read, or one that also uses `var_prop`'s root) is worth more than any switch to the mean output.
4. Both runs agree on the ordering; the flagship's numbers are larger throughout because its σ_out is 1.9× (mu) and 4× (var) that of exp2_128_mean, and its position and variance outputs are not coherent above p ≈ 0.6 (§7), which is why every hint picks the wrong root more often there.

### 13.4 If one of them is to be applied (decided: (a), applied in §14)

- **(b)** is a call-site change only: `hint=w_ratio_vis` instead of `hint=w_joint` in `03_analyze.py` (block 2c′) and in `implied_weight.per_trial_weights`; `variance_weight` itself is unchanged, `04v`/`08v` and `sigma/08` follow automatically. The prop read would use `hint = ratio_prop` by the same rule.
- **(a)** is a new branch: after `variance_weight`, `w[flags == 1] = ratio_vis[flags == 1]` (and the nominal sd becomes σ_out/|Δ| on those trials), in `03_analyze.py` and `per_trial_weights`, with the figure legends changed to say which trials come from which output. If it is wanted *in addition* to the current read rather than instead of it, it can be a fifth estimator (`hybrid`) in `per_trial` and a fifth colour in `sigma/08` A.
- Either way the Δ ≈ 0 side is untouched; the existing tests keep passing for (b), and (a) needs one new case.

Figures of this investigation (per run, under `figures/hybrid_investigation/`): `A_weight_distribution_with_hybrid` (sigma/08 A with the two proposals added: purple (a), teal (b)), `B_fusion_weight_with_hybrid` (04v with both added), `C_weight_regression_current_vs_hybrid` (08v B for the current read, (a) and (b)), `D_precision_by_delta_band` (sd vs posterior by |Δ_v| band for the ratio, the joint read, the current read, (a) and (b)).

## 14. Applied: the hybrid read, option (a), is in the code (2026-09-28)

Decision taken: (a). The hybrid is now a read of its own in the pipeline and in the experiments, **beside** the variance read and the position reads — nothing was removed or replaced, so every run keeps all of them for comparison. `variance_weight` itself is unchanged.

### 14.1 What the hybrid is, exactly

For one channel (visual: `var_vis`, `mu_vis`, `seg_vis_mu`, `seg_vis_var`; the prop channel by symmetry), on every trial:

1. Solve the mixture-variance quadratic equation for the channel's variance output with **no hint**: `variance_weight(v̂, F, S, Δ)` → a root, 
2. Decide the regime from the trial's known quantities alone: `Δ² ≤ c` (c = S − F) or `Δ² > c`.
3. Where **Δ² ≤ c** — the parabola is monotone on [0, 1], the root is unique, and this includes Δ → 0 — the weight **is the variance root** 
4. Where **Δ² > c** — the variance would have two roots — the weight **is the channel's own position ratio**, `w = (mu − seg_mu)/Δ` . 

Nothing is filtered or clipped. The code is `analysis.hybrid_weight(v_hat, fused_var, seg_var, delta, mu_hat, seg_mu, sig_out_mu=None, sig_out_var=None) → (w, sigma, flags)` in `src/cmsi/analysis/causal.py`, right after `variance_weight`, which it calls for step 1.

### 14.2 Where it appears

- **Stage 3** (`scripts/03_analyze.py`, block 2c″, after the variance read): `analysis.npz` gains `fusion_weight_hybrid_vis/prop`, `sigma_w_hybrid_vis/prop`, `hybrid_flags_vis/prop`; `metrics.json` gains `weight_regression_hybrid_vis/prop` (the hybrid on the posterior: slope, CI, intercept), `implied_weight_hybrid_vs_post[_prop]` (slope, R², RMSE against the posterior, every trial) and `hybrid_read` (per channel: the share of trials from the variance root / the position ratio / the variance peak, the fraction outside [−1, 2], the sd against the posterior overall and for each part). The console prints one line per channel.
- **Stage 4** (`scripts/04_figures.py` → `viz.results`): `04v_fusion_weight_variance_read` now carries the hybrid as its own colour (purple: vis solid with triangles, prop dashed with inverted triangles) next to the variance reads (bold) and the ratio reads (faded); `08v_variance_regression` has a third panel, **C**, the hybrid of the visual channel against the posterior on every trial, each point coloured by the output it came from (purple = `var_vis` root, red = `mu_vis` ratio, grey = variance peak), the legend giving each source's share, with the fit. A and B are unchanged. **`17_hybrid_weight_distribution`** (added 2026-09-29) is the sigma analysis's distribution figure (`sigma/05`) with the hybrid in place of the ratio: the hybrid weight on every trial, one panel per channel, as a histogram against the analytical posterior's own distribution (outline), nothing filtered, the axis clipped to [−1, 2] with the overflow piled into the edge bins, and the fraction outside, the share read from the ratio, the median and the IQR printed. `06 figures` draws the same three (04v, 08v, 17) under the experiment folder. Colours are `COLORS["variance"]` and `COLORS["hybrid"]` in `viz/style.py`.
- **Experiment module** (`cmsi.experiments.implied_weight.per_trial_weights`): two new estimators, `hybrid_vis` and `hybrid_prop`, with the same summary as the others (fraction outside, sd vs posterior overall / |Δ_vis| < 1° / > 4°, correlation, median σ_w, fraction with σ_w < criterion, binned mean per posterior decile) under `per_trial.estimators`; `per_trial.hybrid_sources` (shares and the sd of each part); `per_trial.weight_regression.hybrid_vis/prop`; arrays `hybrid_flags_vis/prop`. `sigma/08` panel A draws the hybrid (vis) as the fifth distribution and panel B its binned mean next to the variance read's; `06 sigma` and `06 figures` print its rows and regression; the `06 compare`, `07`, `08` and `09 compare` tables carry a `hyb sd` column (the hybrid's sd against the posterior) next to `var sd`.
- **Tests**: 111 (3 new). On a noiseless model-averaging observer the hybrid equals the true weight on every trial, its flags follow the regime exactly (Δ = 0 is a root trial), its nominal sds are the two formulas, a ratio-regime trial is blind to its variance output and follows its position output; 04v/08v draw it in its own colour and the third panel; in `per_trial_weights` it equals the ratio on Δ² > c trials and the variance root on the others, and it reaches `sigma/08`.
- **Docs**: README (experiments section and pipeline outputs), GUIDE (04v, 08v, side experiments).

### 14.3 What it gives on the two runs (test splits, every trial)

Produced with the delivered code on your `analysis.npz` (the same numbers as the §13 investigation, now from the pipeline functions):

| run | read | outside [−1, 2] | sd vs posterior | of which root part / ratio part | corr | slope on the posterior [95 % CI] | intercept |
|---|---|---|---|---|---|---|---|
| exp2_128_mean | variance read, `var_vis` | 0.0 % | 0.073 | — | 0.979 | 0.991 [0.987, 0.996] | +0.003 |
| | **hybrid, vis** | **0.0 %** | **0.119** | 0.038 (42 % of trials) / 0.154 (58 %) | 0.945 | 0.980 [0.972, 0.988] | +0.010 |
| | hybrid, prop | 0.0 % | 0.109 | — | 0.954 | 0.982 [0.975, 0.989] | +0.014 |
| flagship | variance read, `var_vis` | 0.0 % | 0.179 | — | 0.897 | 0.920 [0.910, 0.931] | +0.023 |
| | **hybrid, vis** | **0.2 %** | **0.252** | 0.140 (39 %) / 0.304 (61 %) | 0.815 | 0.900 [0.886, 0.915] | +0.025 |
| | hybrid, prop | 1.7 % | 0.468 | — | 0.636 | 0.990 [0.963, 1.018] | −0.003 |

Binned by posterior the hybrid (vis) reads 0.02 / 0.44 / 0.84 / 0.90 at p ≈ 0.05 / 0.5 / 0.85 / 0.95 on exp2_128_mean (variance read 0.01 / 0.46 / 0.84 / 0.91) and 0.01 / 0.50 / 0.80 / 0.81 on the flagship (0.02 / 0.51 / 0.81 / 0.86): the binned curves — `04v`, `sigma/08` B — are the variance read's to within a few hundredths, because the ratio's noise averages out inside a bin. The difference is per trial: on the Δ² > c trials the hybrid carries the ratio's sd (0.154 / 0.304) where the variance root had 0.091 / 0.200, which is what panel C of `08v` shows as the wider red cloud. On the Δ² ≤ c trials the two reads are identical by construction.

Two things to keep in mind when reading it. The hybrid's σ_w column is now meaningful on the ratio trials (σ_out/|Δ|, median 0.03 on exp2_128_mean, 77 % of trials below the 0.1 criterion), whereas the variance read's nominal σ_w (0.003) is the control-based figure that §7 already noted understates the real error. And the prop channel's hybrid on the flagship is the weakest read of all (sd 0.47, 1.7 % outside, corr 0.64; its slope of 0.99 comes with a wide CI and a low correlation, a noisy fit rather than a good one): the disparity is split between the channels (median |Δ_prop| 1.9° against |Δ_vis| 3.9°) and c_prop is small (1.8 deg², so the ratio regime begins at |Δ_prop| > 1.35°, where σ_out/|Δ| is already 0.63/1.35 ≈ 0.46), so the prop ratio is poorly conditioned even beyond √c — the same reason `ratio_prop` has 17 % outside there. On exp2_128_mean the prop hybrid is as good as the visual one.

### 14.4 Regenerating existing runs

Runs analysed before this change get the hybrid after `python scripts/03_analyze.py --run <run> --control <run>_pcommon1` (the flagship: `--control pcommon1`) and `python scripts/04_figures.py --run <run> --only model`; `python scripts/06_implied_weight.py sigma --run <run>` redraws `sigma/08`, `… figures --run <run>` the experiment's 04v/08v. The figures produced today for exp2_128_mean and flagship are the experiment's copies, under `results/experiments/implied_weight/<run>/figures/figures/` (04v, 08v) and `…/figures/sigma/` (08); the pipeline's own copies under `results/<run>/figures/model/` appear once stage 3 and 4 are re-run on the Mac (they need the checkpoint).


---

# Part IV — the cleanup: the hybrid read is the pipeline's weight (2026-09-29)

The investigation is concluded: the weight is read by the hybrid of each channel, and the codebase was cut back to that. Nothing that was removed is needed to reproduce the current read; the results the removed experiments produced stay on disk.

## 15. What was removed

- **Experiments.** `experiments/architecture.py`, `sweep.py`, `fixed_variance.py`, `design.py` and the scripts `07_architecture.py`, `08_fixed_variance.py`, `09_sweep.py`; the `sigma`, `compare` and `design` sub-commands of `06_implied_weight.py`; from the experiment module, the readability block (`readability`, `predicted_outside`, `flat_posterior_weights`), the seven-estimator `per_trial_weights` with its coherence test, the eight sigma figures, `print_per_trial`. Their results stay under `results/experiments/{architecture,fixed_variance,sweep}/`, `data/experiments/` and `results/experiments/implied_weight/<run>/` (the retired figures of exp2_128_mean and flagship were moved into `_retired_figures/` there); the story is in Parts I–III above.
- **Estimators.** From `analysis/causal.py`: the per-trial ratio `fusion_weight` with `min_separation`, the `sigma_w` filter and the `sigma_w_criterion`, the joint read across both position outputs, the least-squares per-bin weight `binned_implied_weight`, the two-estimator `implied_weight_by_posterior`, the filtered `weight_consistency`. The variance-only read is no longer a read of its own: `variance_weight` stays as the solver the hybrid calls (no hint parameter), returning the unique root where `Δ² ≤ c` and flagging the two-root regime.
- **Config keys.** `min_separation` and `sigma_w_criterion` were removed from every config (they guarded the ratio).
- **Figures.** `04v` (the variance read beside the ratio) and the two-estimator `15/16`; the sigma-analysis figures.

## 16. What the pipeline reads now

`analysis.hybrid_weight` (Part III, §14.1) is the one per-trial weight. Stage 3 stores the visual and the proprioceptive read as `fusion_weight` and `fusion_weight_prop` in `analysis.npz`, with `sigma_w_vis/prop` (the nominal sd) and `hybrid_flags_vis/prop`; `metrics.json` has `implied_weight_vs_post[_prop]` (every trial, no filter), `weight_regression_vis/prop`, `hybrid_read`, and — unchanged — the two no-division headlines `position_regression_vis/prop` and `variance_regression_vis/prop`. Everything downstream that needs a weight uses it: `weight_consistency` (the two channels' reads, trial by trial, all trials), `reliability_within_disparity`, the transition fit, the behavioural conditioning (`inferred_common` = the visual read > 0.5), the reliability curves of figure 05 and of `06 reliability`, and the prior sweep (`05_prior_sweep.py`: the per-bin mean of the hybrid with its standard error, `analysis.binned_weight`; the stored `t_w_vis` lets `--replot` re-bin). Stage 4 draws `04` (the hybrid per channel, through zero disparity), `05`, `08` and `08v` (variance-domain regression + the weight on the posterior, points coloured by source), `15` (the weight in posterior bins, every trial, `analysis.weight_by_posterior`) and `16` (its distribution per channel against the posterior's; the former 17). The experiment keeps `figures`, `reliability` and `train`. Tests: 97.

## 17. The numbers, refreshed with the hybrid read

Computed on your Mac from the stored `analysis.npz` and datasets with the delivered code (visual channel; the GUIDE's §7.2, §7.4, §7.5, §7.11 and §8.1 carry the same). The stored `metrics.json` blocks and model figures of these runs still come from the earlier estimators until stage 3 and 4 are re-run (`03_analyze.py --run <run> --control pcommon1`, `04_figures.py --run <run> --only model`).

| run | prior | from the ratio | outside [−1, 2] | sd vs posterior (root / ratio) | slope on the posterior [CI] | position / variance regression | vis–prop consistency (corr, mean\|diff\|) | reliability slope | midpoint net / analytical |
|---|---|---|---|---|---|---|---|---|---|
| pcommon028 | 0.28 | 74 % | 0.1 % | 0.211 (0.188 / 0.215) | 0.840 [0.826, 0.855] | 0.797 / 0.815 | 0.45, 0.25 | 0.50 ± 0.03 | 4.22 / 5.29 |
| flagship | 0.5 | 61 % | 0.2 % | 0.252 (0.140 / 0.304) | 0.900 [0.886, 0.915] | 0.856 / 0.871 | 0.47, 0.28 | 0.61 ± 0.04 | 7.62 / 8.04 |
| pcommon07 | 0.7 | 49 % | 0.3 % | 0.263 (0.117 / 0.355) | 0.918 [0.902, 0.934] | 0.927 / 0.866 | 0.35, 0.32 | 0.93 ± 0.03 | 10.05 / 9.93 |
| exp2_128_mean | 0.5 | 58 % | 0.0 % | 0.119 (0.038 / 0.154) | 0.980 [0.972, 0.988] | 0.995 / 0.970 | 0.91, 0.08 | 0.92 ± 0.02 | 5.07 / 5.08 |

Two things changed with the read, both for the better and both worth knowing. The reliability-within-disparity test now uses every trial, so it keeps 5–7 of its 8 disparity bins instead of 2–3 (its combined slope moved: flagship 0.57 → 0.61, pcommon028 0.91 → 0.50, pcommon07 0.71 → 0.93; exp2_128_mean 0.92 ± 0.02), and the per-bin slopes show where the flagship under-tracks reliability (the 4–6° bin). The transition midpoints (4.22 / 7.62 / 10.05) replace the filtered-ratio ones (5.90 / 7.72 / 10.13) and still track the analytical values monotonically. The consistency between the two channels is now measured on all 7,500 trials and is dominated by the proprioceptive channel's ratio trials (its `Δ_prop` is the smaller half of the disparity, so `σ_out/|Δ|` is large there): mean |diff| 0.25–0.32 on the flagship family, 0.08 on exp2_128_mean; the visual read is the one to report.

## 18. Where things are

`analysis/causal.py` (`hybrid_weight`, `variance_weight`, `binned_weight`, `weight_by_posterior`, `weight_regression`, `weight_consistency`, `variance_regression`), `scripts/03_analyze.py` block 2, `scripts/04_figures.py`, `viz/results.py` (`fusion_weight_curve`, `variance_regression_figure`, `weight_vs_posterior`, `hybrid_weight_distribution`), `scripts/05_prior_sweep.py`, `experiments/implied_weight.py` (`weight_analysis`, `weight_by_reliability`, `baseline_figures`, `print_summary`), `scripts/06_implied_weight.py` (`figures`, `reliability`, `train`). README and GUIDE describe the current state; `configs/optimal_sweep.yaml` keeps the recipe the sweeps converged on.
