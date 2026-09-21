# Implied weight: what sets the spikes, what the sweeps say, and how to make the weight readable

*Part I, 2026-09-20 (configuration); Part II, 2026-09-21 (readability), below. Investigation of the question "in what configuration can the weights implied from the network be brought closer to the analytical solution / into [0, 1]", following the exp2_128_mean result (4.3 % of per-trial ratios outside [−1, 2], position-regression slope 0.99) and the supervisor's options A (balanced weight distribution), B (systematic range testing) and C (fixed realistic variances). Nothing in the pipeline was changed; every new analysis lives in `src/cmsi/experiments/` and `scripts/06_implied_weight.py` (`compare`, `design`) and `scripts/09_sweep.py`.*

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
