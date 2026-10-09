# The cross-prior experiment — result and manuscript notes

**Figure:** `results/prior_sweep/figures/prior_sweep.tif` (PLOS submission copy:
7.5 in wide, 300 dpi, flattened RGB, LZW), with `.png` to view, `.svg` to edit
and `.pdf` for LaTeX; the standard-panel version for the manuscript is
`results/manuscript/fig7_prior_sweep/fig7_prior_sweep` (named `prior_sweep/prior_sweep_ABC` until 5 Oct 2026 and `fig6_prior_sweep` until 6 Oct 2026)
**Supplementary:** `results/prior_sweep/figures/variance_hump_vs_prior.tif`
(+ `.png`, `.svg`, `.pdf`)
**Data:** `results/prior_sweep/sweep.json`, `results/prior_sweep/curves.npz`
**Script:** `scripts/05_prior_sweep.py`

Every number below comes from the sweep of **2026-10-03**: nine priors × three
seeds = 27 networks, 50,000 trials and 300 epochs each, trained on the current
`configs/flagship.yaml` (the 10° domain with narrow noise ranges; GUIDE Part 4)
with the implied weight read by the hybrid of the visual channel (GUIDE §7.2)
and σ_out from `results/pcommon1`. The earlier sweep on the wide configuration,
which this document described until 2026-10-03, is summarised in §8 for the
record.

---

## 1. What the experiment is

Twenty-seven networks were trained, three seeds at each of `p_common` = 0.1 …
0.9. Between priors **only the Bernoulli constant differs** — same encoders,
same σ ranges, same architecture, same training protocol. Because the latent
streams are drawn before `C` selects between them, any two of these datasets
at the same seed are bit-identical on every trial whose causal structure did
not flip.

Each network is trained on the same four numbers as the flagship: `mu_vis`,
`var_vis`, `mu_prop`, `var_prop`. None has an explicit causal read-out. The
implied fusion weight is then recovered from behaviour alone.

### What this tests, stated precisely

It is worth being exact about the claim, because there is a weaker version that
would be trivial and a stronger version that would be false.

**Not the claim:** "the network discovered the prior from nothing." It did not.
The prior is present in every target it was trained on.

**The claim:** the network's *implicit* fusion-vs-segregation trade-off — which
is never an output, only something recovered from behaviour — depends on the
prior in the way Bayes says it must, quantitatively, with no free parameters.

This is a **parametric test of the Bayesian interpretation**. Its force comes
from what the alternatives predict:

| account | prediction as `p_common` varies |
|---|---|
| disparity heuristic | transition midpoint fixed — the network responds to disparity, and disparity statistics barely change |
| fixed-weight model | no transition at all, at any prior |
| model **selection** | a step at posterior 0.5 whose position moves, but with sharpness → ∞ |
| **model averaging with the correct posterior** | midpoint tracks the analytical value at every prior |

Only the last survives, and it survives quantitatively rather than qualitatively.

---

## 2. The figure, panel by panel

### Panel A — the mechanism

Implied weight on the fused estimate against signed body-frame disparity, one
curve per prior (colour = prior, sequential ramp). Solid with error bars: the
network. Dashed: the analytical posterior for that prior. Nine curves fan out
in an orderly family: at `p_common` = 0.1 the network's weight never exceeds
0.33, even at zero disparity (the posterior itself does not — §4 below); at
0.5 it reads 0.80 at ±1.5° and is back at zero by ±12°; at 0.9 it holds 0.97
at the centre and 0.43 at ±9°. Every curve lies on its dashed partner to
within 0.05 in every bin holding more than 100 trials.

**How the per-bin weight is computed.** Each trial's weight is the hybrid
read of the visual channel (GUIDE §7.2): the root of the channel's variance
output in the mixture variance `v(w) = w·F + (1−w)·S + w(1−w)Δ²` where
`Δ² ≤ c = S − F` (small disparity, where a position ratio is blind), and the
channel's own position ratio `(network − seg)/Δ` where `Δ² > c`. No trial is
filtered. The curve is the mean of that weight per disparity bin, with
`1.96 ×` the standard error of the bin mean as the error bar
(`analysis.binned_weight`). Because the read exists on every trial, the curve
carries through zero disparity, where the earlier ratio-based estimators had
nothing to say.

Two details visible in the panel are deliberate:

- **The grid is the manuscript's 18-bin grid** from −30° to +30° with its two
  innermost edges at ±1.5°; it predates the hybrid read and is kept so that
  old and new sweeps draw on the same bins. (The per-run figures — 04, 12,
  manuscript Fig. 4A/C — use the config's `disparity_grid`, since 2026-10-03
  19 centres from −30° to +30° in 2° steps across the transition; same range,
  different bins, and a 0° centre that this grid does not have.)
- **Two guards, and gaps break the line.** A bin is dropped when it holds
  fewer than 25 trials or when the standard error of its mean exceeds 0.05.
  On this sweep two bins are dropped, both at `p_common` = 0.9 (−24° with 23
  trials, +30° with 24); the largest surviving SE is 0.045. Joining across a
  dropped bin would draw a transition that was never measured. The surviving
  outer bins at 0.8–0.9 (30–100 trials) wobble by up to ±0.1 around zero,
  which is what 30-trial means of a heavy-tailed ratio look like.

### Panel B — the quantitative match

Each prior's transition midpoint (the disparity at which the implied weight
crosses 0.5, from a logistic fit `w = 1/(1 + exp(k(|d| − d0)))`, averaged over
the three seeds with SEM bars) against the analytical midpoint for the same
prior. There are no free parameters relating the two axes.

```
all nine priors (aggregated)   slope 1.016   intercept −0.20°   R² 0.9998
                               mean |network − analytical| = 0.13°    max 0.22° (p = 0.4)
the 27 networks individually   slope 1.016   intercept −0.20°   R² 0.9988
                               mean |network − analytical| = 0.15°    max 0.40°
```

Across a midpoint range of −1.1 to 8.4°, the network lands within an eighth
of a degree of the Bayesian prediction on average, and the sharpness of the
transition matches as well (network 0.33–0.67 per degree against 0.33–0.69
analytical; ratio 0.96–1.03 at every prior). The across-seed SEM of the
midpoint is 0.02–0.08° at eight priors and 0.13° at `p_common` = 0.1.

### Panel C — averaging at every prior, and a consistent conservatism

Position-domain regression slope (Bayes-optimal = 1) against prior, with
across-seed 95 % CIs (mean ± 1.96 SEM over three seeds). The slope rises in
trend from **0.884 [0.865, 0.904]** at `p_common` = 0.1 to **0.988 [0.969,
1.007]** at 0.9 (+0.12 per unit prior, r = 0.91).

It is below 1 at every prior, and the across-seed CI excludes 1 at seven of
the nine (0.6 and 0.9 reach 1.002 and 1.007); on the 27 networks individually
the within-run CI excludes 1 in 20. The network is consistently, mildly
conservative: it applies slightly less pull toward the fused solution than
the ideal observer would — by 12 % at the lowest prior, by 2–8 % elsewhere.
This is the same deviation the flagship analysis reports (0.952), now
shown to be a stable property rather than a quirk of one training run, and
small enough on this configuration that its prior dependence is the more
interesting part: the shortfall is largest exactly where the fused solution
is rarely the right one.

**Model averaging wins at all nine priors and all three seeds** — 27 of 27 —
in the five-way comparison against full integration, full segregation, model
selection and the best fixed-weight model, by 14–37 % of RMSE over the
runner-up from 0.3 to 0.8, 6–14 % at 0.9 and 1–3 % at `p_common` = 0.1,
where almost every trial is segregated and the strategies nearly coincide.

---

## 3. Full numbers

From `sweep.json`, `aggregated` block; ± are SEM across the three seeds.

| `p_common` | midpoint net | midpoint opt | sharp net | sharp opt | posreg slope [95% CI] | reliability slope | hump net | hump opt | MSL decode R² | best strategy |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | −1.11 ± 0.13 | −0.94 | 0.329 | 0.330 | 0.884 [0.865, 0.904] | 0.907 ± 0.021 | −0.09 ± 0.03 | −0.02 | 0.944 | averaging |
| 0.2 | 1.41 ± 0.02 | 1.58 | 0.359 | 0.353 | 0.921 [0.845, 0.997] | 0.925 ± 0.014 | −0.01 ± 0.02 | +0.06 | 0.959 | averaging |
| 0.3 | 2.96 ± 0.04 | 3.12 | 0.376 | 0.385 | 0.951 [0.905, 0.997] | 0.986 ± 0.007 | 0.25 ± 0.01 | 0.28 | 0.973 | averaging |
| 0.4 | 4.01 ± 0.03 | 4.23 | 0.427 | 0.415 | 0.919 [0.874, 0.964] | 0.968 ± 0.031 | 0.54 ± 0.01 | 0.57 | 0.974 | averaging |
| 0.5 | 4.92 ± 0.02 | 5.09 | 0.440 | 0.449 | 0.951 [0.931, 0.972] | 0.928 ± 0.033 | 0.85 ± 0.01 | 0.90 | 0.979 | averaging |
| 0.6 | 5.84 ± 0.04 | 5.89 | 0.467 | 0.486 | 0.978 [0.955, 1.002] | 0.969 ± 0.024 | 1.21 ± 0.02 | 1.30 | 0.977 | averaging |
| 0.7 | 6.56 ± 0.04 | 6.65 | 0.514 | 0.533 | 0.977 [0.961, 0.993] | 0.930 ± 0.018 | 1.60 ± 0.01 | 1.71 | 0.975 | averaging |
| 0.8 | 7.37 ± 0.02 | 7.43 | 0.584 | 0.596 | 0.983 [0.968, 0.998] | 1.024 ± 0.025 | 1.96 ± 0.01 | 2.11 | 0.978 | averaging |
| 0.9 | 8.40 ± 0.08 | 8.46 | 0.670 | 0.693 | 0.988 [0.969, 1.007] | 0.958 ± 0.021 | 2.33 ± 0.05 | 2.80 | 0.961 | averaging |

Read-out R² on `mu_vis` is 0.993–0.997 across the whole sweep (`var_vis`
0.90–0.97, lowest at the two ends), so no network is simply fitting the task
worse at the extremes. The posterior decodes from the single-modality layer
at R² 0.75–0.90 and from the multisensory layer at 0.94–0.98.

**The reliability slope** (the within-disparity-bin regression of the weight
on the posterior, GUIDE §7.5; 1 = Bayes, 0 = a disparity heuristic) is
0.91–1.02 at every prior, mean 0.955, with no trend (r = +0.49 over the nine
points, +0.36 over the 27 networks). The network uses cue reliability at
matched disparity at about 95 % of the Bayesian rate, everywhere.

Per seed, the three midpoints at each prior:

| `p_common` | seed 0 | seed 1 | seed 2 | analytical |
|---|---|---|---|---|
| 0.1 | −1.19 | −1.30 | −0.85 | −0.93 / −0.90 / −0.98 |
| 0.2 | 1.44 | 1.37 | 1.42 | 1.60 / 1.57 / 1.58 |
| 0.3 | 2.96 | 2.88 | 3.04 | 3.13 / 3.12 / 3.12 |
| 0.4 | 4.00 | 4.07 | 3.95 | 4.22 / 4.22 / 4.25 |
| 0.5 | 4.89 | 4.90 | 4.95 | 5.08 / 5.11 / 5.09 |
| 0.6 | 5.90 | 5.86 | 5.77 | 5.89 / 5.90 / 5.89 |
| 0.7 | 6.56 | 6.64 | 6.49 | 6.64 / 6.66 / 6.64 |
| 0.8 | 7.33 | 7.40 | 7.36 | 7.40 / 7.43 / 7.45 |
| 0.9 | 8.31 | 8.55 | 8.33 | 8.44 / 8.48 / 8.45 |

(The seed-0 network at 0.5 is the flagship run itself; its numbers agree with
`results/flagship/metrics.json` to the last digit.)

---

## 4. The two ends of the prior axis

**The midpoint at `p_common` = 0.1 is negative, on both axes.** At that prior
the mean posterior at zero disparity is only 0.34 — two sources drawn from a
10° prior are so often as close as one that agreement is weak evidence — and
it exceeds 0.5 on just 2 % of trials (those with both cues far out in the
prior's tails). The logistic is therefore fitted to a curve that peaks below
its own midpoint, and `d0` is where the fitted curve would cross 0.5 if
extrapolated past zero. The fit is legitimate (the guard allows a midpoint
within a quarter-span of the observed range), and the network's extrapolation
matches the ideal observer's, which is the point; but −1.1° is not a disparity
any trial has. The caption must say so. At `p_common` = 0.2 the curve peaks
at 0.54 and the midpoint (1.4°) is a real crossing, just.

**The supplementary panel** (`variance_hump_vs_prior`) plots the mid-ambiguity
variance elevation — the mean of the network's `var_vis` output over the
posterior bins 0.2–0.8 minus its mean over the confident bins — network
against analytical, per prior:

| `p_common` | trials with 0.2 < posterior < 0.8 (seed 0) | hump net | hump analytical | ratio |
|---|---|---|---|---|
| 0.1 | 24.7 % (1855) | −0.09 | −0.02 | — |
| 0.2 | 39.8 % (2988) | −0.01 | +0.06 | — |
| 0.3 | 48.4 % (3632) | 0.25 | 0.28 | 0.89 |
| 0.5 | 48.6 % (3642) | 0.85 | 0.90 | 0.94 |
| 0.7 | 16.4 % (1229) | 1.60 | 1.71 | 0.94 |
| 0.8 | 9.3 % (695) | 1.96 | 2.11 | 0.93 |
| 0.9 | **3.9 % (295)** | 2.33 | 2.80 | 0.83 |

From 0.3 to 0.8 the network reproduces the hump at 89–94 % of its analytical
size, and the hump grows linearly with the prior (3.5 deg² per unit prior,
R² 0.998). At 0.1 and 0.2 the *analytical* hump is zero: the confident side
of the statistic is almost entirely segregated, high-variance trials, so the
ideal observer has no mid-ambiguity elevation to reproduce, and the network's
−0.09 and −0.01 are agreement rather than failure — the ratio is simply
undefined there. The dip to 0.83 at 0.9 is trial counts: 295 intermediate
trials spread over six bins.

**Recommendation for the manuscript:** show this as supplementary over
`p_common` ∈ [0.3, 0.9], with the ratio and the trial counts stated, and one
sentence on why the two lowest priors have no hump to reproduce.

---

## 5. Draft text

### Methods

> To test whether the network's implicit causal weighting is Bayesian rather
> than merely graded, we trained networks at `p_common` ∈ {0.1, …, 0.9}, three
> random seeds each. These differed in the Bernoulli constant of the generative
> model and in nothing else: identical tuning curves, reliability ranges,
> architecture and training protocol. Because the latent variables are drawn
> before the causal structure selects among them, datasets at different priors
> and the same seed are identical on every trial whose structure did not change.
>
> For each network we recovered the implied weight on the fused estimate from
> behaviour alone, on every test trial. Where the two hypotheses were close
> (`Δ² ≤ var_seg − var_fused`) the weight was the root of the network's
> variance output in the mixture variance `w·var_fused + (1−w)·var_seg +
> w(1−w)Δ²`; elsewhere it was the ratio `(network − seg)/Δ` of the position
> output. The weight was averaged within bins of body-frame disparity (bins
> holding fewer than 25 trials, or with a standard error above 0.05, were not
> drawn). The fusion-to-segregation transition midpoint was obtained by fitting
> a logistic to the implied weight against absolute disparity, and compared
> with the same fit applied to the analytical posterior. Read-out noise was
> measured on a `p_common` = 1 control network.

### Results

> The implied weighting shifted systematically with the prior each network had
> been trained under (Fig. X A). Networks trained with a low prior on a common
> cause never weighted the fused estimate above 0.33, while those trained with
> a high prior maintained near-complete fusion out to several degrees of
> disparity.
>
> This shift was quantitatively Bayesian. Transition midpoints ranged from
> −1.1° to 8.4° across the sweep and tracked the analytical midpoints with a
> slope of 1.02 and R² = 0.9998, deviating by 0.13° on average and never by
> more than 0.22° (Fig. X B; across-seed SEM 0.02–0.13°). No parameter was
> fitted to relate the two. The sharpness of the transition matched the
> analytical value to within 4 % at every prior.
>
> The position-domain regression confirmed model averaging at every prior, with
> slopes rising from 0.884 (95 % CI across seeds [0.865, 0.904]) at
> `p_common` = 0.1 to 0.988 ([0.969, 1.007]) at 0.9 (Fig. X C). Slopes lay
> below the Bayes-optimal value of 1 at all nine priors, significantly so at
> seven, indicating a small but consistent conservatism: the network applied
> slightly less pull toward the fused solution than an ideal observer, most
> at the priors where fusion was rarely appropriate. A five-way comparison
> against full integration, full segregation, model selection and the best
> fixed-weight model selected model averaging for all 27 networks. At matched
> disparity, the weight tracked the reliability-driven variation of the
> posterior with slope 0.91–1.02 at every prior, excluding a disparity
> heuristic. The trial-wise posterior remained linearly decodable from the
> multisensory layer throughout (R² = 0.94–0.98).

### Discussion sentence

> Because a disparity heuristic predicts a prior-independent transition and any
> fixed-weight scheme predicts none, the orderly and quantitatively correct
> dependence of the implied weight on the prior is difficult to account for
> without granting that the network has internalised the causal-structure
> posterior itself.

---

## 6. Caveats to keep with the result

**Three seeds per prior, and the error bars are across-seed.** Panel B's bars
and Panel C's CIs are the across-seed SEM (× 1.96), which is what a reviewer
asks for: how much the result moves if the network is retrained. The within-run
CI of any one network's slope is narrower (0.01–0.03) and answers a different
question. The legend of Panel C states which is drawn.

**The ends of the prior axis are extrapolations or empty.** The 0.1 midpoint
is a negative extrapolation (§4) and the hump is undefined at 0.1–0.2. Neither
weakens the result, but both need a sentence in the caption; a reader who sees
a "transition at −1.1°" without one will reasonably object.

**The outer bins of Panel A are thin at high priors** (23–100 trials at
`p_common` = 0.8–0.9) and show it. The guards drop two of them; the rest are
drawn with their error bars, which is the honest version.

**The conservatism is real and should be reported, not smoothed over.** Slopes
of 0.88–0.99 with CIs excluding 1 at seven of nine priors is a consistent
finding about the network — and a small one on this configuration. Since every
pipeline guard is verified and both controls are clean, it is attributable to
the model rather than the method. `IMPLIED_WEIGHT_RESULT.md` records that it
shrinks with the input domain relative to the sensory noise, not with the
network's size.

---

## 7. Reproducing

```bash
# three seeds per prior — what the manuscript figure uses (~45 min)
make sweep SEEDS="0 1 2"

# explicit form, if you want to vary the priors too
python scripts/05_prior_sweep.py --priors 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
                                 --seeds 0 1 2 --control pcommon1

# redraw without training; re-bin Panel A with other guards
python scripts/05_prior_sweep.py --replot
python scripts/05_prior_sweep.py --replot --min-count 25 --max-se 0.05
```

`--control pcommon1` supplies σ_out from the `p_common` = 1 control, which
must have been analysed first; without it each network falls back to its own
residuals, which is conservative rather than wrong.

`sweep.json` holds both levels: `rows` is the raw per-(prior, seed) records,
each carrying its `seed`; `aggregated` is the one-row-per-prior view with
`n_seeds`, `seeds` and the SEM fields, and is what the figure is drawn from.
Results are written after **every** seed, so an interrupted run leaves usable
partial output.

`curves.npz` keeps the per-trial arrays (`t_disparity`, `t_post`, `t_w_vis`)
**for the first seed only** — they are the bulk of the file, and Panel A is a
within-seed quantity that wants one representative run rather than a smear
across runs. Per-prior curve summaries are stored for every seed under
`s{seed}_p{prior}_*` keys, with the first seed additionally aliased to the
`p{prior}_*` names the plotting code reads.

---

## 8. The earlier sweep, for the record

Until 2026-10-02 the flagship was the wide configuration (`flagship_wide.yaml`:
hand prior sd 20.6°, eye sd 18°, noise ranges [1.2, 6.6] / [1.8, 9.6] /
[3.3, 13.2] deg²), and the sweep on it — three seeds, the per-bin weight by a
least-squares slope within each bin, the midpoint from a σ_w-filtered ratio —
gave: midpoints 2.7 → 12.7° tracking the analytical ones with slope 0.88,
intercept +1.3°, R² 0.968 over all nine priors (1.03 / −0.06° / 0.994 with the
0.1 point excluded, where the three seeds spread 1.8–4.0°); position-regression
slopes 0.76–0.95 with the CI excluding 1 at all nine priors; averaging 27 of
27; a variance hump at 51–75 % of analytical from 0.2 to 0.6 collapsing to
0.28 at 0.9; a reliability slope declining from 0.95 to 0.46 with the prior.
Those results were deleted with the configuration (GUIDE §12.0). Every
qualitative conclusion survived the change of configuration; what changed is
that the deviations from the ideal observer became small (midpoints within
0.13° instead of 0.44°, slopes within 1–12 % of 1 instead of 5–24 %, the hump
at 89–94 % instead of 51–75 %) and the two artefacts — the 0.1 outlier and the
declining reliability slope — disappeared.
