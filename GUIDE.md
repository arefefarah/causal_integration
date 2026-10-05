# `causal_integration` — Complete Guide

**What this document is.** A walkthrough of the codebase, every config, every
analysis, and every figure — with the calculation behind each one and the number
my own runs actually produced.

**Provenance.** Every number below was read out of files in the repo:
`results/<run>/metrics.json` and `analysis.npz`,
`results/calibration/<config>/metrics.json`, `results/prior_sweep/sweep.json`
and `curves.npz`, as they stood after the rerun of the whole flagship family
on 2026-10-02/03 (Part 5, §12.0). Nothing here is estimated, remembered, or
carried over from an earlier configuration; where the previous (wide)
configuration's numbers are mentioned for contrast they are named as such.

**Start here if you are writing the paper.** Part 12 grades every analysis in
this document as a finding, a supporting result, a control, or something that
should not be presented as a result at all. §12.0 records where each stored
number lives and which few are derived from the stored files rather than read
from them.

---

# Part 1 — What the project is

## 1.1 The task

An observer sees a hand. Two signals report where it is:

- **Vision**, in **retinal** coordinates — where the hand appears on the retina.
  This depends on where the eyes are pointing, so it cannot be compared with
  anything body-centred until it is transformed.
- **Proprioception**, in **body** coordinates — where the arm feels like it is.
  Already in the frame we want.

A third signal reports **eye position**, which is what makes the transformation
possible.

Two things can be true about the world:

- **C = 1** — one object. The seen thing *is* the felt hand. Both signals come
  from a single source, so they should be **fused**.
- **C = 2** — two objects. There is a felt hand and, separately, a visual object
  that is not the hand (the rubber-hand / virtual-hand situation). The signals
  should be kept **segregated**.

The observer is never told which. It must infer it from whether the two signals
agree, *taking into account how trustworthy each one is on this trial* — a small
disagreement between two precise cues is stronger evidence for two causes than a
large disagreement between two vague ones.

## 1.2 What the network is asked to produce

Four numbers per trial, all in **body/spatial coordinates**:

| output | meaning |
|---|---|
| `mu_vis` | estimated position of the **visual source**, and |
| `var_vis` | its uncertainty |
| `mu_prop` | estimated position of the **hand**, and |
| `var_prop` | its uncertainty |

We ask for the visual source *in spatial coordinates* deliberately to force
the reference-frame transformation under **both** causal hypotheses. If the
visual report were allowed to stay retinal, a network could take a shortcut on
C = 2 trials and skip the transformation entirely.

## 1.3 The claim being tested

The network is trained **only** on those four numbers. It never sees `C`, never
sees `p(C=1|x)`, never sees the disparity, never sees the true sources. There is
no output unit for the causal decision.

The claim: if a network trained this way *behaves* as though it is weighting
fusion against segregation by the correct Bayesian posterior, then causal
inference has emerged from the task rather than been supervised into the
network. Training an explicit `p(C)` output would make that claim circular,
which is why the read-out has exactly four units.

---

# Part 2 — The pipeline

Six stages. Each writes files, so any stage can be re-run alone.

```
stage 0   00_calibrate.py     checks the CONFIG is usable      -> results/calibration/<config>/
stage 1   01_generate_data.py samples trials, encodes inputs   -> data/<name>.npz
stage 2   02_train.py         trains the network               -> results/<run>/model.pt
stage 3   03_analyze.py       computes every number            -> results/<run>/metrics.json + analysis.npz
stage 4   04_figures.py       draws every figure               -> results/<run>/figures/  (+ results/manuscript/)
stage 5   05_prior_sweep.py   the cross-prior experiment       -> results/prior_sweep/
```

`run_all.sh` (= `make all`) runs stages 0–4 and also builds the always-fuse
twin. `make quick` does the same with 8,000 trials and 60 epochs. Stage 5 is a
separate experiment (`make sweep`): it trains its own networks, one per (prior,
seed), and depends on `make all` only through the `pcommon1` control that
supplies σ_out (Part 10). Stages 3 and 4 re-run on their own after a change to
a config's `analysis` block (`03_analyze.py --config`, §4.2): the block is not
baked into the network.

**Stage 0 is a gate, not a report.** It exits non-zero on any FAIL and
`run_all.sh` stops there, so a config whose targets are miscalibrated never
reaches training.

---

# Part 3 — The codebase, file by file

```
src/cmsi/
  data/
    generative.py   the generative model + the analytical Bayesian observer
    encoding.py     scalar measurements -> population codes (the network input)
    dataset.py      assembling a dataset; the train/val/test split
  models/
    network.py      the network itself, predict(), hidden_activations()
    losses.py       the objective and the per-output reweighting
    training.py     the training loop with early stopping
  analysis/
    calibration.py  everything stage 0 computes
    accuracy.py     read-out vs observer, per output
    causal.py       the causal-inference analyses (the bulk of stage 3): the
                    implied weight (the hybrid read of each channel) and its
                    binnings by disparity and by posterior, the position- and
                    variance-domain regressions, transition fit, variance
                    signature, model comparison, strategy fit
    units.py        unit-level analyses of the hidden layers
    behavior.py     Körding-style behavioural curves
    decoding.py     what the hidden layers carry
    todo.py         two analyses not yet designed
  experiments/
    implied_weight.py  the weight figures of a run, the weight per reliability level,
                       and `train` (scripts/06_implied_weight.py)
  viz/
    inputs.py       what the network is shown
    training.py     did it converge
    results.py      network vs observer (figures/model), and manuscript_panels()
    prior_sweep.py  the cross-prior figures (stage 5)
    manuscript.py   the standard panel: cell sizes, margins, fonts, lettering
    style.py        PLOS figure defaults, save_figures, exact_frame
  utils/
    config.py  paths.py  seed.py  io.py

scripts/   00_calibrate  01_generate_data  02_train  03_analyze  04_figures
           05_prior_sweep  run_all.sh  (_bootstrap.py puts src/ on the path)
tests/     property tests: generative maths, encoding, models, analyses,
           the design guards, the figure contract, the stage boundaries
```

## 3.1 `data/generative.py` — the world, and the ideal observer

**Sampling one trial** (`_draw_trials`), in this strict order:

```
C      ~ Bernoulli(p_common)                     1 = one source, 2 = two
shared ~ N(mu0, sigma0_sq)
s_vis  = shared              if C = 1,  else an independent draw from N(mu0, sigma0_sq)
s_prop = shared              if C = 1,  else an independent draw from N(mu0, sigma0_sq)
eye    ~ N(eye_mu, eye_sigma_sq)                 same distribution regardless of C
sig2_vis, sig2_prop, sig2_eye ~ Uniform(their ranges)   independent of C and of position
retinal = s_vis - eye                            the sign convention, fixed everywhere
x_vis  ~ N(retinal,  sig2_vis)                   noisy retinal visual measurement
x_eye  ~ N(eye,      sig2_eye)                   noisy eye-position measurement
x_prop ~ N(s_prop,   sig2_prop)                  noisy proprioceptive measurement
```

Two properties matter and are enforced here:

- `p_common` is used **once**, for the Bernoulli draw, and influences nothing
  else. `C` changes **only** whether the two sources are the same number.
- Under C = 2 both sources are drawn from the **identical** prior — same mean,
  same width — as the single source under C = 1. If they differed, the marginal
  statistics of the cues would differ between C values and a network could
  classify C from a side channel instead of from cue agreement.

**Range containment** (`sample_trials(..., contain=(lo, hi))`). A trial whose
`x_vis` falls outside the reliably encoded span is rejected and redrawn through
the identical code path. The bounds come from `containment_bounds(enc)` =
the visual field pulled in by `2 × rf_width`. Redrawing through the same path is
what keeps the rejection rule from becoming a cue to C. The realized rate is
stored per dataset.

**The observer** (`observer`). It sees only `x_vis`, `x_eye`, `x_prop` and the
three variances — never the true sources, never `C`. Four steps:

**(a) Transform vision into the body frame** (`to_body_frame`):

```
var_eye = 1 / (1/sig2_eye + 1/eye_sigma_sq)
k       = var_eye / sig2_eye
eye_hat = k * x_eye + (1 - k) * eye_mu

x_vis_body   = x_vis + eye_hat
var_vis_body = sig2_vis + var_eye
```

The eye measurement is combined with its prior first. That is not a refinement —
it is what the sufficient statistic actually is. Because `x_vis = s − e + noise`
and `x_eye = e + noise` both depend on `e`, the raw sum `x_vis + x_eye` is
unbiased but not efficient.

**(b) The three conditional estimates.** Each includes the prior term:

```
fused (C=1):        precision = 1/var_vis_body + 1/sig2_prop + 1/sigma0_sq
                    fused_mu  = (x_vis_body/var_vis_body + x_prop/sig2_prop + mu0/sigma0_sq) / precision
                    fused_var = 1 / precision

visual alone (C=2): precision = 1/var_vis_body + 1/sigma0_sq
                    seg_vis_mu = (x_vis_body/var_vis_body + mu0/sigma0_sq) / precision
                    seg_vis_var = 1 / precision

hand alone (C=2):   precision = 1/sig2_prop + 1/sigma0_sq
                    seg_prop_mu = (x_prop/sig2_prop + mu0/sigma0_sq) / precision
                    seg_prop_var = 1 / precision
```

**(c) The causal posterior** (`log_bayes_factor` then `common_cause_posterior`),
computed in log space so it cannot underflow at large disparity:

```
sigma_c = sv*sp + sv*s0 + sp*s0          with sv = var_vis_body, sp = sig2_prop, s0 = sigma0_sq
log_c1  = -0.5*log(sigma_c) - 0.5*[ (x_vis_body - x_prop)^2 * s0
                                  + (x_vis_body - mu0)^2 * sp
                                  + (x_prop - mu0)^2 * sv ] / sigma_c

log_c2  = -0.5*log((sv+s0)*(sp+s0)) - 0.5*[ (x_vis_body-mu0)^2/(sv+s0)
                                          + (x_prop-mu0)^2/(sp+s0) ]

log_bf  = log_c1 - log_c2
post_c1 = logistic( log_bf + log(p_common) - log(1 - p_common) )
```

`post_c1` is the trial-wise posterior **p(C=1|x)** while the config key `p_common` is
the **prior**. 

**(d) Model averaging** (`model_average`) — never hard selection:

```
mu  = w*fused_mu + (1-w)*seg_mu
var = w*(fused_var + fused_mu^2) + (1-w)*(seg_var + seg_mu^2) - mu^2
```

with `w = post_c1`. The variance is the **full mixture variance** (law of total
variance), not a weighted average of the two variances. That distinction is the
whole of the §7.6 analysis: the extra term is `w(1−w)(fused_mu − seg_mu)²`,
which is large exactly when the two hypotheses disagree and neither dominates.

## 3.2 `data/encoding.py` — measurements become spike counts

Three populations, concatenated in this order:

| group | code | units in flagship |
|---|---|---|
| `visual_hand` | Gaussian receptive fields over `x_vis`, centres spread evenly across the visual field | 76 |
| `prop_hand` | linear push–pull over `x_prop` (random slopes of mixed sign, random intercepts, rectified) | 76 |
| `prop_eye` | linear push–pull over `x_eye` | 76 |

```
gaussian_code:  r[i,j] = gain[i] * exp( -(x[i] - c[j])^2 / (2 * rf_width^2) )
push_pull_code: r[i,j] = gain[i] * relu( slope[j]*x[i] + intercept[j] )
gain[i] = gain_K / sigma2[i]                     <- reliability enters ONLY here
counts  ~ Poisson(r)
```

Two consequences worth holding onto:

- **Reliability is communicated by spike count, not by a separate input.** A
  precise cue fires more. There is no channel that hands the network σ².
- Because the network's information about a cue is limited by its spike count,
  the *effective* uncertainty can exceed the nominal σ² the targets assume. That
  is what the stage-0 Poisson check measures.

`encode(..., return_clean=True)` also returns the pre-Poisson rates, stored as
`X_clean` in the dataset.

## 3.3 `data/dataset.py` — assembly and the split

`make_dataset(cfg)` runs sample → observe → encode and returns one flat dict of
arrays. Everything the observer computed is kept alongside `X` and `Y`, so any
analysis can mask trials freely (`subset(d, mask)` slices every per-trial array
together, which prevents comparing one trial's prediction to another's truth).

`split_indices(..., stratify=post_c1)` splits **within posterior decile bins**,
so the held-out set covers every decile at full strength. Without this, the
model-comparison bins at intermediate posterior — the bins that discriminate
between averaging and selection — would be the thinnest.

## 3.4 `models/` — network, loss, training

**Architecture** (`network.py`), unchanged from the published network:

```
input (228 units for flagship = 76+76+76)
  -> Linear -> sigmoid      hidden layer 0, "SIL"  (sensory integration), 64 units
  -> Linear -> sigmoid      hidden layer 1, "MSL"  (multisensory),        64 units
  -> Linear                 read-out, 4 units
```

Feedforward only. The two variance columns pass through a softplus so they
cannot go negative. Input standardisation (z-scoring) lives inside the model as
buffers, so a loaded checkpoint is self-contained.

**Loss** (`losses.py`): per-column MSE, each column divided by the variance of
that target in the training set.

```
per_output = mean((pred - target)^2, over trials)
loss       = sum( per_output * 1/var(target) )
```

The reweighting is not cosmetic. The means span ±30 deg and the variances
2–7 deg² on the current flagship; without it the large-scale columns dominate
the gradient and the shared trunk starves `mu_prop`.

**Training** (`training.py`): Adam, lr 0.001, batch 256, up to 300 epochs, early
stopping on validation loss with patience 20. The best-validation weights are
restored at the end. The split is saved in the checkpoint so every later
analysis runs on exactly the trials the model was validated against.

---

# Part 4 — The configs

Fifteen files in `configs/`, plus `configs/experiments/` (the configs `06 train`
used, one pair per experiment). Six are **live** (the flagship, its two
controls, three satellites); one is the **previous flagship**, kept for
reference; one is the `06 train` recipe (`optimal_sweep.yaml`); four are
**legacy** and fail the stage-0 gate; three are configurations the
implied-weight investigation tried (§4.3).

## 4.1 The live set

| config | prior `p_common` | role |
|---|---|---|
| `flagship.yaml` | 0.5 | the main network (since 2026-10-02: the small-domain, narrow-range configuration the implied-weight investigation converged on) |
| `pcommon1.yaml` | 1.0 | always-one-cause control (design document §9.1; here §7.2 and §7.8); train and analyse it **first**, its residuals are σ_out for every other run |
| `pcommon0.yaml` | 0.0 | always-two-causes control (design document §9.2; here §7.8) |
| `pcommon028.yaml` | 0.28 | satellite, matching Körding's human prior fit |
| `pcommon03.yaml` | 0.3 | satellite |
| `pcommon07.yaml` | 0.7 | satellite |

Every one of these is `flagship.yaml` with **only the Bernoulli constant
changed** (the five pcommon files are regenerated from the flagship's body, so
a diff against it shows one line). That is what makes cross-prior comparison
meaningful: same seed, same ranges, same encoders, so any difference in
behaviour is attributable to the prior and nothing else.

`flagship_wide.yaml` is the flagship as it was until 2026-10-02 (σ0² 425, eye²
325, the wide noise ranges). Every stored number in Parts 5–9 and 12 of this
guide was measured on the **new** flagship family, rerun on 2026-10-02/03
(§12.0); the wide configuration's numbers survive only in
`CROSS_PRIOR_RESULT.md` and the `flagship_wide` experiment folder.
`optimal_sweep.yaml` is the training recipe the readability
sweeps converged on (relu 256 × 256, gain_K 720, lr 3e-4, 100k trials) on the
same domain as the new flagship; it is run with `06 train`, not as the
flagship.

## 4.2 What `flagship.yaml` contains

```yaml
seed: 0

generative:
  p_common: 0.5
  mu0: 0.0                        # prior mean on hand position
  sigma0_sq: 100.0                # prior variance -> sd 10 deg
  eye_mu: 0.0
  eye_sigma_sq: 81.0              # eye position sd 9 deg
  sigma2_vis_range:  [1.0, 1.2]   # per-trial visual noise,  sd 1.0-1.1 deg
  sigma2_prop_range: [4.0, 4.5]   # per-trial prop noise,    sd 2.0-2.1 deg
  sigma2_eye_range:  [3.7, 4.2]   # per-trial eye noise,     sd 1.9-2.0 deg

encoding:
  n_vis: 76,  n_prop: 76,  n_eye: 76
  visual_field: [-85.0, 85.0]
  rf_width: 6.0
  gain_K: 180.0                   # gain = gain_K / variance
  slope_range: [-1.0, 1.0]
  intercept_range: [0.0, 10.0]
  poisson_noise: true

model:
  hidden: [64, 64]                # layer 0 = SIL, layer 1 = MSL
  activation: sigmoid
  head: causal                    # 4 outputs

training:
  n_trials: 50000
  split: [0.7, 0.15, 0.15]
  optimizer: adam,  lr: 0.001,  epochs: 300,  batch_size: 256,  patience: 20
  standardize_inputs: true
  balance_loss: true

analysis:
  disparity_grid: [-30,-20,-15,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,15,20,30]
  reliability_levels: [1.0, 1.1, 1.2]
  ridge_alpha: 1.0
  decoder_test_size: 0.25
```

(`min_separation` and `sigma_w_criterion`, the guards of the per-trial ratio,
were removed from every config on 2026-09-29 with the ratio itself; §7.2.)

**The `analysis` block is not a training parameter.** It sets the bin centres
of the disparity curves (figures 03/04/05/12, manuscript Fig. 2A/C, the
conditioned-bias flag), the levels of figure 05 and the decoders of §7.8, and
nothing the network sees. Stage 3 and 4 therefore take it from the config file
given with `--config`, else from `results/<run>/config.yaml` as stage 3 last
wrote it, else from the checkpoint (`utils.analysis_block`), and stage 3
records the block it used in `metrics.json` (`analysis_config`,
`analysis_config_source`) and writes it back into the run's `config.yaml`. A
change to the block is applied to an existing run with one stage-3 and one
stage-4 call, no retraining (Part 10).

**The grid changed on 2026-10-03.** Until then it was the wide configuration's
`[-40, -30, -20, -15, -10, -5, -2, 0, 2, 5, 10, 15, 20, 30, 40]`. On the 10°
domain 81 % of the flagship's trials have |d| < 12.5° and the whole
fusion-to-segregation transition happens there (the weight falls from 0.82 to
0.07 between the 0, ±2, ±5 and ±10 centres), while 99 % of trials lie below
34°: the old grid spent 8 of its 15 centres on the flat tail and its ±40 bins
rested on 30 and 33 trials (16 at +40 on `pcommon07`). The new grid steps by
2° across the transition, ends at ±30, and every bin holds at least 94 trials
on every causal run (147 and 162 at ±30 on the flagship); the few trials
beyond 35° fold into the ±30 bins. It changes no stored statistic except the
binned bias curves and the flag of §7.10, and the four figures.

### Why these numbers and not the previous ones

The encoding, the network and the training are those of `flagship_wide.yaml`,
which were derived from `matlab_match.yaml` by running the stage-0 gate and
fixing what failed (sensory σ² × 0.30, `gain_K` 50 → 180, `rf_width` 10 → 6,
`n_vis` 50 → 76, `visual_field` ±75 → ±85; the reasons are in that file's
header). The generative block changed on 2026-10-02, after the implied-weight
investigation (`IMPLIED_WEIGHT_RESULT.md`):

| parameter | was (`flagship_wide`) | is | why |
|---|---|---|---|
| `sigma0_sq` | 425 (hand sd 20.6°) | **100** (sd 10°) | the input domain relative to the sensory noise is what decides whether the network learns the optimal weighting: on the wide domain the position-regression slope stalled at 0.86 whatever the network size; on this one it is 0.95–0.99 with the same 64 × 64 network (Part I, §4 of the result file) |
| `eye_sigma_sq` | 325 (eye sd 18°) | **81** (sd 9°) | the same lever for the retinal domain the visual population has to cover |
| noise ranges | [1.2, 6.6] / [1.8, 9.6] / [3.3, 13.2] | **[1, 1.2] / [4, 4.5] / [3.7, 4.2]** | narrow ranges make the reliability-dependent weighting easy to learn (slope +0.03 against the wide ranges); the cost is that reliability barely varies, so figure 05's levels differ by little and `06 reliability --levels 5` is the way to see what variation there is |
| `reliability_levels` | [1.5, 3.5, 6.0] | **[1.0, 1.1, 1.2]** | the levels must lie inside `sigma2_vis_range` |

What the configuration gives, measured on the `flagship` run of 2026-10-02
(hybrid read, visual channel; the run reproduces the numbers the config
header records from the `exp2_64_mean` run that trained it before the rename,
to the last digit — same seed, same code): weight on the
posterior slope 0.977 [0.967, 0.988], position regression 0.952 (hand 0.939),
variance regression 0.977, 0.1 % of per-trial weights outside [−1, 2],
transition midpoint 4.89° against 5.08° analytical, reliability-within-disparity
slope 0.89 ± 0.02, σ_out 0.39° on its control. Part 7 has every one of these
with its context.

The trade-off to know: the stage-0 gate **WARNs** on this configuration and
does not stop — 49 % of trials sit at intermediate posterior (the design
document's §8.1 asks for
≈25 %) because two independent sources drawn from a 10° prior are often as
close as one shared source, and the posterior at zero disparity is capped near
0.8. The hand positions reach only ±30°, so the ±85° visual field rejects 0.0 %
of trials.

## 4.3 The legacy configs

`default.yaml`, `realistic.yaml`, `matlab_match.yaml`, `equal_n.yaml` are kept
for reference and comparison. All four **fail** the stage-0 Poisson check, and
three of them also warn on the posterior histogram. They are not deleted because
they document where the project came from, but they should not be used for new
runs. `exp2.yaml`, `exp_weight.yaml` and `lownoise_both.yaml` are configurations
the investigation tried on the way to the current flagship (their runs are
gone; `configs/experiments/` holds the exact configs `06 train` used for the
others).

---

# Part 5 — Short report about `results/`

All six live runs were trained and analysed on 2026-10-02/03 from the
configs in §4.1 (`pcommon1` first, so that its residuals were on disk as σ_out
when the other five were analysed; `sigma_out_source` is `pcommon1` in every
other `metrics.json`).

| run | prior | dataset | test trials | best val loss @ epoch |
|---|---|---|---|---|
| `pcommon1` | 1.0 | pcommon1 | 7501 | 0.1513 @ 116 |
| `flagship` | 0.5 | flagship | 7500 | 0.0784 @ 182 |
| `pcommon0` | 0.0 | pcommon0 | 7500 | 0.1650 @ 198 |
| `pcommon028` | 0.28 | pcommon028 | 7500 | 0.1104 @ 150 |
| `pcommon03` | 0.3 | pcommon03 | 7500 | 0.0993 @ 253 |
| `pcommon07` | 0.7 | pcommon07 | 7500 | 0.0783 @ 281 |

Plus an always-fuse **twin** for each (`*_twin`), the six live calibration
folders under `results/calibration/` (the older `exp2*`, `exp_weight`,
`lownoise*` folders there are from the investigation and can be ignored),
`results/prior_sweep/` (27 networks, §8.2), `results/manuscript/` and
`results/experiments/implied_weight/flagship/` (the `06` side experiment, Part 10).

Two things to notice in that table, both the reverse of what the wide
configuration showed. The two controls now reach a **higher** loss than the
flagship (0.151 and 0.165 against 0.078). The loss is the per-output MSE
divided by the target variance (`balance_loss`), i.e. roughly the sum of
1 − R² over the four outputs, and the `var` targets of a single-cause dataset
are nearly constant — there is no causal ambiguity to spread them — so the same
absolute error on them is a large fraction of their variance (`var_vis` R² is
0.924 on `pcommon1` against 0.966 on the flagship; the position outputs reach
R² 0.998, better than any causal run). Nothing about the fit is worse. And the
causal runs order by prior in a U: 0.110 at 0.28, 0.099 at 0.3, 0.078 at 0.5
and 0.7 — the sweep's per-network `best_val` (`rows` in `sweep.json`, §8.2)
shows the same U with its
minimum near 0.5–0.6. Best epochs lie between 116 and 281 of the 300 available.

---

# Part 6 — Stage 0, the calibration gate

Stage 0 draws a fresh dataset from the config (20,000 trials by default) and
runs eight checks. It writes `results/calibration/<config>/metrics.json` and
four figures.

## 6.1 The eight checks, and what each computes

(The § numbers in the check headings below are the design document's, not
this guide's.)

### Check 1 — realized C = 1 fraction (§3)

Counts `mean(C == 1)` and compares with the configured prior. Catches a
generator whose Bernoulli draw has drifted from the constant the targets use.

**Passes if** |realized − prior| < 0.02.

### Check 2 — posterior histogram (§8.1)

The distribution of `post_c1` across trials. Two quantities:

```
intermediate = fraction of trials with 0.2 < post_c1 < 0.8
confident    = fraction with post_c1 < 0.05 or > 0.95
```

**Why it matters.** If almost every trial is confidently 0 or 1, there is no
ambiguous zone and the causal-inference analyses have nothing to bite on. If
almost every trial is intermediate, the network never sees clean fusion or clean
segregation. The design asks for ≈25% intermediate with substantial confident
mass.

**Passes if** 0.15 ≤ intermediate ≤ 0.40 **and** confident ≥ 0.15.

### Check 3 — reliability coverage (§8.3)

Bins trials by |disparity| into 8 quantile bins; within each bin, records the
spread of `post_c1`. Reports both the standard deviation and the peak-to-peak
range.

**Why it matters.** The Bayes-vs-heuristic test (§7.2) works by holding
disparity fixed and asking whether the network still tracks the posterior as
reliability varies. If the posterior barely moves within a disparity bin, that
test has no leverage.

**Passes if** the mean within-bin **range** > 0.15. (The range, not the std, is
what the test leverages — the regression's leverage comes from how far apart the
extreme values in a bin are. A std-based floor would reject usable designs.)

### Check 4/5 — |Δ| magnitude (§8.2)

`Δ` is the gap between the two hypotheses for a given output:

```
Delta_vis  = fused_mu - seg_vis_mu
Delta_prop = fused_mu - seg_prop_mu
```

Records the median |Δ| and the fraction of trials with |Δ| > 2 deg.

**Why it matters.** The implied weight is recovered by asking where the network's
answer sits between the two hypotheses. When they nearly coincide, the answer
carries no information about the weight, no matter how good the network is.

**Passes if** at least 25% of trials have |Δ| > 2 deg.

### Check 6 — anti-confound audit (§9.4)

For each of `|x_eye|`, `|x_vis|`, `|x_prop|`, `sig2_vis`, `sig2_prop`,
`sig2_eye`, computes the rank AUC for predicting C = 1 from that variable alone.

**Why it matters.** If C leaked into any single channel, the network could
classify the causal structure from a side channel instead of from cue agreement,
and the whole claim collapses. Chance is AUC = 0.5.

**Passes if** the worst deviation from 0.5 is < 0.05.

### Check 7 — range containment (§3)

The realized rejection rate from the containment resampling.

**Passes if** < 3%; warns below 10%; fails above.

### Check 8 — Poisson validity (§4)

The one that failed on every legacy config. Simulates unimodal visual trials at
a fixed position, encodes them with the configured gain, draws Poisson counts,
then maximum-likelihood decodes the position back out of the spike counts:

```
loglik(x) = counts · log(gain * F(x))  -  sum(gain * F(x))
x_hat     = argmax over a fine grid
inflation = Var(x_hat - x) / sigma2
```

**Why it matters.** The targets assume the network's uncertainty about the
visual cue is exactly σ². If the spike code carries less information than that,
the network's true uncertainty is larger, and because gain = K/σ² the shortfall
is **multiplicative and uniform across the whole reliability range** — including
the ambiguous regime where causal inference lives. Every target would be
miscalibrated in the same proportion everywhere.

The statistic is the **paired** ratio `Var(x̂ − x)/σ²`. The unpaired form
`Var(x̂)/σ² − 1` estimates the same thing but carries the sampling noise of an
unpaired variance (≈ ±1.8% at n = 6000), which overstates the problem.

**Passes if** < 5%; warns below 20%; fails above.

## 6.2 Your gate results

Read from `results/calibration/*/metrics.json`, 20,000 trials each, gate run
on 2026-10-02.

| | flagship | pcommon028 | pcommon03 | pcommon07 | pcommon1 | pcommon0 |
|---|---|---|---|---|---|---|
| prior | 0.5 | 0.28 | 0.3 | 0.7 | 1.0 | 0.0 |
| realized C=1 | 0.4966 | 0.2735 | 0.2924 | 0.6980 | 1.0000 | 0.0000 |
| **intermediate posterior mass** | **0.490** ⚠ | **0.468** ⚠ | **0.483** ⚠ | 0.167 | 0.000 | 0.000 |
| confident posterior mass | 0.285 | 0.422 | 0.409 | 0.234 | 1.000 | 1.000 |
| coverage (mean range) | 0.369 | 0.427 | 0.417 | 0.274 | 0.000 | 0.000 |
| median \|Δ\| visual | 1.98 | 3.14 | 3.00 | 1.46 | 1.06 | 4.96 |
| median \|Δ\| hand | 1.73 | 2.76 | 2.64 | 1.27 | 0.92 | 4.33 |
| frac \|Δ_vis\| > 2 | 0.497 | 0.629 | 0.618 | 0.379 | 0.199 | 0.788 |
| frac \|Δ_prop\| > 2 | 0.453 | 0.592 | 0.580 | 0.329 | 0.143 | 0.759 |
| rejection rate | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Poisson inflation | 3.2% | 3.2% | 3.2% | 3.2% | 3.2% | 3.2% |
| worst anti-confound AUC dev | 0.0097 | 0.0191 | 0.0167 | 0.0049 | n/a | n/a |
| **verdict** | 7 PASS **1 WARN** | 7 PASS **1 WARN** | 7 PASS **1 WARN** | **8 PASS** | 2 PASS | 2 PASS |

Notes on reading this table:

- **The flagship and the two low-prior satellites carry one WARN each**, on
  check 2: 47–49 % of their trials land at intermediate posteriors, above the
  40 % ceiling (the wide configuration had 25 % on the flagship and warned on
  `pcommon07` instead, for the opposite reason). This is a property of the
  10° prior, not a bug: two independent sources drawn from it are often as
  close as one shared source, and the posterior at zero disparity is capped at
  about 0.8 at `p_common` = 0.5 (0.64 at 0.28), so even perfectly agreeing cues
  do not reach "confident". The analyses lose nothing by it — if anything the
  ambiguous zone they work in is larger — and the gate continues on a WARN;
  it stops only on a FAIL, which the anti-confound check (deviation ≥ 0.05),
  containment (rejection ≥ 10 %) and Poisson validity (inflation ≥ 20 %) can
  raise. `pcommon07` passes
  because the higher prior pushes mass toward confident fusion (16.7 %
  intermediate).
- **The two controls run only 2 checks.** With the prior at 0 or 1 the posterior
  is constant, so the realized-fraction, posterior-histogram, coverage, Δ and
  anti-confound checks have nothing to test and are skipped (the statistics are
  still written, which is why the table has their realized fraction and |Δ|).
  Containment and Poisson still run.
- **The Poisson inflation is identical (3.2%) across all six.** It should be —
  the encoding parameters are the same in every config; only the Bernoulli
  constant differs.
- **Nothing is rejected.** Hand positions from a 10° prior reach about ±30°,
  and the visual field is ±85°, so the containment resampling never fires.
- **Median |Δ| rises as the prior falls** (4.96 → 3.14 / 3.00 → 1.98 → 1.46 →
  1.06 as the prior goes 0 → 0.28 / 0.3 → 0.5 → 0.7 → 1.0). At a low prior the
  observer segregates more often, so the segregated and fused solutions sit
  further apart. All these gaps are about half the wide configuration's (0.4–0.6
  of them) — the domain is smaller — which is why 31–52 % of the trials in the hybrid read
  of §7.2 come from the variance root, where a position ratio would be blind.

## 6.3 The four calibration figures

Written to `results/calibration/<config>/figures/`.

**`01_posterior_histogram.png`** — histogram of `post_c1` over the drawn trials,
with the 0.2–0.8 band shaded, titled with the intermediate fraction. This is the
picture of check 2.

**`02_joint_w_delta.png`** — two 2-D histograms (`hist2d`, 50×50 bins) of
`post_c1` against |Δ|, one panel for the visual output and one for the hand.
This is the joint distribution the design asks to be recorded: it shows whether
there are trials that are *both* ambiguous *and* readable. A design with plenty
of intermediate posteriors but all at tiny |Δ| would look fine on check 2 and
still leave the weight-recovery analysis blind.

**`03_reliability_coverage.png`** — bar chart of the within-|disparity|-bin
spread of `post_c1`, plotting both the range (light) and the std (dark), with
the 0.15 power floor marked. This is check 3.

**`04_poisson_validity.png`** — two bars, one per end of the σ range, showing
percent inflation with the 5% tolerance line. This is check 8.

---

# Part 7 — The analyses in stage 3

Stage 3 loads the checkpoint, restricts to the stored **test split**, runs
everything below, and writes `metrics.json` (numbers) plus `analysis.npz`
(per-trial arrays for the figures), and the `config.yaml` whose analysis
block it used (§4.2).

## 7.1 Read-out accuracy

For each of the four outputs: slope and intercept of a least-squares fit of
network against analytical, R², RMSE, MAE, and mean bias.

**Your results.**

| run | mu_vis R² | var_vis R² | mu_prop R² | var_prop R² | mu_vis RMSE (deg) |
|---|---|---|---|---|---|
| flagship | 0.9961 | 0.9657 | 0.9965 | 0.9646 | 0.61 |
| pcommon028 | 0.9951 | 0.9460 | 0.9948 | 0.9439 | 0.68 |
| pcommon03 | 0.9951 | 0.9589 | 0.9944 | 0.9546 | 0.68 |
| pcommon07 | 0.9971 | 0.9654 | 0.9974 | 0.9666 | 0.54 |
| pcommon1 | 0.9985 | 0.9240 | 0.9985 | 0.9243 | 0.39 |
| pcommon0 | 0.9978 | 0.9149 | 0.9979 | 0.9372 | 0.46 |

Slopes are 0.91–1.00 and biases are within ±0.09 deg everywhere.

**How to read it.** The mean channels are near-perfect in every run (R² ≥
0.994, RMSE 0.4–0.7° on positions that span ±30°). The variance channels are
the harder ones everywhere, 0.94–0.97 on the four causal runs and — unlike on
the wide configuration — *lower* on the two controls (0.91–0.94). That is not a
worse fit: with the prior pinned there is no mixture term, the variance target
is a smooth function of the reliabilities alone and, with the narrow noise
ranges, spans only a few hundredths of a deg², so the same read-out noise
(RMSE 0.014 deg² on `pcommon1` against 0.22 on the flagship) is a larger
fraction of a much smaller variance. R² on a nearly constant target is the
wrong yardstick; the RMSE column is the one to compare.

## 7.2 σ_out, and how the implied weight is read

```
residual_std = std(network - analytical), per output        (sigma_out on a control)
```

`sigma_out` is the read-out noise per output. It is meaningful only when
measured on the **p_common = 1 control**, where the target is single-valued so
residuals are pure read-out noise; on the flagship the residuals also contain
any causal-inference misweighting. Stage 3 takes it from `--control pcommon1`
when given (`run_all.sh` passes it automatically once
`results/pcommon1/metrics.json` exists) — your flagship, pcommon0, pcommon028,
pcommon03 and pcommon07 runs all record `sigma_out_source: "pcommon1"`, the
control itself records `"self"` — and writes the value itself, `sigma_out`
(one entry per output), next to that key. A run's own `residual_std` is also
stored, and on the flagship it is 0.60 rather than 0.39 for `mu_vis` and 0.22
rather than 0.014 for `var_vis`, because it contains the causal misweighting
as well as read-out noise.

**Your σ_out** (from `results/pcommon1/metrics.json`):

```
mu_vis 0.3882    var_vis 0.0137    mu_prop 0.3883    var_prop 0.0137  (deg, deg^2)
```

The two mean channels agreeing to three decimals is a good sign — under C = 1
they are the same quantity, and the network treats them as such. The control
also checks that its targets collapse the way they should
(`pcommon1_target_reduction_maxdiff` 1.5 × 10⁻⁸: with the prior at 1 the
mixture target equals the fused one on every trial).

**The implied weight: the hybrid read** (`analysis.hybrid_weight`). The
network never outputs a weight; on every trial it is recovered from the two
outputs of one channel. A model-averaging observer's variance is the mixture
variance

```
v(w) = w * var_fus + (1 - w) * var_seg + w (1 - w) * Delta^2        (Eq. 10)
```

a parabola in `w` opening downward, with `v(0) = var_seg`, `v(1) = var_fus`
and its peak at `w* = 1/2 - c / (2 Delta^2)`, where `c = var_seg - var_fus` is
the variance fusion saves (2.42 deg² on the flagship's median trial for the
visual channel — segregated 4.64, fused 2.22 — and 1.86 for the hand; the
narrow noise ranges keep it within 2.3–2.6 on 80 % of trials). Two regimes
follow from where that peak sits:

- **`Delta^2 <= c`** (small disparity, including zero): the peak is at or left
  of `w = 0`, the parabola only falls on [0, 1], and the channel's variance
  output has exactly one root — that root is the weight. This is where a
  position ratio `(network - seg)/Delta` is blind (it divides by ~0), and the
  variance read is at its sharpest: its sensitivity `|dv/dw| = |Delta^2 (1 - 2w)
  - c|` equals `c` at `Delta = 0`.
- **`Delta^2 > c`** (large disparity): the parabola rises before it falls, the
  variance output would have two roots, and the weight is instead the
  channel's own position ratio, which is well conditioned exactly here
  (`|Delta| > sqrt(c)`, 1.56 deg on the flagship's visual channel, 1.36 on the
  hand).

Every trial gets a weight; nothing is filtered or clipped. Stage 3 stores the
visual and the proprioceptive read in `analysis.npz` as `fusion_weight` and
`fusion_weight_prop`, with a flag per trial saying which output it came from
(`hybrid_flags_vis/prop`: 0 variance root, 1 position ratio, 2 the parabola's
peak, returned when the variance output exceeds any mixture variance) and the
nominal per-trial sd (`sigma_w_vis/prop`: `sigma_out(var) / |dv/dw|` on root
trials, `sigma_out(mu) / |Delta|` on ratio trials). `metrics.json` carries
`hybrid_read` (the shares of trials per source, the fraction outside [-1, 2],
the sd against the posterior overall and for each part), `weight_regression_
vis/prop` (the weight regressed on the posterior) and `implied_weight_vs_post
[_prop]` (slope, R², RMSE, every trial). Every analysis that needs a weight —
the consistency test (§7.4), the reliability test (§7.5), the transition fit
(§7.11), the behavioural conditioning (§7.10), figures 04, 05, 15 and 16, the
prior sweep — reads this one.

**Your results** (visual channel; the proprioceptive one in `metrics.json`):

| run | prior | trials from the ratio | outside [-1, 2] | sd vs posterior (root / ratio trials) | corr | slope on the posterior [CI] | R² |
|---|---|---|---|---|---|---|---|
| pcommon028 | 0.28 | 69 % | 0.05 % | 0.156 (0.078 / 0.181) | 0.87 | 0.969 [0.957, 0.982] | 0.75 |
| pcommon03 | 0.3 | 68 % | 0.03 % | 0.170 (0.069 / 0.201) | 0.86 | 0.965 [0.951, 0.978] | 0.73 |
| flagship | 0.5 | 58 % | 0.08 % | 0.160 (0.044 / 0.208) | 0.91 | 0.977 [0.967, 0.988] | 0.82 |
| pcommon07 | 0.7 | 48 % | 0.00 % | 0.134 (0.031 / 0.191) | 0.93 | 0.973 [0.964, 0.982] | 0.86 |

The hand channel reads the same way (ratio share 48–69 %, sd 0.127–0.175,
slopes 0.937 [0.923, 0.951], 0.967, 0.953 and 0.998 [0.990, 1.007] in the
same order).

**How to read it.** The share of trials read from the ratio is set by the
prior: a higher prior means more common-cause trials, smaller disparities and
more trials with `Delta^2 <= c`. The root trials are the precise part of the
read (sd 0.03–0.08 against the posterior), the ratio trials the noisier part
(their error is `sigma_out / |Delta|`, largest just above `sqrt(c)`), and the
sd against the posterior mixes both with the network's own deviation from
optimality. The slope on the posterior says the same as the position
regression (§7.3): every network pulls toward the fused solution very slightly
less than the ideal observer — 2–6 % short, with the CI excluding 1 on every
run but the hand read of `pcommon07`. The intercepts lie within ±0.02. These are
the numbers that were 0.84–0.92 on the wide configuration; the move to the
10° domain is what closed the gap (`IMPLIED_WEIGHT_RESULT.md`, Part I).

## 7.3 The position-domain regression — the headline

(The headline *statistic*; in the manuscript's Figure 2 its scatter was
replaced on 2026-10-03 by the weight on the posterior of §7.2, panel B of
figure 08v, which shows the same slope-below-one from the per-trial read with
the two sources of the read coloured. Both regressions stay in
`metrics.json` and figure 08 stays in `figures/model/`; §9.4.)

The problem with computing a per-trial weight is the division. This analysis
avoids it entirely:

```
regress   (network_estimate - seg)   on   post_c1 * (fused - seg)
```

Both sides are in degrees. A Bayes-optimal model-averaging observer gives
**slope 1, intercept 0**. Every trial enters with its natural leverage: trials
where Δ ≈ 0 sit near the origin and pull the fit almost not at all, which is
exactly right, because those trials genuinely carry no information about the
weight.

Reported with standard errors and 95% CIs, per output.

**Your results.**

| run | prior | visual slope [CI] | visual intercept | R² | hand slope [CI] | hand intercept |
|---|---|---|---|---|---|---|
| pcommon028 | 0.28 | 0.862 [0.832, 0.891] | +0.083 | 0.30 | 0.912 [0.877, 0.947] | −0.030 |
| pcommon03 | 0.3 | 0.955 [0.927, 0.984] | −0.064 | 0.37 | 0.923 [0.889, 0.957] | −0.083 |
| flagship | 0.5 | 0.952 [0.936, 0.969] | −0.044 | 0.63 | 0.939 [0.921, 0.957] | +0.066 |
| pcommon07 | 0.7 | 0.970 [0.959, 0.981] | −0.012 | 0.80 | 0.976 [0.964, 0.988] | +0.001 |

**How to read it.** Intercepts are essentially zero everywhere (|intercept| ≤
0.09 deg, against outputs that range over tens of degrees). Slopes sit at
0.86–0.98 — below 1, and the CIs exclude 1 in every case. The network
consistently applies *slightly less* pull toward the fused solution than the
ideal observer would, and the shortfall shrinks as the prior rises (the sweep
in §8.2 shows the same trend over nine priors).

The R² column is low because of what the regression is, not because the fit is
poor: at `p_common` = 0.28 most trials are confidently segregated, so both
sides of the regression sit near zero and the spread of the predictor is
small; the slope's standard error (0.015) is what measures the fit. This is a
real, reproducible finding rather than a pipeline artefact: every guard has
been verified, both controls are clean, and the deviation appears at four
different priors and across the sweep. It says the network's implicit causal
inference is close to Bayesian but systematically a little conservative — on
this configuration by 3–5 % on the flagship and above, by 14 % at the lowest
prior.

The companion **variance-domain regression** (`variance_regression_vis/prop`,
figure 08v panel A) regresses the network's variance output minus the
segregated variance on the analytical mixture's excess over it, the same
construction one output over. Its slopes are 0.956 / 0.957 / 0.977 / 0.971
(visual, in the table's order) and 0.959 / 0.956 / 0.981 / 0.968 (hand), all
with standard errors of 0.002, intercepts within ±0.04 deg² and R² 0.94–0.97:
the variance channel tracks its target more tightly than the position channel
does, and shows the same slight shortfall.

## 7.4 Hand-vs-visual consistency

Both channels are mixtures with the **same** `w`, so the visual and the
proprioceptive read of the weight must agree. `weight_consistency` compares
the two hybrid reads on every trial: the correlation and the mean absolute
difference.

**Your results.**

| run | n | correlation | mean abs difference |
|---|---|---|---|
| pcommon028 | 7500 | 0.740 | 0.127 |
| pcommon03 | 7500 | 0.683 | 0.101 |
| flagship | 7500 | 0.851 | 0.108 |
| pcommon07 | 7500 | 0.848 | 0.105 |

**How to read it.** The two reads of a quantity bounded in [0, 1] agree to
about 0.1 on average, on every trial of every run, with correlations of
0.68–0.85. The residual disagreement is read-out noise, not incoherence: each
channel's ratio trials carry an error of `sigma_out / |Delta|` that is
independent between the channels, and the hand channel's `Delta_prop` is the
smaller of the two (median 1.7 deg against 2.0 deg for vision on the
flagship), so its ratio trials are the noisier ones. The two low-prior
satellites have the lowest correlations because most of their trials are
confidently segregated — both reads sit near 0 and the correlation has little
variance to work with (the MAD, 0.10–0.13, is as good as the flagship's). On
the wide configuration these numbers were 0.35–0.47 and 0.25–0.32; the narrow
domain, where σ_out is 0.39° instead of 0.63° and the trials with `Delta^2 <=
c` are read from the variance root, is what brought them here. The visual
channel's read is the one to report; the proprioceptive one is the check.

## 7.5 Bayes versus a disparity heuristic

The concern: a network could produce a plausible-looking weight curve just by
responding to disparity, without doing anything Bayesian. The two are separable
because at **matched disparity** the optimal weight still varies with the cue
reliabilities.

The test bins |disparity| into 8 quantile bins and, within each, regresses
`w_implied` on `post_c1`.

- A pure disparity heuristic predicts **slope 0** in every bin — within a bin the
  disparity is fixed, so a disparity-driven network has no reason to vary.
- Bayes predicts **slope 1**.

Bins where `post_c1` barely varies are **dropped**, not merely downweighted, and
listed under `skipped`. With almost no variation in the predictor the slope is
0/0-ish and comes back in the tens with an equally large standard error.

**Your results** (the visual hybrid read, every trial).

| run | prior | combined slope | 95% CI | bins kept | bins skipped |
|---|---|---|---|---|---|
| pcommon028 | 0.28 | **0.906 ± 0.016** | [0.875, 0.938] | 6 | 2 |
| pcommon03 | 0.3 | **0.978 ± 0.012** | [0.954, 1.002] | 6 | 2 |
| flagship | 0.5 | **0.887 ± 0.018** | [0.852, 0.922] | 7 | 1 |
| pcommon07 | 0.7 | **0.894 ± 0.016** | [0.864, 0.925] | 7 | 1 |

The flagship's per-bin detail:

| \|disparity\| bin | n | spread of post_c1 | slope ± SE |
|---|---|---|---|
| 0.0 – 0.8 | 938 | 0.234 | +0.967 ± 0.024 |
| 0.8 – 1.6 | 937 | 0.251 | +0.598 ± 0.103 |
| 1.6 – 2.6 | 938 | 0.292 | +0.764 ± 0.065 |
| 2.6 – 3.8 | 937 | 0.385 | +0.727 ± 0.085 |
| 3.8 – 5.8 | 937 | 0.608 | +0.653 ± 0.063 |
| 5.8 – 9.8 | 938 | 0.813 | +0.879 ± 0.039 |
| 9.8 – 16.7 | 937 | 0.183 | +1.476 ± 0.372 |

and the one skipped bin (|d| 16.7–53, spread of `post_c1` 0.000): past about
17 deg the posterior is numerically pinned at zero for every trial regardless
of reliability, so there is nothing left to test. (The satellites at 0.28 and
0.3 skip two bins, from 13 deg on; `pcommon07` one, from 12 deg.)

**How to read it.** Every run is far from 0, so the disparity-heuristic
explanation is ruled out — the network *is* using reliability — and every
combined slope sits at 0.89–0.98 with an interval of ±0.03. Because the
hybrid read exists on every trial, the test covers the whole disparity range
rather than the few far bins a filtered ratio would leave it, and the per-bin
slopes say where reliability is tracked less than fully: at 1–6 deg the
flagship's slopes are 0.60–0.76, in the first bin (|d| < 0.8, read almost
entirely from the variance root) 0.97, and in the 6–10 deg bin, which has the
most reliability-driven variation and therefore the most weight in the
inverse-variance-weighted combination, 0.88. The last kept bin (10–17 deg,
spread 0.18) is where the network is already near `w = 0` and the ratio read
is noisiest; its slope of 1.5 ± 0.4 is consistent with 1 and carries little
weight. On the wide configuration the same table had slopes of 0.50–0.61 on
the flagship and pcommon028, with a negative bin in the transition region;
nothing of that remains.

## 7.6 The variance signature of causal ambiguity

This analysis interrogates a different output from every other section in §7.
Everything above reads the **mean** channels (`mu_vis`, `mu_prop`). This one
reads the **variance** channels, and it discriminates where the mean channels
are weakest: for a mean estimate, model averaging and a well-tuned fixed weight
produce similar numbers; for a variance estimate they are qualitatively
different, and the figure shows why.

### The equation the figure is built on

If you average two hypotheses with weight `w`, the variance of the mixture is
**not** the weighted average of the two variances. By the law of total variance:

```
Var  =  w*V_fused + (1-w)*V_seg      <- WITHIN:  how noisy each hypothesis is
      + w*(1-w)*d^2                   <- BETWEEN: how far apart they are,
                                                  times how unsure you are
```

with `d = fused_mu - seg_mu`.

The second term is the diagnostic one. It is zero at `w = 0` and zero at
`w = 1`. It is nonzero only when you do not know which hypothesis to believe.
It is uncertainty **about the causal structure**, as distinct from measurement
noise. **No fixed-weight model can produce it**, because fixing `w` deletes the
`w(1-w)` factor.

### The four lines

The analysis bins trials by `post_c1` into 10 bins and plots, per bin:

| line | what it is |
|---|---|
| **black dashed** | the full right-hand side above — the correct answer |
| **blue** | the network's mean Var output |
| **black dotted** | the `w(1-w)d^2` term **alone** |
| **green** | control: a *fixed* weight, `w̄·V_fused + (1−w̄)·V_seg` with `w̄` = the mean posterior (any constant gives a flat line; §7.7's least-squares best constant, 0.045, would sit at 4.5 deg² instead of 3.4) |

The x-axis is the analytical posterior, **not disparity**. Far left: certain
there were two causes. Far right: certain there was one. Middle: genuinely
unsure which.

The scalar summary is the **hump**: mean variance in the intermediate bins
(0.2 < centre < 0.8) minus mean variance in the confident bins.

### Your flagship, `var_vis`, per bin (from `metrics.json`)

| posterior bin | network | analytical mixture | between term | fixed-weight | n |
|---|---|---|---|---|---|
| 0.05 | 4.85 | 4.81 | 0.19 | 3.43 | 2141 |
| 0.15 | 6.16 | 6.25 | 1.97 | 3.44 | 227 |
| 0.25 | 6.23 | 6.37 | 2.34 | 3.42 | 179 |
| 0.35 | 5.93 | 6.05 | 2.26 | 3.44 | 190 |
| 0.45 | 5.49 | 5.55 | 2.00 | 3.44 | 243 |
| 0.55 | 4.76 | 4.79 | 1.49 | 3.44 | 345 |
| 0.65 | 3.94 | 3.92 | 0.86 | 3.44 | 651 |
| 0.75 | 3.05 | 3.02 | 0.22 | 3.44 | 2034 |
| 0.85 | 2.77 | 2.73 | 0.13 | 3.43 | 1079 |
| 0.95 | 2.48 | 2.44 | 0.07 | 3.44 | 411 |

(The fixed line uses `w̄` = the mean posterior, 0.50 here, so it sits at
0.50 × 2.22 + 0.50 × 4.64 = 3.43 deg².)

### Reading it: two effects are stacked

Splitting the mixture into its two brackets (`within = mixture − between`,
derived from the table above) is what makes the blue curve legible:

| posterior | 0.05 | 0.15 | 0.25 | 0.35 | 0.45 | 0.55 | 0.65 | 0.75 | 0.85 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|
| within | 4.62 | 4.28 | 4.03 | 3.79 | 3.55 | 3.30 | 3.05 | 2.80 | 2.60 | 2.37 |
| between | 0.19 | 1.97 | **2.34** | 2.26 | 2.00 | 1.49 | 0.86 | 0.22 | 0.13 | 0.07 |
| between as % of total | 4% | 31% | **37%** | 37% | 36% | 31% | 22% | 7% | 5% | 3% |

**The ramp.** `within` falls monotonically 4.62 → 2.37. Segregation is
expensive: if the cues are separate, the visual estimate of the hand is the
visual measurement carried through the noisy eye position (variance 4.64),
while fusion buys 2.22. The curve therefore slopes downhill left-to-right even
with zero causal uncertainty.

**The hump.** The dotted line rides on top of that ramp, peaking at 2.34.

Blue is the sum of the two, which is why it is high on the left, humped in the
middle, and low on the right.

**At the peak, more than a third of the network's reported uncertainty is
causal uncertainty** — 2.34 of 6.37 at bin 0.25 (the network reports 6.23
there). That is the headline of the figure. On the wide configuration the
share was half; here the hypotheses are closer together (median |Δ| 2° against
4°), so the between term is a smaller part of a smaller total.

### Three features that look wrong and are not

**The peak sits at 0.25, not 0.5.** Two factors multiply. `w(1-w)` peaks at
0.5, but `d^2` is falling as `w` rises — mean |disparity| across the bins runs
17.3° at the left edge down to about 1.5° at the right. Rising times falling
peaks left of centre.

**The leftmost point is not part of the hump.** Bin 0.05 has the largest
disparity of any bin (mean 17.3°) — the hypotheses are further apart there than
anywhere else — yet its between term is the *smallest* (0.19). The reason is
that mean `w(1-w)` in that bin is 0.0086: the network is certain the causes are
separate. The between term measures **indecision, not disagreement**. That
bin's total of 4.85 is essentially all `within`.

**The green line is flat by construction, not by fit.** Any constant weight,
whatever value is chosen, gives a flat line, because fixing `w` removes the
`w(1-w)` factor. Green pinned at 3.42–3.44 across every bin is the null
hypothesis drawn on the plot.

### Hump values across your runs

| run | prior | var_vis network | var_vis analytical | ratio | var_prop network | var_prop analytical | ratio |
|---|---|---|---|---|---|---|---|
| pcommon028 | 0.28 | **+0.18** | +0.22 | 0.80 | +0.12 | +0.17 | 0.73 |
| pcommon03 | 0.3 | **+0.23** | +0.27 | 0.85 | +0.18 | +0.20 | 0.91 |
| flagship | 0.5 | **+0.83** | +0.89 | 0.93 | +0.66 | +0.69 | 0.95 |
| pcommon07 | 0.7 | **+1.58** | +1.71 | 0.92 | +1.19 | +1.30 | 0.92 |

The hump is present in every run and in both output channels, and the network
reproduces it at 92–96 % of its analytical magnitude on the flagship and
`pcommon07`, 73–91 % at the two low priors (on the wide configuration it was
60–75 % everywhere). The same slight conservatism the position regression
reports, appearing in the uncertainty channel, but now small.

The hump's absolute size grows with the prior (0.2 → 0.8 → 1.6 deg²), because
the statistic averages the intermediate bins against the confident ones and a
higher prior puts more trials, at larger disparity, into the intermediate
zone. Across the sweep (§8.3) this growth is linear from 0.3 to 0.9.

The network also slightly *over*-reports variance in the confident bins (by
0.04–0.05 deg² at the two ends) and *under*-reports in the ambiguous ones (by
up to 0.14 at bin 0.25). That is compression toward the mean, the signature of
a read-out smoothing a sharply peaked target — at a tenth of the size it had
on the wide configuration.

### `var_prop` is the same figure at about three-quarter scale

Figure 10 is not an independent result; it is a consistency check. Its between
term peaks at 1.81 against 2.34 for `var_vis`, and its hump is +0.66 against
+0.83.

The reason is arithmetic. Recovering `|d|` per bin as
`sqrt(between / mean[w(1-w)])`, the vis/prop ratio is

```
1.14  1.14  1.14  1.14  1.14  1.14  1.15  1.15  1.10  1.16
```

— flat at about 1.14 across all ten bins. The between term goes as `d^2`, hence
1/1.14² ≈ 0.77 of the visual one.

`d_prop` is 1.14 times smaller than `d_vis` because **the two cues are nearly
equally reliable in this configuration**: the segregated visual estimate of
the hand has variance 4.64 (visual noise 1.1 plus the 3.8 deg² of eye-position
uncertainty it passes through, shrunk a little by the 10° prior) against 4.08
for proprioception. Precision weighting
predicts the ratio `4.64 / 4.08 = 1.137`, and the measured 1.14 is that number.
(On the wide configuration proprioception was twice as reliable and the ratio
was 1.97.)

### One caveat to carry

The middle bins are thin: 179–345 trials in the four bins from 0.2 to 0.6
(651 at 0.65), against 2141 at the left edge and 2034 at 0.75. The peak of
the hump is estimated from roughly 950 trials in total. The per-bin network means there
carry standard errors of about 0.03 deg² (trial-to-trial sd 0.4–0.55), so the
0.06–0.14 under-report in the ambiguous bins is resolved, if only just.

## 7.7 The five-way model comparison

Compares the network against five candidate strategies, binned by posterior
decile:

| strategy | prediction |
|---|---|
| **averaging** | `w·fused + (1−w)·seg` — the target, Bayes-optimal |
| **integration** | always `fused` (equivalent to p_common = 1) |
| **segregation** | always `seg` (equivalent to p_common = 0) |
| **selection** | `fused` if `w > 0.5` else `seg` — hard choice, not averaging |
| **fixed** | `w*·fused + (1−w*)·seg` with the single best constant `w*` |
| *(matching)* | `fused` with probability `w` — bonus comparator |

Two outputs: overall RMSE against the network, and **per-decile implied weight**.

The per-bin statistic is a **least-squares slope**, not a mean signed bias.
Averaging signed biases within a bin cancels, because Δ is signed and roughly
symmetric within a bin — that is why the first version of this figure looked
flat and had to be changed.

`fixed` nests integration (w* = 1) and segregation (w* = 0), so in-sample it can
never lose to them. The code therefore prefers a parameter-free strategy when it
comes within 0.5% — an implicit complexity penalty.

**Your overall RMSEs, visual output:**

| run | averaging | selection | fixed | segregation | integration | matching | best |
|---|---|---|---|---|---|---|---|
| pcommon028 | **0.679** | 0.802 | 0.801 | 0.806 | 6.293 | 0.949 | averaging |
| pcommon03 | **0.683** | 0.818 | 0.844 | 0.859 | 6.142 | 0.969 | averaging |
| flagship | **0.606** | 0.781 | 0.964 | 0.993 | 5.198 | 0.943 | averaging |
| pcommon07 | **0.536** | 0.715 | 1.117 | 1.198 | 4.026 | 0.868 | averaging |

Same verdict on the hand output in all four runs (averaging 0.505–0.727, the
runner-up 0.647–0.821).

**Flagship per-decile implied weight** (`bin_weight_net` against the
strategies' predictions):

| decile centre | network | averaging | selection | fixed |
|---|---|---|---|---|
| 0.000 | 0.009 | 0.000 | 0.000 | 0.045 |
| 0.000 | −0.001 | 0.000 | 0.000 | 0.045 |
| 0.044 | 0.023 | 0.034 | 0.000 | 0.045 |
| 0.336 | 0.301 | 0.310 | 0.018 | 0.045 |
| 0.604 | 0.584 | 0.594 | 1.000 | 0.045 |
| 0.712 | 0.676 | 0.707 | 1.000 | 0.045 |
| 0.761 | 0.739 | 0.758 | 1.000 | 0.045 |
| 0.784 | 0.770 | 0.786 | 1.000 | 0.045 |
| 0.824 | 0.815 | 0.826 | 1.000 | 0.045 |
| 0.911 | 0.787 | 0.914 | 1.000 | 0.045 |

**How to read it.** Averaging wins on RMSE in every run and both outputs, and
integration loses catastrophically (5.2 vs 0.6), which is the sanity check that
the network is not simply always fusing. The margin over selection, the
nearest real alternative, is 18–33 % of the averaging RMSE, largest at the
high prior where selection has the most trials to be wrong on.

In the per-decile table the network tracks averaging to within 0.03 up to the
0.82 decile and falls short only in the top one — 0.79 where averaging
predicts 0.91. That top-decile shortfall is the same under-fusion the position
regression measures; on the wide configuration it began at the 0.75 decile and
reached 0.50 against 0.94, so most of it is gone. The hand channel shows the
same thing one decile earlier (0.70 and 0.77 against 0.83 and 0.91).

The decile centres bunch (two at ≈0.000, 0.044, then a jump to 0.34, then
0.60–0.91) because the posterior distribution is bimodal — deciles of a bimodal
variable are not evenly spaced. This is honest, not a plotting bug. The
`fixed` weight is 0.045 (0.014 at `p_common` 0.28, 0.10 at 0.7): the single
constant that best explains the mean output is almost pure segregation,
because the signed disparities on the confidently segregated trials are the
largest and dominate the least-squares fit.

## 7.8 What the hidden layers carry

Ridge regression (`alpha = 1.0`) from each hidden layer's activations to a
target, scored by held-out R² on a 25% split.

The target is `post_c1`, the **trial-wise posterior** — never the prior. The
prior is one number per network; "decoding" it would just read out a disparity
confound.

**Your results:**

| run | SIL (layer0) | MSL (layer1) | twin SIL | twin MSL |
|---|---|---|---|---|
| pcommon028 | 0.890 | **0.966** | 0.296 | 0.286 |
| pcommon03 | 0.899 | **0.972** | 0.238 | 0.329 |
| flagship | 0.888 | **0.973** | 0.365 | 0.288 |
| pcommon07 | 0.887 | **0.976** | 0.266 | 0.232 |
| pcommon1 | −0.041 | −0.041 | −0.041 | −0.041 |
| pcommon0 | 0.072 | 0.276 | 0.150 | 0.151 |

**How to read it.**

- **The posterior is linearly decodable from the MSL at R² = 0.97–0.98** in
  every causal run, and — new with this configuration — it is already
  decodable at **0.89** from the SIL. On the wide domain the SIL gave only
  0.27–0.48, and the rise across the layers was the emergence argument. Here
  the rise is 0.89 → 0.97. The likely reason is the domain: with the prior sd
  at 10° and the noise at 1–2°, the posterior is close to a function of the
  disparity alone (§6.2: the within-disparity-bin range of the posterior is
  0.37, so reliability moves it by a third at most), and the disparity is a
  *linear* function of the input positions, which a single layer of 64 sigmoid
  units reading three population codes can already expose; a ridge decoder
  with 64 features has no trouble finding it. Whatever the mechanism, the
  emergence claim no longer rests on the SIL-to-MSL rise; it rests on the
  twin.
- **The always-fuse twin reaches only 0.23–0.37, at either layer.** The twin
  has the same architecture, the same inputs and the same training recipe, but
  was trained on fused targets only, so it was never asked for anything causal
  — and the posterior is no more decodable from its MSL than from its SIL.
  Same inputs, same first-layer width, and the posterior is three times less
  decodable: the contrast is what
  the twin exists for, and it is as clean as before (0.19–0.21 on the wide
  domain).
- **The two controls decode at ≈0.** This is expected and not a failure: with
  the prior at 1.0 the posterior is *constant* (every trial has p = 1), so there
  is no variance to explain and R² comes out slightly negative. At 0.0 the
  posterior is not quite constant (it ranges over 10⁻¹⁰ numerically), which is
  what the 0.28 is reading; it is not a result. These rows are a consistency
  check.

`mu_vis` also decodes from MSL at R² = 0.996 (flagship), confirming the layer
carries the estimate itself, not only the causal variable.

## 7.9 Unit-level analyses (MSL)

### Congruent / opposite classification

Sweeps the visual position and the proprioceptive position independently
(the other cue held at 0, eye at 0, noiseless rates, mid-range reliability), then
correlates each MSL unit's two tuning curves.

```
congruency index = corr( response to visual sweep, response to prop sweep )
> +0.5  -> congruent   (same preferred direction for both cues)
< -0.5  -> opposite    (opposite preferred directions)
otherwise -> mixed;  no measurable tuning -> untuned
```

This follows the congruency logic of Rideaux et al. (2021). The hypothesis is
that opposite cells encode *disagreement* between cues — exactly the quantity
causal inference needs.

**Your counts (64 MSL units):**

| run | congruent | opposite | mixed | untuned |
|---|---|---|---|---|
| pcommon028 | 12 | **31** | 21 | 0 |
| pcommon03 | 17 | **31** | 16 | 0 |
| flagship | 15 | 21 | 28 | 0 |
| pcommon07 | 18 | 17 | 29 | 0 |
| pcommon1 | **42** | **11** | 11 | 0 |
| pcommon0 | 25 | 21 | 18 | 0 |

**How to read it.** The four causal-inference runs produce a balanced or
opposite-heavy population (opposite ≥ congruent on three of the four; 18 / 17
at 0.7), and the two low-prior
satellites — the ones with the most segregated trials — have the most
opposite units (31 of 64). The p_common = 1 control produces 42 congruent
against 11 opposite — which makes sense: a network that always fuses has no
use for disagreement detectors. That contrast is a useful piece of evidence
that the opposite units in the flagship exist *because* of the causal task.
(The always-segregate control, `pcommon0`, sits in between, 25 / 21: a
network that reports each cue separately has a use for both signs.)

### Congruent-minus-opposite balance

Per trial: mean activity of congruent units minus mean activity of opposite
units. Correlated with `post_c1`.

**Your results:**

| run | correlation | R² of a 1-D linear read-out |
|---|---|---|
| pcommon028 | **−0.436** | 0.190 |
| pcommon03 | **−0.373** | 0.139 |
| flagship | **−0.373** | 0.139 |
| pcommon07 | **−0.725** | 0.526 |
| pcommon1 | 0.017 | 0.000 |
| pcommon0 | 0.112 | 0.013 |

**How to read it — the sign is not what the hypothesis predicts, and it is
not the sign the wide configuration gave.** All four causal runs now show a
*negative* correlation, −0.37 to −0.73: the opposite units are relatively
*more* active when the posterior favours a common cause. The two controls
correctly show nothing, since their posterior is constant. On the wide
configuration the same statistic was +0.50 and +0.54 on two runs and −0.05 on
the third (`CROSS_PRIOR_RESULT.md`).

So the balance is reproducible *within* a configuration and flips *between*
configurations. That rules it out as a finding in either sign. The statistic
is a crude one — a difference of mean sigmoid activations, which depends on
where each unit's baseline sits as much as on its tuning, and a unit is
"opposite" by the sign of its tuning correlation, not by what it does on a
disparate trial — and a quantity that reverses with the input domain while
every behavioural measure in §7.2–7.7 stays put is telling you about the
statistic, not the network. Do not present it. If the unit-level story
matters for the paper, the question has to be asked properly (a decoder from
each subpopulation separately, or the per-unit correlation with the posterior
rather than a population difference).

### Lesion

Ablates a subpopulation's MSL units and re-measures the read-out error. Two
things make this interpretable, and both were added after the first version of
this analysis gave misleading numbers:

**1. Mean-clamping, not zeroing.** An ablated unit is clamped to its *mean
activation across trials*. That removes the unit's information — it no longer
varies with the trial — while leaving the read-out's operating point intact.
Zeroing is wrong for a sigmoid layer: the per-unit mean activations spread
over most of (0, 1) (on the wide flagship they ran 0.013 to 0.990, median
0.41), so forcing a unit to 0 does not remove it, it injects a large constant
perturbation the read-out's weights and biases were never calibrated for. The
damage then measures the size of that perturbation, not the unit's role.

**2. A size-matched random baseline.** Ablating any *k* of 64 units costs
something. The question is whether ablating *these k* costs more than ablating
*k arbitrary ones*. For each subpopulation the code draws `n_random` random
subsets of the same size, ablates each, and reports where the real lesion falls
in that null distribution as a z-score and a percentile.

**Your flagship, mean-clamp mode, as stored in `results/flagship/metrics.json`**
(intact RMSE 0.441; `lesion_n_random` defaults to 100 draws in stage 3):

| ablation | k | RMSE | random null | z | percentile |
|---|---|---|---|---|---|
| congruent | 15 | 2.276 | 2.035 ± 0.293 | +0.82 | 79% |
| opposite | 21 | 2.366 | 2.669 ± 0.308 | −0.98 | 16% |
| mixed | 28 | 3.021 | 3.356 ± 0.347 | −0.97 | 17% |

The stored block also carries `rmse_per_output` for each lesion (2.70 / 0.44 /
3.62 / 0.37 for the congruent lesion, in the order mu_vis, var_vis, mu_prop,
var_prop, against 0.44 intact), but not a per-output null.

And the same three z-scores on the other runs:

| run | congruent z (k) | opposite z (k) | mixed z (k) |
|---|---|---|---|
| pcommon028 | +0.12 (12) | +0.13 (31) | +0.03 (21) |
| pcommon03 | **+2.58** (17) | +0.67 (31) | −1.31 (16) |
| pcommon07 | **+2.49** (18) | +0.30 (17) | −1.01 (29) |
| pcommon1 | **+2.91** (42) | **−3.03** (11) | +0.37 (11) |
| pcommon0 | +2.09 (25) | +1.56 (21) | −2.08 (18) |

**How to read it.** On the flagship, **no subpopulation is more load-bearing
than a random set of the same size**: all three z-scores lie within ±1.
Removing any 15–28 of 64 units costs a lot (RMSE 0.44 → 2.3–3.0), but removing
*these* units costs the same as removing any others. Across the family the
congruent lesion is the only one that is ever reliably worse than random, and
it is so on `pcommon03`, `pcommon07` and both controls (z 2.1–2.9) but not on
the flagship or `pcommon028`; the opposite lesion is never worse than random
(z −3.0 to +1.6), and the mixed lesion never. On the wide flagship the
congruent lesion had z = +6.99, concentrated in the mean channels, and the
opposite lesion z = −3.96; the first of those does not carry over.

**What this does and does not say.** The one statement that survives every
run is the negative one: the network does not *need* its opposite units to
produce its four outputs, so whatever they carry is carried redundantly. The
positive statement — that the position estimates are carried by congruent
units — holds on four runs out of six and not on the flagship, and should not
be made. Necessity and representation are different questions; this analysis
only answers the first, and on the flagship it answers it with "no
subpopulation is necessary", which is what a distributed code looks like.

(The zero-clamp comparison that motivated mean-clamping, and the per-output
z-scores, were a one-off 200-draw run on the wide flagship; they are not in
`results/` and are not repeated here.)

### Reference-frame (RF) shift and gain fields

Sweeps visual position at five eye positions, each time presenting the sweep in
*spatial* coordinates and encoding it retinally (`x_vis = s − e`, `x_eye = e`).
For each unit, fits how its preferred spatial position moves with eye position.

```
shift gain 0   -> the unit codes position SPATIALLY (body frame)
shift gain +1  -> the unit still codes RETINALLY
```

Peak response versus eye position is the classical **gain field**.

**Your median shift gains:**

| run | median shift gain |
|---|---|
| pcommon028 | −0.040 |
| pcommon03 | −0.076 |
| flagship | **−0.087** |
| pcommon07 | +0.041 |
| pcommon0 | +0.096 |
| pcommon1 | +0.063 |

**How to read it.** Median gains within ±0.1 of zero in every run (against 1
for a retinal code) mean the MSL population codes position in spatial
coordinates — the reference-frame transformation is complete by that layer.
This is the continuity check with the published paper. The gains are a few
times larger in magnitude than on the wide configuration (±0.02 there), which
is expected: the eye now moves over ±9° instead of ±18°, so the five eye
positions of the sweep span half the range and the per-unit fit of preferred
position against eye position has half the leverage.

## 7.10 Behavioural signatures

### Bias versus disparity (Körding Fig. 2e analog)

```
bias = network_hand_estimate - seg_prop_mu
```

— how far the hand report is pulled toward vision, relative to the hand-alone
solution — binned by **signed** disparity, with the Bayes-optimal prediction
`post_c1 · (fused_mu − seg_prop_mu)` overlaid. Bins holding fewer than 30 trials
are dropped.

The expected shape is non-monotonic: the pull grows with disparity while the
posterior is still high, then collapses back toward zero as the disparity itself
becomes evidence for two separate causes.

### Conditioned on the inferred cause (Fig. 3b–c analog)

Splits trials by the network's *own* causal judgment (the visual hybrid read
of the weight > 0.5, which exists on every trial) and plots bias against
|disparity| separately for each branch.

Conditioning on "inferred two causes" selects trials whose noise happened to
exaggerate the disparity, which can produce a counter-intuitive **negative**
bias — a fingerprint of inference over causal structure that no fixed-weight
scheme produces.

**Your results** (`conditioned_bias_negative_seen`, on the 2026-10-03 grid;
the flag reads the same on the old grid):

| run | negative bias seen in the separate-cause branch | the branch itself (\|d\| bins 2–30°, bias in deg) |
|---|---|---|
| pcommon028 | **True** | +0.07 +0.05 +0.05 +0.02 **−0.04** 0.00 0.00 0.00 **−0.13** (318–786 trials per bin) |
| pcommon03 | **True** | **−0.08 −0.11 −0.04 −0.04 −0.06 −0.07 −0.07 −0.12 −0.38** (203–763) |
| flagship | **False** | +0.15 +0.09 +0.07 +0.10 +0.13 +0.13 +0.12 +0.18 (297–553; no 2° bin survives) |
| pcommon07 | **True** | +0.11 +0.08 **−0.01 −0.07 −0.11 −0.02** +0.07 **−0.26** (78–332; from 4°) |

**How to read it.** The flag is a single boolean — "does any bin of the
separate-cause branch have a negative mean bias" — and it is True on three of
the four runs, but not in the pattern the mechanism predicts: it is absent on
the flagship and present at 0.7, so it does not grow as the prior falls. On
the wide configuration it was True only at 0.28. The curves behind it say
more than the flag. At `p_common` = 0.3 the inferred-two-causes branch is
negative in **every** bin from 2° to 30°, by 0.04–0.12° on bins of 200–760
trials (a bin mean's standard error is 0.02–0.05° here: the hand output's
residual sd of 0.6–0.7° over √n), so for that network the truncation effect
is real and consistent; at
0.28 it appears in two bins and at 0.7 in five, at the same few hundredths
of a degree; on the flagship the branch is positive everywhere. One network
per prior, and the sweep does not measure it, so this stays a lead rather
than a result — but it is now a lead with a curve, not a boolean on thin
bins: the old grid put the flag on 16–44-trial bins at ±40°, the new one
rests every bin on at least 78 trials.

## 7.11 Transition fit

Fits `w = 1 / (1 + exp(k(|d| − d0)))` to the implied weight against
|disparity|, giving a midpoint `d0` (the disparity at which fusion gives way to
segregation) and a sharpness `k`. Also fitted to the analytical posterior for
comparison.

**A guard on the fit.** `curve_fit` can report convergence on a midpoint far
outside the data — on an undertrained smoke run it returned −48219°, which is
not a transition but a flat curve fitted by a logistic whose midpoint ran away.
The fit is therefore rejected (returning NaN) unless `k > 0` and `d0` lies
within a quarter-span of the observed disparity range. A NaN midpoint means "no
transition was measurable", which is information; a large finite one would be a
silent lie.

**Your results** (fitted to the visual hybrid read on every trial):

| run | prior | network midpoint | analytical midpoint | network sharpness | analytical sharpness |
|---|---|---|---|---|---|
| pcommon028 | 0.28 | 2.63 deg | 2.88 deg | 0.402 | 0.385 |
| pcommon03 | 0.3 | 2.96 deg | 3.13 deg | 0.382 | 0.386 |
| flagship | 0.5 | 4.89 deg | 5.08 deg | 0.451 | 0.452 |
| pcommon07 | 0.7 | 6.56 deg | 6.64 deg | 0.509 | 0.536 |

**How to read it.** This is the cleanest cross-prior result in your data. The
midpoint moves monotonically with the prior — 2.63 → 2.96 → 4.89 → 6.56 deg
as the prior goes 0.28 → 0.3 → 0.5 → 0.7 — and tracks the analytical
prediction (2.88 → 3.13 → 5.08 → 6.64) to within 0.25 deg at every point,
always a little below it. The sharpness matches too (0.38–0.51 against
0.39–0.54 per degree). A network that believes one cause is more likely
tolerates more disparity before segregating, by the amount Bayes says it
should. The whole transition is at smaller disparities than on the wide
configuration (where the flagship's midpoint was 7.6 deg) because the
hypotheses are closer together on the 10° domain; the sweep in §8.2 extends
this table to nine priors.

---

# Part 8 — The cross-prior experiments

Two things go under this heading. The **satellite runs** (§8.1) are full
pipelines at four priors (the flagship and three satellites), each with the
complete analysis suite. The **prior sweep** (§8.2) is a dedicated experiment
at nine priors and three seeds that measures a smaller set of quantities. The
sweep is the stronger evidence and is what belongs in the manuscript; the
satellites are what let you inspect any individual network in depth.

## 8.1 The satellite runs

The satellites test one prediction: **implied weights should shift with the
prior, per Bayes**. Since the only difference between these configs is the
Bernoulli constant, any systematic movement is attributable to the prior.

| quantity | p = 0.28 | p = 0.3 | p = 0.5 | p = 0.7 | moves with prior? |
|---|---|---|---|---|---|
| transition midpoint (network, §7.11) | 2.63 | 2.96 | 4.89 | 6.56 | **yes, monotone** |
| transition midpoint (analytical) | 2.88 | 3.13 | 5.08 | 6.64 | yes |
| position regression slope, visual (§7.3) | 0.862 | 0.955 | 0.952 | 0.970 | rises, not strictly |
| weight on the posterior, slope (§7.2) | 0.969 | 0.965 | 0.977 | 0.973 | flat (all 0.97) |
| reliability slope (§7.5) | 0.906 | 0.978 | 0.887 | 0.894 | flat (all 0.9–1.0) |
| variance hump, var_vis (network) | 0.18 | 0.23 | 0.83 | 1.58 | **yes, monotone** |
| variance hump, var_vis (analytical) | 0.22 | 0.27 | 0.89 | 1.71 | yes |
| MSL posterior decoding R² (§7.8) | 0.966 | 0.972 | 0.973 | 0.976 | flat (all high) |
| balance-vs-posterior correlation (§7.9) | −0.44 | −0.37 | −0.37 | −0.73 | negative everywhere; see §7.9 |
| best strategy (§7.7) | averaging | averaging | averaging | averaging | consistent |

**What this supports.** The two quantities Bayes says must move with the
prior — the transition midpoint and the variance hump — move monotonically
with it, in the predicted direction and by the predicted amount (the midpoint
to within 0.25°, the hump to 80–93 % of its analytical size). The quantities
that should *not* depend on the prior — how closely the weight tracks the
posterior, how closely the variance tracks the mixture, how well reliability
is used at matched disparity — do not: slopes of 0.97 on the posterior and
0.9–1.0 on reliability at every prior.

**What it does not support.** The position-regression slope is lowest at
0.28 (0.862) and 0.95–0.97 elsewhere; with one seed per point that is a
tendency, not a dependence, and §8.2 is where the question is answered. The
balance correlation has the same sign on all four runs, but it is the
opposite sign to the wide configuration's (§7.9) and is not a finding.

Four points and one seed each is a weak design for a claim about a
continuous dependence. That is what §8.2 exists to fix.

## 8.2 The prior sweep — nine priors, three seeds

`scripts/05_prior_sweep.py` trains a network at every combination of nine priors
(0.1 … 0.9) and three seeds — **27 networks**, 50,000 trials and 300 epochs
each — and recovers the implied fusion weight from behaviour alone. No network
has a causal read-out; each is trained on the same four numbers as the flagship.

Between any two priors **only the Bernoulli constant differs**: same encoders,
same σ ranges, same architecture, same training protocol. Because the latent
streams are drawn before `C` selects between them, two datasets at different
priors are bit-identical on every trial whose causal structure did not flip.

The stored sweep (`results/prior_sweep/`, 2026-10-03) was run on the current
flagship configuration with the hybrid read (§7.2) and σ_out from
`results/pcommon1`; `curves.npz` holds the per-trial hybrid weights
(`t_w_vis`), so `--replot` can re-bin it without retraining. Every number
below is from that run.

### What the sweep is a test of

There is a weaker version of this claim that would be trivial and a stronger one
that would be false, so it is worth stating exactly.

**Not the claim:** "the network discovered the prior from nothing." It did not.
The prior is present in every target it was trained on.

**The claim:** the network's *implicit* fusion-vs-segregation trade-off — never
an output, only recovered from behaviour — depends on the prior in the way Bayes
says it must, quantitatively, with no free parameters.

Its force comes from what the alternatives predict:

| account | prediction as `p_common` varies |
|---|---|
| disparity heuristic | midpoint fixed — disparity statistics barely change |
| fixed-weight model | no transition at all, at any prior |
| model **selection** | a step at posterior 0.5 that moves, but with sharpness → ∞ |
| **model averaging with the correct posterior** | midpoint tracks the analytical value at every prior |

Only the last survives.

### How the per-bin weight is recovered

Each network's implied weight is read on every test trial by the hybrid read
of the visual channel (§7.2: the mixture-variance root where `Delta^2 <= c`,
the position ratio where `Delta^2 > c`), and the curve of Panel A is the mean
of that weight per disparity bin with the standard error of the bin mean
(`analysis.binned_weight`). No trial is filtered, so the curve carries
through zero disparity, where the ratio-based estimators of earlier versions
had nothing to say.

Two consequences are visible in Panel A and are deliberate:

- **The grid is the manuscript's.** The 18-bin grid (`GRID` in
  `05_prior_sweep.py`) runs from −30° to +30° with its two innermost edges at
  ±1.5°; it was chosen when the innermost bins could not identify a weight,
  and is kept so the stored sweep and a new one draw on the same bins.
- **Two guards.** A bin is dropped when it holds fewer than `MIN_COUNT = 25`
  trials (a floor on how few trials a mean may rest on; at `p_common = 0.9`
  the far-disparity bins hold 23–42 trials and two of them, −24° and +30°,
  fall under the floor) or when the standard error of its mean exceeds
  `MAX_SE = 0.05` (the largest surviving one is 0.045, at +12° for
  `p_common` = 0.9). Both can be overridden without retraining:
  `05_prior_sweep.py --replot --min-count N --max-se X` re-bins the stored
  per-trial weights (`t_w_vis` in `curves.npz`).
- **When a bin is dropped, the line breaks rather than bridging the gap.**
  Joining across a missing bin would draw a transition that was never
  measured.

### Your results — the aggregated table

From `results/prior_sweep/sweep.json`, `aggregated` block. All ± are **SEM
across the three seeds**, not within-run CIs. The distinction matters: a
within-run CI says how well *one network's* behaviour is pinned down; only the
across-seed spread says whether the result survives retraining.

| `p_common` | midpoint net | midpoint opt | posreg slope | reliability slope | hump net | hump opt | sharp net | sharp opt | MSL R² |
|---|---|---|---|---|---|---|---|---|---|
| 0.1 | −1.11 ± 0.13 | −0.94 | 0.884 ± 0.010 | 0.907 ± 0.021 | −0.09 ± 0.03 | −0.02 | 0.329 | 0.330 | 0.944 |
| 0.2 | 1.41 ± 0.02 | 1.58 | 0.921 ± 0.039 | 0.925 ± 0.014 | −0.01 ± 0.02 | +0.06 | 0.359 | 0.353 | 0.959 |
| 0.3 | 2.96 ± 0.04 | 3.12 | 0.951 ± 0.023 | 0.986 ± 0.007 | 0.25 ± 0.01 | 0.28 | 0.376 | 0.385 | 0.973 |
| 0.4 | 4.01 ± 0.03 | 4.23 | 0.919 ± 0.023 | 0.968 ± 0.031 | 0.54 ± 0.01 | 0.57 | 0.427 | 0.415 | 0.974 |
| 0.5 | 4.92 ± 0.02 | 5.09 | 0.951 ± 0.010 | 0.928 ± 0.033 | 0.85 ± 0.01 | 0.90 | 0.440 | 0.449 | 0.979 |
| 0.6 | 5.84 ± 0.04 | 5.89 | 0.978 ± 0.012 | 0.969 ± 0.024 | 1.21 ± 0.02 | 1.30 | 0.467 | 0.486 | 0.977 |
| 0.7 | 6.56 ± 0.04 | 6.65 | 0.977 ± 0.008 | 0.930 ± 0.018 | 1.60 ± 0.01 | 1.71 | 0.514 | 0.533 | 0.975 |
| 0.8 | 7.37 ± 0.02 | 7.43 | 0.983 ± 0.008 | 1.024 ± 0.025 | 1.96 ± 0.01 | 2.11 | 0.584 | 0.596 | 0.978 |
| 0.9 | 8.40 ± 0.08 | 8.46 | 0.988 ± 0.010 | 0.958 ± 0.021 | 2.33 ± 0.05 | 2.80 | 0.670 | 0.693 | 0.961 |

Read-out R² on `mu_vis` runs 0.993–0.997 across the whole sweep (`var_vis`
0.90–0.97, lowest at the two ends), so no network is simply fitting the task
worse at the extremes. The seed-0 network at each prior is the one the
satellite of §8.1 would be (the flagship's own numbers appear in the 0.5 row
of `rows`); its `best_val` goes 0.21 → 0.10 → 0.08 → 0.08 → 0.11 from 0.1 to
0.9, the U of Part 5.

### The three headline numbers

**1. The midpoint tracks Bayes with no free parameters.** Regressing the network
midpoint on the analytical midpoint:

```
all nine priors     slope 1.016   intercept -0.20 deg   R² 0.9998
                    mean |net − analytical| = 0.13 deg    max 0.22 deg (at p = 0.4)

p_common >= 0.2     slope 1.022   intercept -0.24 deg   R² 0.9997
                    mean |net − analytical| = 0.12 deg    max 0.22 deg

the 27 networks     slope 1.016   intercept -0.20 deg   R² 0.9988
individually        mean |net − analytical| = 0.15 deg    max 0.40 deg
```

Over a midpoint range of −1 to 8.5°, the network lands **within 0.13° of the
Bayesian prediction on average, with a fitted slope of 1.02 and R² = 0.9998,
over all nine priors.** The sharpness tracks too (0.33–0.67 against 0.33–0.69
per degree; ratio 0.96–1.03 at every prior). That is the strongest
quantitative result in the project. On the wide configuration the same fit
needed `p_common` = 0.1 excluded to reach R² 0.994; here the point at 0.1 is
the one with the largest seed spread (SEM 0.13° against 0.02–0.08° elsewhere)
and it still sits 0.18° from its prediction, so there is nothing to exclude.

**2. Model averaging wins at every prior and every seed** — 27 of 27 — in the
five-way comparison against full integration, full segregation, model selection,
and the best fixed-weight model. The margin is smallest at `p_common` = 0.1
(averaging 0.70–0.90 against 0.72–0.91 for the best fixed weight, 1–3 %),
where almost every trial is segregated and the strategies nearly coincide;
from 0.3 to 0.8 it is 14–37 % over the runner-up, and 6–14 % at 0.9, where
selection is right on most trials.

**3. A consistent, mild conservatism.** The position-regression slope rises
in trend from 0.884 at `p_common` = 0.1 to 0.988 at 0.9 (+0.12 per unit
prior, r = 0.91), and **`slope + 1.96·SEM < 1` at seven of the nine priors**
(the exceptions, 0.6 and 0.9, reach 1.002 and 1.007); on the 27 networks
individually the within-run CI excludes 1 in 20. The network applies slightly
less pull toward the fused solution than the ideal observer — by 12 % at the
lowest prior, by 2–8 % elsewhere — at every prior and across every seed.
This is the same deviation the flagship position regression reports (0.952),
now established as a property of the model rather than of one training run,
and small enough on this configuration that its prior dependence is the more
interesting part: the shortfall is largest exactly where the fused solution is
rarely the right one.

### The point at p_common = 0.1

Both midpoints there are *negative*: −1.11° for the network, −0.94°
analytical. That is not a sign error. At `p_common` = 0.1 the mean posterior
at zero disparity is only 0.34 — two sources drawn from a 10° prior are so
often as close as one that agreement is weak evidence — and it exceeds 0.5 on
just 2 % of trials (those with both cues far out in the prior's tails). The
logistic in |d| is therefore fitted to a curve that peaks below its own
midpoint, and `d0` is where the fitted curve *would* cross 0.5 if it were
extrapolated past zero. The fit is legitimate (the guard allows a midpoint
within a quarter-span of the observed range) and the network's extrapolation
matches the ideal observer's, which is the point; but the number is not a
disparity any trial has. Say so when the panel is shown, and that the seed
spread at this prior (−0.85, −1.19, −1.30) is the largest in the sweep. At
`p_common` = 0.2 the curve peaks at 0.54 and the midpoint (1.4°) is a real
crossing, just.

## 8.3 What the sweep settles that four runs could not

| question | the satellites said | 27 networks say |
|---|---|---|
| does the midpoint track Bayes? | yes, 4 points within 0.25° | **yes, slope 1.02, R² 0.9998 over all nine priors** |
| is the conservatism real? | slopes 0.86–0.97, one seed each | **yes — CI excludes 1 at 7 of 9 priors, 3 seeds; rises with the prior** |
| does model averaging always win? | 4 for 4 | **27 for 27** |
| does the reliability slope move with the prior? | flat, 0.89–0.98 | **flat: 0.91–1.02, mean 0.96, no trend** (see below) |
| does the balance result replicate? | same sign on 4 runs, opposite to before | *not measured by the sweep* |

**The reliability slope is flat and a little below 1.** Across the nine priors
it reads 0.907, 0.925, 0.986, 0.968, 0.928, 0.969, 0.930, 1.024, 0.958 —
mean 0.955, with no trend (r = +0.49 against the prior over the nine
aggregated points, +0.36 over the 27 networks, both the wrong sign for a
decline and neither significant on nine points). The across-seed CI excludes
1 at six of the nine priors. So the network uses reliability at matched
disparity at about 95 % of the Bayesian rate, at every prior. On the wide
configuration this statistic declined from 0.95 to 0.46 with the prior
(r = −0.87), which Part 12 of that guide flagged as a lead needing a control
for the changing composition of the surviving bins; on the 10° domain the
decline is gone, which is what one would expect if it was a property of the
bins that survived there rather than of the network. The composition caveat
still applies in principle (the statistic drops bins whose posterior spread
is below 0.05; §7.5), but it no longer has a pattern to explain.

### The variance hump across the sweep

The hump is a real effect over most of the range and undefined at the bottom.
The network-to-analytical ratio, `p_common` = 0.3 → 0.9:

```
0.89   0.94   0.94   0.93   0.94   0.93   0.83
```

It holds at 93–94 % from 0.4 to 0.8 and dips to 0.83 at 0.9. At `p_common` =
0.1 and 0.2 the *analytical* hump is itself zero (−0.02 and +0.06 deg²): the
statistic averages the intermediate-posterior bins against the confident ones,
and at those priors the confident side is almost entirely the segregated,
high-variance end, so the "hump" is negative or absent even for the ideal
observer. The network's values there (−0.09, −0.01) agree with that; the ratio
is just not a meaningful number when its denominator is zero. The absolute
hump grows linearly with the prior from 0.3 to 0.9 (0.25 → 2.33 deg², 3.5
deg² per unit prior, R² 0.998), tracking the analytical growth.

The dip at 0.9 is trial counts, not mechanism: the hump is defined on
intermediate-posterior trials (0.2 < posterior < 0.8) and at `p_common` = 0.9
the network has 4 % of its test trials there (295 of 7500, against 3642 for
the flagship), spread over six bins.

**For the manuscript:** supplementary, over `p_common` ∈ [0.3, 0.9] with the
ratio and the trial counts stated, and a sentence saying why 0.1 and 0.2 have
no hump to reproduce. On the wide configuration this figure had to be
restricted to [0.2, 0.7] because the ratio collapsed to 0.28 at the top; that
is no longer true.

---

# Part 9 — Every figure, and the calculation behind it

**Every figure is written to the PLOS ONE specification** (`cmsi/viz/style.py`
carries the numbers): drawn at its final printed size — 5.2 in wide for a
single panel, 7.5 in for a multi-panel figure, never taller than 8.75 in —
at 300 dpi, in Arial (Liberation Sans on Linux, which is metrically identical)
with every font between 8 and 12 pt, and lettered panels on every multi-panel
figure. Each figure is saved three times: `.png` to look at, `.tif` for
submission (flattened RGB, no alpha channel, LZW-compressed, 300 dpi metadata),
and `.svg` to edit; no `.pdf` is written. `save_figures` reads every file
back and warns if it breaks a limit;
`tests/test_figures.py` pins the contract.

**About the SVGs.** They are the same physical size as the raster files, text
stays text (editable in Inkscape or Illustrator, searchable, and set in Arial
on any machine that has it), and axes, lines and markers are vectors. The dense
point clouds — the 12,500-trial output scatters, the 50,000-trial gain plots,
the 2-D histograms — are embedded as single 300-dpi images inside the SVG
rather than tens of thousands of vector markers, which is what keeps the
whole set at about 3 MB instead of hundreds. Output is deterministic (no
timestamp, fixed element ids), so regenerating an unchanged figure gives a
byte-identical file. PLOS ONE does not accept SVG; the `.tif` is the
submission copy.

Titles are kept inside the figures by your decision (PLOS asks for them in the
caption instead); the panel letters are the one addition the journal requires.

## 9.1 `figures/inputs/` — what the network is shown

Drawn from the dataset, before any training. Worth checking first: most
"the model won't learn" problems are visible here.

**`01_tuning_curves.png`** — left panel: 8 of the Gaussian visual receptive
fields evaluated across the visual field at unit gain. Right panel: 8 push–pull
proprioceptive units. Confirms the bumps have sensible width and the push–pull
slopes have mixed sign.

**`02_population_heatmap.png`** — takes ~300 trials sorted by `x_vis` and images
each of the three population-code matrices (trials × units). The visual panel
should show a diagonal band — the bump tracking the measurement. The two
push–pull panels show smooth gradients rather than bumps, since they are linear
codes.

**`03_reliability_gain.png`** — scatter of total population activity against
`1/σ²`, one panel per group. Should be a straight line through the origin,
because gain = K/σ² by construction. This is the visual confirmation that
reliability reaches the network only through spike count.

**`04_latent_distributions.png`** — three histograms: body-frame disparity,
analytical `post_c1`, and the three per-trial noise variances overlaid. The
middle panel is the same quantity the calibration gate checks.

## 9.2 `figures/training/` — did it converge

**`01_loss_curves.png`** — train and validation loss per epoch on a log axis,
with the early-stopping epoch marked. Your flagship's best validation loss,
0.0784, came at epoch 182 (patience 20, so it stopped at 202).

**`02_per_output_loss.png`** — validation loss split by output column. A flat,
high curve for one output while the others fall is the scale-imbalance
signature; it is what `balance_loss` exists to prevent.

## 9.3 `figures/model/` — network versus observer

Sixteen numbered figures (seventeen files, since `08v` sits beside `08`).
Figures 01–07 exist for every run; 08–16 are produced only for
runs with a causal head and a prior strictly between 0 and 1 — which is why
`pcommon1` and `pcommon0` have 7 figures and the other four have 17. All of
them were rendered on 2026-10-03 from the rerun. The numbers quoted below are
the flagship's.

**`01_output_scatter.png`** — network against analytical, one panel per output,
with the identity line and R²/RMSE in each title. The first thing to look at.

**`02_error_histograms.png`** — distribution of `network − analytical` per
output, with mean bias in the title.

**`03_post_c1_vs_disparity.png`** — the analytical posterior binned by signed
disparity on the config's `disparity_grid` (19 centres from −30° to +30°, 2°
apart across the transition; §4.2). The classic Körding-shaped curve: high
near zero disparity, falling away on both sides — on the flagship 0.82 at 0°,
0.79 at ±2°, 0.68 at ±4°, 0.43 at ±6°, 0.16 at ±8°, 0.03 at ±10° and zero
beyond. This is a property of the *generative model*, not of the network — it
is the reference the next figure is read against.

**`04_fusion_weight.png`** — the classical view: the implied weight against
disparity, per channel, with the analytical posterior dashed. The weight is
the hybrid read (§7.2) — the mixture-variance root of the channel's variance
output where `Δ² ≤ c`, the channel's own position ratio where `Δ² > c` — so it
exists on every trial and the curve carries through zero disparity, where the
two hypotheses coincide and a ratio has nothing. Both channels encode the same
weight, so the two curves are two readings of one quantity; the visual one is
the more precise (§7.4). The curves are plain bin means on the config grid
(`analysis.mean_by_bin`, no error bars, no count floor; every bin holds 147
or more trials on the flagship — figure 15 is the version with error bars).
On the flagship the visual curve sits on the analytical one to within 0.05 at
every one of the 19 bins: 0.82 against 0.82 at the centre, 0.78 against 0.79
at ±2°, 0.65 against 0.68 at ±4°, 0.38 / 0.42 against 0.43 / 0.44 at ∓6°,
0.15 / 0.20 against 0.16 / 0.15 at ∓8°, and 0.00–0.02 against 0.03 and below
from ±10° out; the standard error of a bin mean is 0.002–0.015. The hand
curve lies within 0.04 of the visual one everywhere, mostly a little above it
in the outer bins (its ratio trials are the noisier ones, §7.4). (On the previous grid
the ±40° bins held 30 and 33 trials and showed the ratio's heavy tails; those
trials now fold into the ±30° bins.)

**`05_fusion_weight_by_reliability.png`** — the same implied-weight curve
computed separately at each of the config's `reliability_levels` (nearest-match
on `sig2_vis`). The prediction: less reliable cues tolerate more disparity
before segregating, so the transition midpoint moves outward. On this
configuration the levels (1.0, 1.1, 1.2 deg²) span the whole of the narrow
`sigma2_vis_range`, so the three curves differ by very little by design —
`06 reliability --levels 5` is the sharper version of this figure (Part 10,
`results/experiments/implied_weight/flagship/`), and even there the
analytical midpoints move by only 0.1° across the five levels.

**`06_decoding.png`** — bar chart of held-out R² for decoding `post_c1` from
each hidden layer, the layers named SIL and MSL as the paper names them.
Your flagship: 0.888 at SIL, 0.973 at MSL.

**`07_emergent_vs_imposed.png`** — the same bars for the causal network and
the always-fuse twin (grey) side by side. Your flagship: 0.888/0.973 against
0.365/0.288. The contrast is the emergence argument (§7.8 on why the SIL bar
is now high). Since 2026-10-03 both figures are drawn by the routine behind
the manuscript's figure 7A (`results._draw_decoding`): bars 0.2 of the group
spacing wide rather than matplotlib's 0.4, each carrying its value, the y
axis with headroom above 1.0 for the labels and the legend. On a control the
posterior is constant and the decoder's R² is slightly negative (−0.04); the
label then sits on the baseline rather than below the axes.

**`08_position_regression.png`** — the headline. Scatter of
`(network − seg)` against `post_c1 · (fused − seg)`, both in degrees, with the
identity line (Bayes-optimal) dashed and the actual fit drawn with its CI in the
legend. Your flagship: slope 0.952 [0.936, 0.969], intercept −0.044.

The vertical spine of points at x = 0 is the ill-conditioned zone — trials where
the two hypotheses agree. They are visible, and correctly carry almost no
leverage on the fit.

**`08v_variance_regression.png`** — the headline in the variance domain, and
the weight on the posterior. A: `var_vis output − var_seg` against
`v_opt − var_seg`, the optimal mixture's variance reduction, with the fit
(`analysis.variance_regression`; slope 1 = Bayes-optimal, no per-trial
division). B: the implied weight of the visual channel against the analytical
posterior on every trial, its points coloured by where each weight came from —
the `var_vis` root on Δ² ≤ c trials, the `mu_vis` ratio on Δ² > c trials —
with its fit (`analysis.weight_regression`); the legend gives each source's
share of the trials. Panel B is the manuscript's Figure 2B (§9.4), in place
of the position-domain regression since 2026-10-03. `metrics.json` carries
`variance_regression_vis/prop`, `weight_regression_vis/prop`,
`implied_weight_vs_post[_prop]` and `hybrid_read`. Your flagship:
variance-domain slope 0.977 [0.973, 0.981],
weight on the posterior 0.977 [0.967, 0.988], 42 % of the trials from the
root and 58 % from the ratio. The two domains agree with the position
regression on the verdict, measured on different outputs of the network. In
panel B the root trials form the tight band along the identity line (sd 0.04
against the posterior) and the ratio trials the wider cloud around it (sd
0.21); 0.08 % of the points lie outside the [−1, 2] frame.

**`09_variance_hump_vis.png`** and **`10_variance_hump_prop.png`** — four lines
against the analytical posterior in 10 bins: network Var output, analytical
mixture variance, the between-component term alone, and the fixed-weight
prediction (w̄ = mean posterior). The last is flat by construction and is the control.

Your flagship `var_vis`: the network peaks at 6.2 around posterior 0.15–0.35
and falls to 2.5 at 0.95, against a fixed-weight line pinned at 3.4
throughout. The hump is +0.83 for the network against +0.89 analytical (93 %);
`var_prop` reads +0.66 against +0.69.

Note the x-axis is the **posterior**, not disparity, and that the curve carries
two superimposed effects — a downhill within-component ramp plus the hump —
which is why the leftmost point is high without being part of the hump.
**§7.6 works through both figures line by line**; read that first if the shape
is not obvious.

**`11_model_comparison.png`** — two panels. Left: per-posterior-decile implied
weight for the network and all five strategies. This is the discriminating
panel — averaging predicts a smooth sigmoid tracking the posterior, selection
predicts a step at 0.5. Right: per-decile RMSE of each strategy against the
network.

Your flagship: the network tracks averaging to within 0.03 through every
decile but the top one, departs clearly from the selection step (which jumps
to 1.0 at the 0.60 decile where the network reads 0.58), and falls short of
averaging only in the top decile (0.79 against 0.91). The right panel shows
averaging's RMSE below every alternative in every decile.

**`12_behavioral_bias.png`** — two panels. Left: bias against signed disparity,
network versus Bayes-optimal (Körding Fig. 2e analog). Right: bias against
|disparity| split by the network's own causal judgment (Fig. 3b–c analog). Both
drop bins with fewer than 30 trials — without that filter the right panel swings
by several degrees on a handful of trials and reads as structure.

Your flagship, left panel: the pull peaks at −1.04 / +1.16° at ∓4–6°
disparity and collapses by ±10°, on top of the Bayesian curve (−1.15 / +1.18
at ∓4°, −1.10 / +1.13 at ∓6°, −0.55 / +0.52 at ∓8°) through the transition.
One thing to see and not over-read: on the positive side the network keeps a
pull toward vision where Bayes has none — +0.27° at 10°, +0.10 at 12°, +0.21
at 15°, +0.34 at 20° and +0.47 at 30° (162–247 trials per bin, so it is real)
— while the negative side sits within ±0.15° of zero. It is the +0.07°
hand-channel intercept of §7.3 seen bin by bin, and an asymmetry of this
network rather than of the task. Right panel: the "inferred two causes"
branch sits at +0.07 to +0.18° everywhere on the flagship, never negative
(`conditioned_bias_negative_seen` False; §7.10 has the four runs' curves).

**`13_congruency.png`** — two panels. Left: histogram of the congruency index
across MSL units, with the ±0.5 classification thresholds marked. Right: scatter
of the congruent-minus-opposite activity balance against the analytical
posterior, titled with the correlation.

Your flagship: the index histogram has 15 congruent, 21 opposite and 28 mixed
units — less cleanly bimodal than the wide flagship's — and the balance
correlates **−0.373** with the posterior. The other three causal runs give
−0.44, −0.37 and −0.73; the wide configuration gave +0.50. See §7.9 before
quoting either sign.

**`14_rf_shifts.png`** — two histograms. Left: distribution of RF shift gain
across MSL units, with 0 (spatial code) and +1 (retinal code) marked. Right:
distribution of gain-field slopes. Your flagship median shift gain: −0.087
(±0.1 across the family, against +1 for a retinal code). The left histogram
is panel B of the manuscript's figure 7 (§9.4).

**`15_weight_vs_posterior.png`** — the implied weight against the analytical
posterior in 10 equal-width posterior bins, every trial
(`analysis.weight_by_posterior`), with the Bayes-optimal identity line dashed
and 1.96 standard errors of the bin mean as error bars; a point sits at the
mean posterior of its trials, not at the nominal bin centre. A Bayes-optimal
model-averaging observer puts every point on the identity line. Your flagship
sits on it to within 0.03 from the first bin to the 0.84 bin (0.83 there) and
reads 0.88 in the top bin (mean posterior 0.94): the one visible shortfall,
0.06, is the top-bin shortfall that §7.7 measures as 0.13 on its
least-squares decile (a different binning and estimator), and the error bars (±0.004 to ±0.04;
the middle bins hold 179–345 trials) are about the size of the markers. (Earlier versions drew this figure twice, once by the
σ_w-filtered ratio and once by the least-squares per-bin weight, to show the
filter's selection bias; both estimators are gone.)

**`16_weight_distribution.png`** — the implied weight on every trial, one
panel per channel, as a histogram against the analytical posterior's own
distribution drawn as an outline. Nothing is filtered; the axis is clipped to
[−1, 2] with the overflow piled into the edge bins, and the panel prints the
fraction of trials outside, the share read from the position ratio, and the
median and IQR. A Bayes-optimal network would reproduce the outline: on the
flagship both reads do, with 0.08 % (visual) and 0.04 % (hand) of trials
outside the frame; the ratio trials widen the peaks a little because on them
the read *is* the ratio, so the histogram's two modes are broader than the
posterior's. The 16 % of visual per-trial weights that fall outside [0, 1]
(14 % below 0, median −0.04; 2 % above 1, median 1.08 — 27 % of the ratio
trials and 1 % of the root trials) are inside the frame and visible as the
skirts of the two modes.

## 9.4 `results/manuscript/` — the standard panel, and every figure built on it

**Where.** All manuscript figures live in one directory, `results/manuscript/`,
one folder per figure, whichever run produced them (the flagship run for the
per-run figures, the sweep for the cross-prior figure). Rendering a run other
than the flagship writes to `results/manuscript_<run>/` instead, so a
satellite can never overwrite the paper's figures. Each folder holds every
format of its figure — `.png`, `.tif` and `.svg` — saved at exactly
the frame (`viz.exact_frame`) rather than cropped to content, which is what
lets panels tile edge to edge.

**The standard** (`cmsi/viz/manuscript.py`). A panel is a cell of fixed size
whose plotting area sits inside fixed **absolute margins** — 0.56 in left,
0.44 in bottom, 0.31 in top, 0.10 in right (`PANEL_MARGIN`) — with one font
specification (`PANEL_FONT`: 8 pt ticks, 9 pt axis labels and titles, 8 pt
legends; panel letters 12 pt bold). Three cell widths share one height and
one set of margins:

| cell | size (in) | axes (in) | used for |
|---|---|---|---|
| square (`CELL_SQUARE`) | 2.5 × 2.5 | 1.84 × 1.75 | every square panel — the default |
| wide (`CELL_WIDE`) | 3.75 × 2.5 | 3.09 × 1.75 | rectangular panels, two to a row |
| full (`CELL_FULL`) | 7.5 × 2.5 | 6.84 × 1.75 | one panel across the page |

A figure is an arrangement of cells (`manuscript_grid` for uniform grids,
`cell_axes` for mixed layouts), so its size follows from its layout: 1 × 3
squares, 1 × 2 wides or one full = 7.5 × 2.5 in; 2 × 2 squares = 5.0 × 5.0 in;
a full over two wides = 7.5 × 5.0 in. Because the margins are absolute, every
axes in every figure starts at the same offset inside its cell and has the
same height — figures align when stacked. Any two panels in any two figures
have the same margins and fonts by construction. This is the standard for
every panel from here on, unless a figure is explicitly given another size.

| folder | contents | layout | size (in) | built from |
|---|---|---|---|---|
| `fig2_weight_regression_bias/` | `A_fusion_weight`, `B_weight_regression`, `C_bias_vs_disparity`, `row_ABC` | 3 squares; 1 × 3 row | 2.5 × 2.5 each; 7.5 × 2.5 | `04_fusion_weight`, panel B of `08v_variance_regression`, left half of `12_behavioral_bias` |
| `output_scatter_2x2/` | `output_scatter_2x2` | 2 × 2 squares, A B / C D | 5.0 × 5.0 | `01_output_scatter` |
| `error_histograms_2x2/` | `error_histograms_2x2` | 2 × 2 squares, A B / C D | 5.0 × 5.0 | `02_error_histograms` |
| `fig4_variance_hump/` | `row_AB` | 1 × 2 wides | 7.5 × 2.5 | `09_variance_hump_vis`, `10_variance_hump_prop` |
| `fig7_decoding_rf_shift/` | `A_decoding`, `B_rf_shift`, `row_AB` | 2 wides; 1 × 2 row | 3.75 × 2.5 each; 7.5 × 2.5 | `07_emergent_vs_imposed`, left half of `14_rf_shifts` |
| `prior_sweep/` | `prior_sweep_ABC` | a full over two wides | 7.5 × 5.0 | the cross-prior figure (§8.2, §9.5) |

(Folders without a figure number are named by content until one is assigned;
the names are single strings in `results.py` and `prior_sweep.py`.)

Deliberate details, all measured rather than estimated:

- the scatter titles read `mu_vis   R² 0.996   RMSE 0.61` with no `=` — with
  them the title is 2.03 in wide and touches the 2.5-in cell edge; without,
  1.82 in with 0.11 in to spare;
- superscripts are Unicode glyphs everywhere (`R²`, `deg²`, `w(1−w)d²`),
  never mathtext, which would set them at 6.3 pt in figures whose fonts are
  otherwise exactly {8, 9, 12} pt. For the same reason the prior is labelled
  "common-cause prior" rather than a subscripted *p*;
- figure 4 carries one legend, in panel A, since both panels draw the same
  four series; panel A's y axis has 40 % headroom so the legend sits above
  the curves — at 30 % the nearest marker was 0.035 in from the legend box;
- figure 2B (`panel_weight_regression`, since 2026-10-03 in place of the
  position-domain regression) carries two legends, because one of five
  entries would cover the cloud on a 2.5-in panel: the two sources with their
  shares of all trials in the lower right, where a good network has almost
  no points (high posterior, low weight), and the two lines in the upper
  left under the same y headroom as panel A (limits −0.25 to 1.4, ticks to
  1.0). The root trials are drawn last, so their tight band along the
  identity line shows on top of the ratio trials' wider cloud; the scatter
  is subsampled to 6,000 trials for drawing, the shares are not;
- in the cross-prior figure the colourbar is carved from panel A's axes width
  (0.62 in) so the cell's outer margins are untouched, and the y labels are
  shortened to fit a 1.75-in axes ("implied fusion weight", "network
  midpoint (deg)");
- figure 7 (`panel_decoding`, `panel_rf_shift`, added 2026-10-03) is two wide
  cells like figure 4. Panel A: held-out R² for p(C=1|x) per hidden layer,
  the causal network (blue) beside its always-fuse twin (grey,
  `COLORS["twin"]`), the layers named SIL and MSL as the paper names them;
  the bars are 0.2 of the group spacing wide (0.28 in) rather than
  matplotlib's 0.4, each carries its value in 8 pt above it, and the y axis
  runs to 1.45 with ticks to 1.0 so the two-entry legend sits above the
  "0.97" label rather than on it. Panel B: the RF shift gains of the 64 MSL
  units in 0.1-wide bins aligned to zero, white bar edges, the spatial (0,
  dashed) and retinal (+1, dotted) marks as vertical lines; the x range
  always includes both marks, so the retinal mark stands alone on the
  right (the gains run −1.3 to 0.4), the legend sits upper left over the
  low tail, 40 % headroom as in figure 4, and the median is in the title
  the way the error histograms carry their bias. Skipped, with a message,
  when `metrics.json` has no `post_c1_decoding_r2` or `analysis.npz` no
  `rf_shift_gain`.

The tests parse the SVGs and assert the cell size, identical axes rectangles
across cells, the font-size set, and that no artist crosses its own cell's
edge; legends are checked against the data they could cover.

**Placing them.** Use each file at its native size — `width=0.333\textwidth`
for a 2.5-in panel, `0.667\textwidth` for a 5-in grid, `\textwidth` for a
7.5-in figure. Scaling changes the font size. A single 2.5-in panel is below
PLOS's 2.63-in minimum on its own; it is a component, and the composed
figures are the submission files.

Regenerate with `python scripts/04_figures.py --run flagship --only manuscript`
(everything from the flagship run) and `python scripts/05_prior_sweep.py
--replot` (the cross-prior figure). `make figures RUN=flagship` includes the
former.

## 9.5 `results/prior_sweep/figures/` — the cross-prior experiment

Produced by `scripts/05_prior_sweep.py`, not by `04_figures.py`. These are not
per-run figures: each one summarises all 27 networks.

**`prior_sweep.png`** (with `.tif` for submission and `.svg` to edit) — the
main figure, three panels, 7.5 in wide. Its manuscript version,
on the standard panel, is `results/manuscript/prior_sweep/prior_sweep_ABC`
(§9.4).

- **Panel A** — implied fusion weight against signed body-frame disparity, one
  curve per prior, colour on a sequential viridis ramp (light = low prior).
  Solid with error bars: the network, the mean of the implied weight per bin
  (`analysis.binned_weight`, §8.2).
  Dashed: the analytical posterior for the same prior. The curves fan out in an
  orderly family — at `p_common` = 0.1 the network's weight never exceeds
  0.33 even at zero disparity (§8.2, the point at 0.1); at 0.5 it reads 0.80
  at ±1.5° and is back at zero by ±12°; at 0.9 it holds 0.97 at the centre and
  0.43 at ±9°. Every curve lies on its dashed partner to within 0.05 in every
  bin holding more than 100 trials. Error bars are `1.96 × SE` where `SE` is the standard error
  of the bin mean of the per-trial hybrid weights (not σ_out / √ΣΔ², the
  earlier ratio-based estimator). A broken line marks a dropped bin, not
  missing data — see §8.2; with the current guards two bins are dropped, both
  at `p_common` = 0.9 (−24° with 23 trials, +30° with 24), and the surviving
  outer bins at 0.8–0.9 (30–100 trials) wobble by up to ±0.1 around zero
  (−0.08 at −15° for 0.9), which is what 30-trial means of a heavy-tailed
  ratio look like. The curves are the first seed's; the stored per-trial
  arrays are the first seed's too.
- **Panel B** — network transition midpoint against analytical midpoint, one
  point per prior, with an identity line. No parameter relates the two axes.
  With three or more seeds the points carry across-seed SEM bars. If every
  logistic fit is rejected by the §7.11 guard the panel prints "no usable
  transition fit" instead of crashing. On the current sweep the nine points
  lie on the identity line to within 0.22° (slope 1.016, R² 0.9998), the
  lowest one at (−0.94, −1.11) — a negative midpoint on both axes, explained
  in §8.2.
- **Panel C** — position-regression slope against prior, with the Bayes-optimal
  value of 1 marked. The legend states which kind of error bar is drawn:
  "within-run 95% CI" for a single seed, "mean ± 95% CI across N seeds"
  otherwise. Read the label — the two mean different things (§8.2). On the
  current sweep the points climb from 0.88 to 0.99 with the prior, all below
  1 and seven of nine significantly so.

**`variance_hump_vs_prior.png`** — supplementary. Mid-ambiguity variance
elevation, network against analytical, per prior. Interpret over `p_common`
∈ [0.3, 0.9] (ratio 0.83–0.94); at 0.1 and 0.2 the analytical hump is itself
zero, so there is nothing to reproduce there (§8.3).

Both are redrawn by `05_prior_sweep.py --replot` from `sweep.json` and
`curves.npz`, with no training; `--min-count` and `--max-se` re-bin Panel A
from the stored per-trial arrays of the first seed (§8.2).

---

# Part 10 — How to run things

```bash
make test                                       # 97 property tests, ~45 s
make calibrate CONFIG=configs/flagship.yaml     # the gate alone
make all       CONFIG=configs/flagship.yaml     # full pipeline, 50k trials
make quick     CONFIG=configs/flagship.yaml     # same, 8k trials / 60 epochs
make sweep     SEEDS="0 1 2"                    # the cross-prior sweep, ~45 min
make figures   RUN=flagship                     # redraw one run's figures
python scripts/04_figures.py --run flagship --only manuscript
                                                # just the standard-panel figures
python scripts/05_prior_sweep.py --replot       # redraw the sweep, no training
python scripts/05_prior_sweep.py --replot --min-count 25 --max-se 0.05
                                                # re-bin Panel A from curves.npz
```

`04_figures.py --only` takes `inputs`, `training`, `model` or `manuscript`;
without it every group is rendered.

**Side experiments** live outside the numbered stages. `scripts/06_implied_weight.py`
(`src/cmsi/experiments/implied_weight.py`) applies the implied-weight read
(§7.2) to any pipeline run, writing only under
`results/experiments/implied_weight/<name>/`: `figures` draws the pipeline's
weight figures (04, 05, 08v, 15, 16) for one run and prints its numbers, and
`reliability` draws the weight against disparity at many reliability levels of
each input, per channel, with the transition midpoint against the level. Its
`train` sub-command takes a configuration (a yaml, or the flagship with
`--set key=value` overrides) through stages 0–4 into `results/<name>/`, together
with a p_common = 1 control of the same configuration in
`results/<name>_pcommon1/`, and records the configs in `configs/experiments/`;
the run then has everything a pipeline run has, and the analyses find its
control by themselves. The configuration sweeps that led to the current read
(`07_architecture`, `08_fixed_variance`, `09_sweep`, and the `sigma`,
`compare` and `design` sub-commands of 06) were retired on 2026-09-29; their
results stay under `results/experiments/`, and `IMPLIED_WEIGHT_RESULT.md`
has the story.

Every figure command writes `.png`, `.tif` and `.svg` side by side (Part 9);
the two redraw commands are what to run after a style change, since neither
retrains anything.

**`make all` does not include the sweep**, and the sweep does not depend on
`make all` having been run — except for `pcommon1`, which supplies σ_out. The
two are separate experiments with separate output directories.

Stage by stage:

```bash
python scripts/00_calibrate.py     --config configs/flagship.yaml
python scripts/01_generate_data.py --config configs/flagship.yaml --name flagship
python scripts/02_train.py         --data flagship --run flagship
python scripts/03_analyze.py       --run flagship --twin flagship_twin --control pcommon1
python scripts/04_figures.py       --run flagship
```

**Order matters for `--control`.** Stage 3 takes σ_out from the p_common = 1
control, so that control must be trained *and analysed* first. `run_all.sh`
passes `--control pcommon1` only if `results/pcommon1/metrics.json` exists and
contains `residual_std`; otherwise stage 3 warns and falls back to the run's own
residuals. The fallback is conservative rather than wrong — the flagship's own
residuals also contain any causal-inference misweighting, so the weight's
nominal sd comes out too large, never too small. Whichever it used, stage 3
records the value as `sigma_out` and its origin as `sigma_out_source` in
`metrics.json` (§7.2).

**`--config` re-analyses without retraining.** The `analysis` block of a
config (`disparity_grid`, `reliability_levels`, `ridge_alpha`,
`decoder_test_size`, `lesion_n_random`) is not a training parameter (§4.2).
`03_analyze.py --run flagship --config configs/flagship.yaml` takes the block
from that file, records it in `metrics.json` (`analysis_config`,
`analysis_config_source`) and writes it back into `results/flagship/config.yaml`;
`04_figures.py` and `06_implied_weight.py figures / reliability` then follow
the run's `config.yaml` by default (they accept `--config` too). `run_all.sh`
passes its config to stages 3 and 4 explicitly, so a fresh run is consistent
either way. After the grid change of 2026-10-03 the six runs need one such
pass each (the README's "Re-analysing without retraining" has the loop); the
sweep does not use the config grid and needs none.

The sweep, stage by stage:

```bash
python scripts/05_prior_sweep.py --priors 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
                                 --seeds 0 1 2 --control pcommon1
```

`--seeds` trains every (prior, seed) combination and collapses them to one row
per prior with across-seed SEMs; the figure picks the error bars up
automatically. Results are written after **every** seed, so an interrupted run
leaves usable partial output. `sweep.json` keeps both levels — `rows` is the raw
per-(prior, seed) records, `aggregated` is what the figure is drawn from.
Per-trial arrays go to `curves.npz` for the first seed only, so a curve can be
restyled or re-binned without retraining.

The whole family, in the order the rerun of 2026-10-02/03 used it (the
README's "Rerunning the flagship family" has the same list with its
explanations):

```bash
python scripts/00_calibrate.py --config configs/flagship.yaml
bash scripts/run_all.sh --config configs/pcommon1.yaml    # first: its residuals are sigma_out
bash scripts/run_all.sh --config configs/flagship.yaml
bash scripts/run_all.sh --config configs/pcommon0.yaml
bash scripts/run_all.sh --config configs/pcommon028.yaml
bash scripts/run_all.sh --config configs/pcommon03.yaml
bash scripts/run_all.sh --config configs/pcommon07.yaml
python scripts/06_implied_weight.py figures     --run flagship
python scripts/06_implied_weight.py reliability --run flagship --levels 5
python scripts/05_prior_sweep.py --priors 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
                                 --seeds 0 1 2 --control pcommon1
```

---

# Part 11 — Things to be careful about when reading these results

**One seed for everything except the sweep.** Every number in Parts 6, 7 and 8.1
is `seed: 0` and carries no run-to-run error bar. The prior sweep (§8.2) is the
exception: three seeds per prior, 27 networks, with across-seed SEMs. So the
transition midpoint, the position-regression slope, the reliability slope, the
variance hump and the model comparison *are* seed-checked; the unit-level
analyses, the behavioural curves and the calibration gate are not.

**The unit-level results changed sign or vanished between configurations.**
The congruent/opposite balance correlation is −0.37 to −0.73 on every causal
run of this family and was +0.50 / +0.54 / −0.05 on the wide one (§7.9); the
congruent lesion is within ±1 z of a random lesion on the flagship (+0.82)
and was +6.99 on the wide flagship. Neither is a finding, and the balance in
particular should not be presented in either sign. The truncation flag
(`conditioned_bias_negative_seen`) is True at 0.28, 0.3 and 0.7 and False at
0.5 — not the monotone pattern the mechanism predicts; the curve behind it is
consistently negative only at 0.3 (§7.10). All of these are single-run
observations the sweep does not measure.

**The flagship and the two low-prior satellites carry a calibration WARN**,
on the opposite check from before: 47–49 % intermediate posterior mass
against a 40 % ceiling, because the 10° prior keeps even agreeing cues from
being confident (§6.2). Nothing in Parts 7–8 is weakened by it; the
ambiguous zone the analyses work in is larger, not smaller. `pcommon07`
passes.

**The sparse outer bins of the sweep.** With hand positions from a 10° prior,
the sweep's outer bins at `p_common` = 0.8–0.9 hold 23–100 trials (the two
under the 25-trial floor are dropped) and its Panel A wobbles by ±0.1 there
(§9.5); read those points as what 30-trial means of a heavy-tailed ratio are.
The per-run figures no longer have this problem: since 2026-10-03 their grid
ends at ±30° and every bin holds at least 94 trials on every causal run
(§4.2). What the finer grid shows more clearly is a real asymmetry of the
flagship's hand output — 0.1–0.5° of pull toward vision from 10° to 30° on
the positive side only, where Bayes has none (§9.3, figure 12).

**The SIL already decodes the posterior at R² 0.89** (§7.8), so the
emergence argument rests on the always-fuse twin (0.23–0.37), not on a rise
across layers. Say it that way.

**The two controls' decoding R² of ≈0 is expected**, not a finding. Their
posterior is constant, so there is no variance to explain. And their
*higher* validation loss (Part 5) and lower `var` R² (§7.1) are the same
non-finding seen through a variance-normalised loss.

**The design document's §5 eye-transform formula disagrees with the code**, and
the code is the one that matches the exact posterior. The document is the thing
to amend.

**The consistent sub-optimality is real, and now small.** Position-regression
slopes of 0.88–0.99 with `slope + 1.96·SEM < 1` at seven of nine priors across
three seeds, a variance hump at 89–94 % of analytical from 0.3 to 0.8, a
weight-on-posterior slope of 0.97 on every run, and a top-decile weight
shortfall (0.79 against 0.91 in the model comparison) all point the same way: the network's implicit causal
inference is close to Bayesian and systematically a little conservative, most
so at the lowest priors. Since every pipeline guard is verified and both
controls are clean, this is attributable to the network. Per the design
document, that makes it a reportable finding rather than a failure — but on
this configuration it is a 2–5 % effect at the flagship prior, and the
paper should size it that way.

**The lesion result answers necessity, not representation**, and on the
flagship it answers "no subpopulation is necessary". Mean-clamping plus a
size-matched random baseline (§7.9) puts all three lesions within ±1 z of
random. A redundant code can make every subpopulation dispensable.

**The reliability slope is flat at 0.95 and is no longer a lead.** On the
wide configuration it declined with the prior (r = −0.87) and was flagged as
needing a control for the composition of the surviving bins; on this one it
reads 0.91–1.02 across the nine priors with no trend (§8.3). The composition
caveat still applies to the statistic, but there is no pattern left to
explain.
---

# Part 12 — What is a result, and what is not

This part exists because the guide documents **every** analysis in the codebase,
and only some of them are findings. Several are controls, several are method
verification, and several are leads that a reviewer would take apart. Putting a
control in the Results section as though it were a discovery is one of the
easier ways to lose a referee's confidence in everything around it.

Grades used below:

| grade | meaning | where it goes |
|---|---|---|
| **A** | headline finding — carries a claim, deserves a figure | Results, main figures |
| **B** | supporting result — real, but subordinate to an A | Results, a sentence or a supplementary panel |
| **C** | method verification or control | Methods, or a supplement. **Not** a finding |
| **D** | not a finding — negative, non-replicating, or measurement-limited | Omit, or state explicitly as a limitation |

## 12.0 Which stored numbers are current

**Every number in this guide is from the rerun of 2026-10-02/03.** The
flagship became the small-domain, narrow-range configuration on 2026-10-02
(Part 4); the results of the previous one (`flagship_wide.yaml`) were deleted
that day and the whole family — the six runs of Part 5, their twins,
`results/manuscript/`, the 27-network sweep and the `06` experiment on the
flagship — was regenerated from the current code, with the hybrid read (§7.2)
in every weight-based block and σ_out from the new `pcommon1` in every run.
There is no longer any stored number that predates the hybrid read or the
lesion fix, and no table in Parts 5–9 and 12 that describes `flagship_wide`
(its numbers survive in `CROSS_PRIOR_RESULT.md`, labelled as such, and in
`IMPLIED_WEIGHT_RESULT.md`, where the two configurations are compared).

**What is stored where.** `results/<run>/metrics.json` holds every scalar
quoted in Parts 5–7 (`implied_weight_vs_post`, `weight_regression_*`,
`hybrid_read`, `position_regression_*`, `variance_regression_*`,
`weight_consistency`, `reliability_within_disparity`, `variance_signature_*`,
`model_comparison_*`, `transition`, `conditioned_bias_negative_seen`,
`post_c1_decoding_r2`, `twin_post_c1_decoding_r2`, `congruency`,
`balance_vs_post`, `lesion`, `rf_shift_median_gain`); `analysis.npz` the
per-trial arrays behind them; `results/prior_sweep/sweep.json` the sweep's
`rows` and `aggregated`; `results/calibration/<config>/metrics.json` the gate.
The lesion blocks carry `mode: "mean"` and `n_random: 100`; the random null
is seeded, so re-running stage 3 reproduces the stored z-scores exactly, while
changing `lesion_n_random` does not. The zero-clamp comparison and the
per-output lesion z-scores that earlier versions of §7.9 tabulated were a
one-off run on the wide flagship and are not stored anywhere; §7.9 no longer
quotes them.

**What is derived here rather than stored.** A few numbers in this guide are
computed from the stored files rather than read from them: the `within`
rows and percentages of §7.6, the `|d|` ratios there, the midpoint
regressions and trend statistics of §8.2–8.3 (from `aggregated` and `rows`),
the per-bin counts of figures 04/12 and the sweep's Panel A (from the
per-trial arrays), and the posterior-mass fractions quoted for the sweep
priors (from `curves.npz`). Each is a one-line calculation on the named file.

Stage 3 does not retrain, so refreshing every stored number after a code or
analysis-block change is cheap (`--config` applies the live config's analysis
block to the run; §4.2, Part 10):

```bash
python scripts/03_analyze.py --run pcommon1 --twin pcommon1_twin --config configs/pcommon1.yaml
python scripts/04_figures.py --run pcommon1
for r in flagship pcommon0 pcommon028 pcommon03 pcommon07; do
  python scripts/03_analyze.py --run $r --twin ${r}_twin --control pcommon1 --config configs/$r.yaml
  python scripts/04_figures.py --run $r
done
python scripts/05_prior_sweep.py --replot
```

(`pcommon1` first, as always, so that its fresh `residual_std` is what the
others read.)

**The 2026-10-03 grid.** The `disparity_grid` of every live config changed on
2026-10-03 (§4.2). The change touches only the binned bias curves of
`analysis.npz`, the `conditioned_bias_negative_seen` flag (which reads the
same on both grids, §7.10) and figures 03/04/05/12 with manuscript Fig. 2A/C;
every other stored number is unaffected. The figure descriptions of §9.3 and
the curves of §7.10 are on the new grid, computed from the stored per-trial
arrays with the pipeline's own functions; the loop above (or the README's) is
what regenerates the stored arrays and the figure files to match, and a
`metrics.json` carrying `analysis_config` with the 19-centre grid is the sign
that a run has had it.

## 12.1 Grade A — the findings

**A1. The implied fusion weight depends on the prior exactly as Bayes requires.**
§8.2. Twenty-seven networks. Over all nine priors the network transition
midpoint regresses on the analytical midpoint with slope **1.016**, intercept
**−0.20°**, **R² = 0.9998**, mean deviation **0.13°** (max 0.22°) — with **no
free parameter relating the two axes** — and the sharpness of the transition
matches to within 4 % at every prior. This is the strongest result in the
project and should be the paper's central figure. On the wide configuration
the fit needed the 0.1 point excluded to reach R² 0.994; nothing is excluded
now (the 0.1 point is a negative midpoint on both axes, which the caption
must explain; §8.2).

Its force is in what it rules out: a disparity heuristic predicts a
prior-independent midpoint, a fixed-weight scheme predicts no transition at all,
and model selection predicts a step with sharpness → ∞. All three are excluded
by the same measurement.

**A2. Model averaging, not model selection.** §7.7 and §8.2. The five-way
comparison selects averaging in **27 of 27** sweep networks plus the flagship
and the three satellites, by 14–37 % over the runner-up from 0.3 to 0.8 and
6–14 % at 0.9; only at `p_common` = 0.1 do the strategies nearly coincide
(1–3 %). This is the
Körding et al. (2007) question asked of a network that was never given a
causal read-out, and it answers cleanly.

**A3. The variance output carries the between-component signature of causal
ambiguity.** §7.6. The `w(1−w)d²` term is the part of the mixture variance that
**no fixed-weight model can produce**, and the network reproduces it — at the
peak, more than a third of its reported uncertainty (2.34 of 6.37 deg²) is
causal uncertainty rather than measurement noise — at **93 % of its analytical
size** on the flagship, 89–94 % across the sweep from 0.3 to 0.8, and growing
linearly with the prior as it should. Present in both output channels and
every run. The flat fixed-weight control line makes the argument visually on
the figure.

The claim is now an *amplitude* result as well as an existence-and-shape one;
what remains weaker is the bottom of the prior range, where the ideal
observer has no hump either (D4).

**A4. The posterior is emergent, not imposed.** §7.8. The trial-wise posterior
decodes linearly from the multisensory layer at R² = 0.97–0.98 (and from the
single-modality layer at 0.89 — on this domain the disparity is close to a
linear function of the inputs, so the rise across layers is no longer the
argument), while the always-fuse twin — same architecture, same inputs, same
training, fused targets only — reaches just 0.23–0.37 at either layer. The
twin is what turns this from an observation into a controlled contrast, and
it is the whole of the argument now: report the twin alongside, not the
layer-to-layer rise.

**A5. A small, consistent conservatism.** §7.3 and §8.2. Position-regression
slopes run 0.884–0.988 across the nine priors and `slope + 1.96·SEM < 1` at
**seven of the nine, across three seeds** (0.6 and 0.9 reach 1.002 and
1.007); the slope rises with the prior (r = 0.91). The network applies
slightly less pull toward the fused solution than the ideal observer — by
12 % at `p_common` = 0.1, by 2–8 % elsewhere.

**Frame this as a finding, not an apology — and size it.** It is a
reproducible property of a network trained on exactly Bayesian targets, it
is visible in four independent measures (position slope, variance-hump
amplitude, weight-on-posterior slope of 0.97, the top-decile weight
shortfall), and every pipeline guard and both controls are clean. On the
wide configuration it was a 10–25 % effect; here it is a few per cent at the
flagship prior and largest where fusion is rarely right. A referee will ask
whether it is a bug; the answer is documented in `IMPLIED_WEIGHT_RESULT.md`
(it shrinks with the input domain relative to the noise, not with network
size) and it is not.

## 12.2 Grade B — supporting results

**B1. The disparity-heuristic alternative is excluded directly.** §7.5 and
§8.3. Within-disparity-bin slopes of 0.89–0.98 on the four runs and
0.91–1.02 across the sweep (mean 0.955, no trend with the prior), far from
the 0 a heuristic predicts and close to the 1 Bayes predicts, on 6–7 of 8
bins per run. Supporting rather than headline only because A1 and A2 rule
out the same alternative on more data; on this configuration it could carry
a sentence of its own ("at matched disparity the network uses cue reliability
at 95 % of the Bayesian rate").

**B2. Opposite units are not necessary for the read-out.** §7.9, with
mean-clamping and a size-matched random baseline. This is the one lesion
statement that holds on every run (opposite z −3.0 to +1.6, never worse than
random). The companion statement — that congruent units *are* necessary —
held at z = +6.99 on the wide flagship and holds at z = 2.1–2.9 on four runs
of this family, but on the flagship itself it is z = +0.82 and on
`pcommon028` +0.12; do not make it. State the scope precisely: this is a
claim about **necessity**, not about representation, and a redundant code
can make any subpopulation dispensable — which on the flagship is what every
lesion says.

**B3. The reference-frame transformation is complete by the multisensory
layer.** §7.9, median RF shift gain −0.087 on the flagship, within ±0.1 of
zero on every run (against +1 for a retinal code). This is the continuity
check with Farahmandi et al. — it says the present network reproduces the
previous result before adding anything causal, which is what licenses the
comparison.

**B4. Congruent/opposite units emerge at all, and more of them when the task
needs them.** §7.9: 15 congruent / 21 opposite / 28 mixed on the flagship,
31 opposite at the two low priors, against 42 congruent / 11 opposite on the
always-fuse control. Descriptive, and the link to Rideaux et al. (2021). The
*balance* analysis built on it is D1 — the classification is fine, the
correlation is not.

**B5. The Körding Fig. 2e bias curve is reproduced.** §7.10 and §9.3, a
non-monotonic pull that peaks at ±1.1–1.2° at ±4–6° disparity and collapses
by ±10°, on top of the Bayesian curve through the transition. A recognisable
qualitative signature; it is not independent evidence, since it is the same
behaviour A1 measures quantitatively. State the one blemish if the panel is
shown: 0.1–0.5° of residual pull toward vision on the positive side from 10°
to 30°, where Bayes has none and the negative side shows none.

**B6. Hand and visual channels imply the same weight.** §7.4, mean absolute
difference 0.10–0.13 on a quantity bounded in [0, 1], correlation 0.68–0.85,
on every trial of every run. Both numbers can now be quoted (D5 is retired).

**B7. The variance channel tracks its target as a regression.** §7.3, the
variance-domain regression (figure 08v A): slopes 0.956–0.981 with SE 0.002
and R² 0.94–0.97 on every run. A one-sentence companion to the position
regression, from the other output.

## 12.3 Grade C — method verification and controls

None of these is a finding. They belong in Methods or a supplement, and they
matter — a referee who wants to know whether the pipeline is sound is asking for
exactly this list. But writing them up as discoveries reads as padding.

| item | § | what it establishes |
|---|---|---|
| the eight-check calibration gate | Part 6 | the dataset can support the analyses at all; its one WARN (47–49 % intermediate posterior mass) is a property of the 10° prior and is explained there |
| read-out accuracy, R² 0.994–0.999 on the positions, 0.91–0.97 on the variances | §7.1 | the network learned the task; not a result about causal inference |
| σ_out measured on the `p_common` = 1 control (0.39° / 0.014 deg²) | §7.2 | the noise scale is measured where the target is single-valued |
| the always-fuse twin | §7.8 | the contrast that makes A4 a controlled claim |
| `pcommon0` / `pcommon1` decoding R² ≈ 0 | §7.8 | **expected** — the posterior is constant, so there is no variance to explain |
| Poisson validity, range containment, anti-confound AUCs | Part 6 | the encoders behave as specified (3.2 % inflation, 0.0 % rejection, AUC within 0.02 of chance) and no confound is available |
| the `transition_fit` range guard | §7.11 | a failed fit returns NaN instead of a fabricated midpoint |
| the hybrid read of the weight | §7.2 | one weight per trial with no filter: the variance root where a ratio is blind, the ratio where the variance is ambiguous |
| the `pcommon1` target-reduction check | §7.2 | with the prior at 1 the mixture target equals the fused one (max difference 10⁻⁸); the targets are what the derivation says |

The hybrid read and the fit guard are worth a Methods sentence each. They are
the kind of detail that pre-empts a reviewer question rather than inviting one.

## 12.4 Grade D — do not present these as findings

**D1. The congruent-minus-opposite balance correlation.** §7.9. −0.44, −0.37,
−0.37 and −0.73 on the four causal runs of this family; +0.54, +0.50 and
−0.05 on the wide one. It replicates within a configuration and reverses
between them, so it is a property of the statistic — a difference of mean
sigmoid activations — not of the computation. Omit it. If the unit-level
question matters, ask it with a decoder from each subpopulation or a per-unit
correlation, and run it across the sweep's 27 networks.

**D2. The truncation / negative-bias effect.** §7.10,
`conditioned_bias_negative_seen`: True at 0.28, 0.3 and 0.7, False at 0.5 — not
the monotone-in-prior pattern the mechanism predicts, and on the wide
configuration it was True only at 0.28. The flag is a boolean and should not
be quoted. The curves behind it are more interesting than they were: on the
2026-10-03 grid the `pcommon03` network's inferred-two-causes branch is
negative in every bin from 2° to 30° (−0.04 to −0.12°, 200–760 trials per
bin), `pcommon07`'s in five of eight, `pcommon028`'s in two, the flagship's
in none. The direction is theoretically sensible, which makes it tempting
and makes it worse: one network per prior, and the sweep does not measure
it. A lead for future work, to be pursued on the conditioned curves across
seeds, not on the flag.

**D3. The congruent lesion as a positive claim.** §7.9. z = +6.99 on the wide
flagship, +0.82 on this one (+0.12 at 0.28, +2.1–2.9 at 0.3, 0.7 and on the
controls). Whatever the congruent units are for, a result that moves from
seven standard deviations to under one between two configurations of the
same model is not one to build on. B2 keeps the negative half (opposite units
are never necessary), which does hold everywhere.

**D4. The variance hump at `p_common` = 0.1 and 0.2.** §8.3. There the
*analytical* hump is zero (−0.02, +0.06 deg²) — the ideal observer has no
mid-ambiguity elevation to reproduce, because the confident side of the
statistic is almost all segregated, high-variance trials — so the network's
−0.09 and −0.01 are agreement, and the ratio is undefined. From 0.3 to 0.9
the hump is an A3 result (ratio 0.83–0.94). Show the supplementary panel over
[0.3, 0.9] with the trial counts and say in one sentence why the two lowest
priors are blank. (The collapse to 0.28 at 0.9 that this item described on
the wide configuration is gone.)

**D5. Retired.** The hand-vs-visual consistency correlation was ≈ 0.11 on the
filtered ratio of the wide configuration (0.35–0.47 once that configuration
was read by the hybrid) and is 0.68–0.85 on the hybrid read of this one
(§7.4). Quote it with the
MAD (B6).

**D6. The `p_common` = 0.1 transition midpoint, read literally.** §8.2. It is
−1.11° for the network and −0.94° analytical: a negative "disparity at which
fusion gives way to segregation", because at that prior the mean posterior at
zero disparity is 0.34 and the logistic crosses 0.5 only by extrapolation.
The network's extrapolation matches the ideal observer's, which is a real
result; the number is not a disparity any trial has. Show the point with its
error bar (seed spread −0.85 to −1.30, the largest in the sweep), explain it
in the caption, and do not describe it as a transition.

**D7. Any lesion claim about causal behaviour specifically.** §7.9 measures
lesion effects on read-out RMSE. It does not establish that any subpopulation
is necessary for the *causal* computation as distinct from the position
estimate, and on the flagship it does not establish that any subpopulation is
necessary at all. Do not extend B2 into a claim about causal inference.

**D8. The SIL-to-MSL rise in posterior decodability.** §7.8. 0.89 → 0.97 on
this configuration (0.27–0.48 → 0.93–0.96 on the wide one). The rise is real
but small, and its size depends on how linear the posterior is in the inputs,
which the domain sets. The emergence claim is the twin contrast (A4), not the
rise.

## 12.5 The shortest honest version of the paper's claims

If the Results section had to be four sentences:

> A feedforward network trained only to report position and uncertainty, with no
> causal read-out, develops an implicit fusion-vs-segregation trade-off that
> tracks the Bayesian posterior over causal structure quantitatively across nine
> priors and three seeds (midpoint slope 1.02, R² 0.9998, mean deviation
> 0.13°, no free parameters), with a small, prior-dependent conservatism of a
> few per cent. A five-way model comparison selects model averaging over model
> selection, full integration, full segregation and the best fixed-weight
> observer in all 27 networks. The network's uncertainty output carries the
> between-component variance term that no fixed-weight scheme can produce, at
> 89–94 % of its Bayesian amplitude. The posterior itself is linearly decodable
> from the network's hidden layers (R² 0.97), but not from an
> identically-structured network trained on fused targets alone (R² ≤ 0.37).

Everything in 12.2 supports those four sentences. Everything in 12.3 belongs in
Methods. Everything in 12.4 belongs in a limitations paragraph or nowhere.
