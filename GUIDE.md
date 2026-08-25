# `causal_integration` — Complete Guide

**What this document is.** A walkthrough of the codebase, every config, every
analysis, and every figure — with the calculation behind each one and the number
your own runs actually produced.

**Provenance.** Every number below was read out of files in your repo:
`results/<run>/metrics.json` and `results/calibration/<config>/metrics.json`,
as they stood after your `make all` runs. Nothing here is estimated, remembered,
or carried over from anywhere else. Where a result is absent or degenerate I say
so rather than filling the gap.

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

Asking for the visual source *in spatial coordinates* is deliberate: it forces
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

Five stages. Each writes files, so any stage can be re-run alone.

```
stage 0   00_calibrate.py     checks the CONFIG is usable      -> results/calibration/<config>/
stage 1   01_generate_data.py samples trials, encodes inputs   -> data/<name>.npz
stage 2   02_train.py         trains the network               -> results/<run>/model.pt
stage 3   03_analyze.py       computes every number            -> results/<run>/metrics.json + analysis.npz
stage 4   04_figures.py       draws every figure               -> results/<run>/figures/
```

`run_all.sh` (= `make all`) runs all five and also builds the always-fuse twin.
`make quick` does the same with 8,000 trials and 60 epochs.

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
    causal.py       the causal-inference analyses (the bulk of stage 3)
    units.py        unit-level analyses of the hidden layers
    behavior.py     Körding-style behavioural curves
    decoding.py     what the hidden layers carry
    todo.py         two analyses not yet designed
  viz/
    inputs.py       what the network is shown
    training.py     did it converge
    results.py      network vs observer
    style.py        shared figure defaults
  utils/
    config.py  paths.py  seed.py  io.py
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

*(This is where the code and the design document disagree. The document
specifies the plain sum. Numerical integration of the full posterior over
(C, source, eye) shows the code's form is the exact ideal observer to ~1e−13,
and the plain sum deviates from it by up to 0.076 in p(C=1|x). The code is
right; the document is the thing to amend.)*

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

`post_c1` is the trial-wise posterior **p(C=1|x)**. The config key `p_common` is
the **prior**. They are different quantities and the code now names them
differently — the dataset key used to be called `p_common` too, which is exactly
the confusion to avoid.

**(d) Model averaging** (`model_average`) — never hard selection:

```
mu  = w*fused_mu + (1-w)*seg_mu
var = w*(fused_var + fused_mu^2) + (1-w)*(seg_var + seg_mu^2) - mu^2
```

with `w = post_c1`. The variance is the **full mixture variance** (law of total
variance), not a weighted average of the two variances. That distinction is the
whole of the §7.3 analysis: the extra term is `w(1−w)(fused_mu − seg_mu)²`,
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

The reweighting is not cosmetic. The means live on ±20 deg and the variances on
0–20 deg²; without it the large-scale columns dominate the gradient and the
shared trunk starves `mu_prop`.

**Training** (`training.py`): Adam, lr 0.001, batch 256, up to 300 epochs, early
stopping on validation loss with patience 20. The best-validation weights are
restored at the end. The split is saved in the checkpoint so every later
analysis runs on exactly the trials the model was validated against.

---

# Part 4 — The configs

Ten files in `configs/`. Five are **live** (the flagship, its two controls, two
satellites); four are **legacy** and now fail the stage-0 gate; one is a
satellite you have not run yet.

## 4.1 The live set

| config | prior `p_common` | role |
|---|---|---|
| `flagship.yaml` | 0.5 | the main network |
| `pcommon1.yaml` | 1.0 | always-one-cause control (§9.1) |
| `pcommon0.yaml` | 0.0 | always-two-causes control (§9.2) |
| `pcommon03.yaml` | 0.3 | satellite — **not yet run** |
| `pcommon07.yaml` | 0.7 | satellite |
| `pcommon028.yaml` | 0.28 | satellite, matching Körding's human prior fit |

Every one of these is `flagship.yaml` with **only the Bernoulli constant
changed**. That is what makes cross-prior comparison meaningful: same seed, same
ranges, same encoders, so any difference in behaviour is attributable to the
prior and nothing else.

## 4.2 What `flagship.yaml` contains

```yaml
seed: 0

generative:
  p_common: 0.5
  mu0: 0.0                        # prior mean on hand position
  sigma0_sq: 425.0                # prior variance -> sd 20.6 deg
  eye_mu: 0.0
  eye_sigma_sq: 325.0             # eye position sd 18.0 deg
  sigma2_vis_range:  [1.2, 6.6]   # per-trial visual noise,  sd 1.1-2.6 deg
  sigma2_prop_range: [1.8, 9.6]   # per-trial prop noise,    sd 1.3-3.1 deg
  sigma2_eye_range:  [3.3, 13.2]  # per-trial eye noise,     sd 1.8-3.6 deg

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
  disparity_grid: [-40,-30,-20,-15,-10,-5,-2,0,2,5,10,15,20,30,40]
  reliability_levels: [1.5, 3.5, 6.0]
  min_separation: 1.0
  sigma_w_criterion: 0.1
  ridge_alpha: 1.0
  decoder_test_size: 0.25
```

### Why these numbers and not the previous ones

`flagship.yaml` was derived from `matlab_match.yaml` by running the stage-0 gate
and fixing what failed:

| parameter | was | is | why |
|---|---|---|---|
| sensory σ² | as in `matlab_match` | **× 0.30** | `matlab_match` put 56% of trials at intermediate posteriors; the target is ≈25%. Tightening the sensory noise was chosen over widening `sigma0_sq` because it leaves the retinal spread — and therefore the containment rate — untouched. |
| `gain_K` | 50 | **180** | Poisson validity (below) |
| `rf_width` | 10 | **6** | Poisson validity |
| `n_vis` | 50 | **76** | Poisson validity; also equalises the three group sizes |
| `visual_field` | ±75 | **±85** | containment: 4.6% of trials fell outside the usable span, now 0.8% |

## 4.3 The legacy configs

`default.yaml`, `realistic.yaml`, `matlab_match.yaml`, `equal_n.yaml` are kept
for reference and comparison. All four **fail** the stage-0 Poisson check, and
three of them also warn on the posterior histogram. They are not deleted because
they document where the project came from, but they should not be used for new
runs.

---

# Part 5 — What you currently have in `results/`

| run | prior | dataset | test trials | best val loss @ epoch |
|---|---|---|---|---|
| `flagship` | 0.5 | flagship | 7500 | 0.2158 @ 298 |
| `pcommon07` | 0.7 | pcommon07 | 7500 | 0.2429 @ 294 |
| `pcommon028` | 0.28 | pcommon028 | 7500 | 0.1571 @ 271 |
| `pcommon1` | 1.0 | pcommon1 | 7498 | 0.0066 @ 202 |
| `pcommon0` | 0.0 | pcommon0 | 7500 | 0.0146 @ 220 |

Plus an always-fuse **twin** for each (`*_twin`), and five calibration folders
under `results/calibration/`.

Two things to notice in that table. The two controls reach far lower loss than
the flagship — that is expected, because with the prior pinned at 0 or 1 there
is no causal inference left to do; the task collapses to a fixed linear
combination. And `pcommon028` reaches lower loss than `flagship`, because at a
low prior most trials are confidently segregated and therefore easy.

**Not yet run:** `pcommon03` (the 0.3 satellite). The design lists it alongside
0.7; running it would give you four points on the prior axis instead of three.

---

# Part 6 — Stage 0, the calibration gate

Stage 0 draws a fresh dataset from the config (20,000 trials by default) and
runs eight checks. It writes `results/calibration/<config>/metrics.json` and
four figures.

## 6.1 The eight checks, and what each computes

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

Read from `results/calibration/*/metrics.json`, 20,000 trials each.

| | flagship | pcommon028 | pcommon07 | pcommon1 | pcommon0 |
|---|---|---|---|---|---|
| prior | 0.5 | 0.28 | 0.7 | 1.0 | 0.0 |
| realized C=1 | 0.4966 | 0.2734 | 0.6976 | 1.0000 | 0.0000 |
| **intermediate posterior mass** | **0.246** | 0.391 | **0.103** ⚠ | 0.000 | 0.000 |
| confident posterior mass | 0.363 | 0.510 | 0.343 | 1.000 | 1.000 |
| coverage (mean range) | 0.373 | 0.402 | 0.315 | 0.000 | 0.000 |
| median \|Δ\| visual | 3.91 | 7.80 | 2.66 | 1.84 | 12.82 |
| median \|Δ\| hand | 1.87 | 3.43 | 1.24 | 0.82 | 5.69 |
| frac \|Δ_vis\| > 2 | 0.692 | 0.792 | 0.600 | 0.463 | 0.917 |
| rejection rate | 0.82% | 0.83% | 0.83% | 0.87% | 0.76% |
| Poisson inflation | 3.1% | 3.1% | 3.1% | 3.1% | 3.1% |
| worst anti-confound AUC dev | 0.0109 | 0.0192 | 0.0051 | n/a | n/a |
| **verdict** | **8 PASS** | **8 PASS** | 7 PASS **1 WARN** | 2 PASS | 2 PASS |

Notes on reading this table:

- **`pcommon07` carries one WARN**: only 10.3% of its trials land at intermediate
  posteriors, below the 15% floor. This is not a bug — it is what a prior of 0.7
  does. Pushing the prior toward "one cause" makes most trials confidently
  fused, so the ambiguous zone shrinks. It means the §7.2 and §7.3 analyses have
  fewer trials to work with there, and its numbers deserve slightly more caution
  than the flagship's.
- **The two controls run only 2 checks.** With the prior at 0 or 1 the posterior
  is constant, so the posterior-histogram, coverage, Δ and anti-confound checks
  have nothing to test and are skipped. Containment and Poisson still run.
- **The Poisson inflation is identical (3.1%) across all five.** It should be —
  the encoding parameters are the same in every config; only the Bernoulli
  constant differs.
- **Median |Δ| rises as the prior falls** (12.82 → 7.80 → 3.91 → 2.66 → 1.84 as
  the prior goes 0 → 0.28 → 0.5 → 0.7 → 1.0). At a low prior the observer
  segregates more often, so the segregated and fused solutions sit further
  apart.

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
(per-trial arrays for the figures).

## 7.1 Read-out accuracy

For each of the four outputs: slope and intercept of a least-squares fit of
network against analytical, R², RMSE, MAE, and mean bias.

**Your results.**

| run | mu_vis R² | var_vis R² | mu_prop R² | var_prop R² |
|---|---|---|---|---|
| flagship | 0.9921 | 0.8764 | 0.9944 | 0.9388 |
| pcommon028 | 0.9935 | 0.8746 | 0.9953 | 0.9584 |
| pcommon07 | 0.9918 | 0.8405 | 0.9932 | 0.9182 |
| pcommon1 | 0.9990 | 0.9975 | 0.9990 | 0.9975 |
| pcommon0 | 0.9982 | 0.9909 | 0.9987 | 0.9976 |

Slopes are 0.87–1.00 and biases are within ±0.14 deg everywhere.

**How to read it.** The mean channels are near-perfect in every run. The
variance channels are consistently the harder ones in the three causal-inference
runs (0.84–0.96) and near-perfect in the two controls (0.99). That difference is
informative rather than worrying: with the prior pinned, the variance target is
a smooth function of the reliabilities alone, whereas at an intermediate prior it
carries the mixture term, which is a sharply non-linear function of the
disparity.

## 7.2 σ_out and the σ_w filter

```
residual_std = std(network - analytical), per output
sigma_w      = sigma_out / |Delta|,       per trial
```

`sigma_out` is meaningful only when measured on the **p_common = 1 control**,
where the target is single-valued so residuals are pure read-out noise. On the
flagship the residuals also contain any causal-inference misweighting.

Stage 3 takes σ_out from `--control pcommon1` when given, and your flagship,
pcommon07, pcommon028 and pcommon0 runs all record `sigma_out_source:
"pcommon1"`, so they used the real control.

**Your σ_out** (from `results/pcommon1/metrics.json`):

```
mu_vis 0.6275    var_vis 0.0551    mu_prop 0.6281    var_prop 0.0552  (deg, deg^2)
```

The two mean channels agreeing to three decimals is a good sign — under C = 1
they are the same quantity, and the network treats them as such.

**What σ_w is for.** A trial where the two hypotheses nearly coincide cannot
reveal the weight: dividing by a near-zero Δ turns read-out noise into an
enormous apparent error. σ_w quantifies that per trial, and the default
criterion (`sigma_w_criterion: 0.1`) keeps only trials where the weight is
readable to better than 0.1.

## 7.3 The position-domain regression — the headline

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

| run | prior | visual slope [CI] | visual intercept | hand slope [CI] | hand intercept |
|---|---|---|---|---|---|
| pcommon028 | 0.28 | 0.797 [0.761, 0.833] | −0.041 | 0.931 [0.867, 0.995] | −0.078 |
| flagship | 0.5 | 0.856 [0.830, 0.881] | −0.007 | 0.858 [0.815, 0.901] | +0.061 |
| pcommon07 | 0.7 | 0.927 [0.907, 0.947] | +0.100 | 0.883 [0.847, 0.920] | +0.101 |

**How to read it.** Intercepts are essentially zero everywhere (|intercept| ≤
0.10 deg, against outputs that range over tens of degrees). Slopes sit at
0.80–0.93 — below 1, and the CIs exclude 1 in every case. The network
consistently applies *slightly less* pull toward the fused solution than the
ideal observer would.

This is a real, reproducible finding rather than a pipeline artefact: every
guard has been verified, both controls are clean, and the deviation appears at
three different priors. It says the network's implicit causal inference is
close to Bayesian but systematically a little conservative.

## 7.4 The joint weight and hand-vs-visual consistency

Each trial gives two equations with one common `w`:

```
w = [ Dv*(est_vis - seg_vis) + Dp*(est_prop - seg_prop) ] / ( Dv^2 + Dp^2 )
```

`weight_consistency` then compares the weight read off the visual output with
the one read off the hand output, on trials where **both** are readable
(σ_w < 0.1 on both).

**Your results.**

| run | n usable | correlation | mean abs difference |
|---|---|---|---|
| flagship | 1732 | 0.124 | 0.141 |
| pcommon07 | 1042 | 0.108 | 0.181 |
| pcommon028 | 2476 | 0.101 | 0.119 |

**How to read it — and this is the one number in the whole set that is easy to
misread.** The correlation looks poor. It is not measuring what it appears to.
The σ_w filter keeps high-|Δ| trials, and those are overwhelmingly
confident-segregation trials where the true weight is near zero for all of them.
With almost no variance in the underlying quantity, a correlation has nothing to
correlate — it is dominated by noise even if both readings are accurate.

**The mean absolute difference is the informative statistic**, and 0.12–0.18 on
a quantity bounded in [0, 1] is reasonable agreement. If you want a stronger
version of this test, restrict it to trials with intermediate posterior *and*
large Δ, where the weight actually varies; that subset is small, which is itself
the reason this test is weaker than the position regression.

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

**Your results.**

| run | prior | combined slope | 95% CI | bins kept | bins skipped |
|---|---|---|---|---|---|
| pcommon028 | 0.28 | **0.906 ± 0.093** | [0.724, 1.088] | 2 | 6 |
| flagship | 0.5 | **0.567 ± 0.073** | [0.424, 0.710] | 3 | 5 |
| pcommon07 | 0.7 | **0.712 ± 0.072** | [0.572, 0.853] | 3 | 5 |

The flagship's per-bin detail:

| \|disparity\| bin | n | spread of post_c1 | slope ± SE |
|---|---|---|---|
| 6.9 – 12.2 | 364 | 0.931 | +0.477 ± 0.079 |
| 12.2 – 16.4 | 363 | 0.456 | +1.071 ± 0.195 |
| 16.4 – 20.3 | 364 | 0.070 | +3.258 ± 1.503 |

and the five skipped bins, with the spread that got them dropped:

```
|d| 20.3-25.4   spread 2.5e-03
|d| 25.4-30.9   spread 3.6e-05
|d| 30.9-38.5   spread 4.3e-08
|d| 38.5-49.2   spread 1.5e-12
|d| 49.2-102.3  spread 2.9e-19
```

**How to read it.** Every run is far from 0, so the disparity-heuristic
explanation is ruled out — the network *is* using reliability. Two of the three
are also below 1, consistent with the position-regression slope: reliability is
tracked, but under-tracked.

The skipped bins show why this test can only ever use the near-in bins: past
about 20 deg of disparity the posterior is numerically pinned at zero for every
trial regardless of reliability, so there is nothing left to test. That is a
property of the world, not a limitation of the code.

The third kept bin (spread 0.070, slope 3.26 ± 1.50) is barely above the
exclusion threshold and contributes almost nothing to the inverse-variance
weighted combination — the first two bins carry the result.

## 7.6 The variance signature of causal ambiguity

This is the analysis that covers the zone where weight recovery is blind.

The mixture variance decomposes as

```
Var = w*fused_var + (1-w)*seg_var        <- the within-component part
    + w*(1-w)*(fused_mu - seg_mu)^2      <- the BETWEEN-component part
```

The second term is zero when `w` is 0 or 1 and maximal at intermediate `w` with
separated means. **No fixed-weight model can produce it** — that is what makes
it diagnostic.

The analysis bins trials by `post_c1` into 10 bins and reports, per bin: the
network's mean Var output, the analytical mixture variance, the
between-component term alone, and the best fixed-weight prediction
(`w̄·fused_var + (1−w̄)·seg_var` with `w̄` the mean posterior), which is flat by
construction.

The scalar summary is the **hump**: mean variance in the intermediate bins
(0.2 < centre < 0.8) minus mean variance in the confident bins.

**Your flagship, `var_vis`, per bin** (from `metrics.json`):

| posterior bin | network | analytical mixture | between term | fixed-weight | n |
|---|---|---|---|---|---|
| 0.05 | 11.89 | 11.82 | 0.41 | 7.54 | 2588 |
| 0.15 | 16.38 | 18.07 | 7.30 | 7.89 | 137 |
| 0.25 | 16.27 | 18.89 | 8.91 | 7.93 | 122 |
| 0.35 | 16.08 | 18.73 | 9.46 | 7.97 | 139 |
| 0.45 | 13.88 | 15.99 | 7.96 | 7.64 | 148 |
| 0.55 | 13.15 | 14.08 | 6.62 | 7.92 | 208 |
| 0.65 | 10.63 | 10.80 | 4.22 | 7.83 | 346 |
| 0.75 | 7.81 | 7.67 | 1.89 | 7.99 | 883 |
| 0.85 | 5.48 | 5.25 | 0.40 | 7.59 | 2233 |
| 0.95 | 4.54 | 4.08 | 0.21 | 7.28 | 696 |

**Hump values across your runs:**

| run | prior | var_vis network | var_vis analytical | var_prop network | var_prop analytical |
|---|---|---|---|---|---|
| flagship | 0.5 | **+3.40** | +4.55 | +0.94 | +1.24 |
| pcommon07 | 0.7 | **+4.58** | +7.07 | +1.09 | +1.59 |
| pcommon028 | 0.28 | **+1.15** | +1.89 | +0.29 | +0.49 |

**How to read it.** The hump is present in every run and in both output
channels. The network reproduces it at roughly 60–75% of its analytical
magnitude — the same conservatism the position regression reports, showing up in
the uncertainty channel.

The fixed-weight column is the control: it sits flat at 7.3–8.0 across every
bin, confirming that no constant weight can generate this shape.

One detail in the table worth noticing: the network slightly *over*-reports
variance in the most confident bins (4.54 vs 4.08 at centre 0.95) and
*under*-reports in the ambiguous ones. That is compression toward the mean, the
signature of a read-out that is smoothing a sharply peaked target.

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

| run | averaging | selection | fixed | segregation | integration | best |
|---|---|---|---|---|---|---|
| flagship | **1.797** | 2.047 | 2.197 | 2.244 | 13.755 | averaging |
| pcommon07 | **1.837** | 2.045 | 2.558 | 2.652 | 10.864 | averaging |
| pcommon028 | **1.633** | 1.863 | 1.802 | 1.812 | 16.632 | averaging |

Same verdict on the hand output in all three runs.

**Flagship per-decile implied weight:**

| decile centre | network | averaging | selection | fixed |
|---|---|---|---|---|
| 0.000 | 0.017 | 0.000 | 0.000 | 0.032 |
| 0.000 | 0.011 | 0.000 | 0.000 | 0.032 |
| 0.001 | −0.005 | 0.001 | 0.000 | 0.032 |
| 0.157 | 0.182 | 0.119 | 0.000 | 0.032 |
| 0.594 | 0.539 | 0.574 | 0.754 | 0.032 |
| 0.761 | 0.673 | 0.757 | 1.000 | 0.032 |
| 0.814 | 0.710 | 0.812 | 1.000 | 0.032 |
| 0.842 | 0.619 | 0.841 | 1.000 | 0.032 |
| 0.873 | 0.617 | 0.875 | 1.000 | 0.032 |
| 0.936 | 0.505 | 0.938 | 1.000 | 0.032 |

**How to read it.** Averaging wins on RMSE in every run and both outputs, and
integration loses catastrophically (13.8 vs 1.8), which is the sanity check that
the network is not simply always fusing.

In the per-decile table the network tracks averaging closely up to about the
0.75 decile and then falls away — 0.505 where averaging predicts 0.938. That
top-decile shortfall is the same under-fusion the position regression measures,
and it is where most of the slope deficit comes from.

The decile centres bunch (three at ≈0.000, then a jump to 0.157, then 0.594+)
because the posterior distribution is strongly bimodal — deciles of a bimodal
variable are not evenly spaced. This is honest, not a plotting bug.

## 7.8 What the hidden layers carry

Ridge regression (`alpha = 1.0`) from each hidden layer's activations to a
target, scored by held-out R² on a 25% split.

The target is `post_c1`, the **trial-wise posterior** — never the prior. The
prior is one number per network; "decoding" it would just read out a disparity
confound.

**Your results:**

| run | SIL (layer0) | MSL (layer1) | twin SIL | twin MSL |
|---|---|---|---|---|
| flagship | 0.482 | **0.958** | 0.082 | 0.189 |
| pcommon07 | 0.436 | **0.948** | 0.064 | 0.186 |
| pcommon028 | 0.274 | **0.927** | 0.029 | 0.208 |
| pcommon1 | −0.100 | −0.100 | −0.100 | −0.100 |
| pcommon0 | −0.010 | 0.107 | 0.008 | 0.062 |

**How to read it.**

- **The rise from SIL to MSL** (0.27–0.48 → 0.93–0.96) is the emergence claim:
  the posterior is not handed to the network by the input encoding, it is built
  across the layers.
- **The always-fuse twin reaches only 0.19–0.21.** The twin has the same
  architecture and the same inputs but was trained on fused targets only, so it
  was never asked for anything causal. That it carries so little of the posterior
  is the emergent-vs-imposed contrast.
- **The two controls decode at ≈0.** This is expected and not a failure: with
  the prior at 1.0 the posterior is *constant* (every trial has p = 1), so there
  is no variance to explain and R² is undefined-ish, coming out slightly
  negative. Same at 0.0. These rows are a consistency check, not a result.

`mu_vis` also decodes from MSL at R² = 0.993 (flagship), confirming the layer
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
| flagship | 17 | 24 | 21 | 2 |
| pcommon07 | 14 | 29 | 21 | 0 |
| pcommon028 | 21 | 20 | 17 | 6 |
| pcommon1 | **36** | **6** | 22 | 0 |
| pcommon0 | 32 | 17 | 15 | 0 |

**How to read it.** The three causal-inference runs produce a roughly balanced
population. The p_common = 1 control produces 36 congruent against only 6
opposite — which makes sense: a network that always fuses has no use for
disagreement detectors. That contrast is a useful piece of evidence that the
opposite units in the flagship exist *because* of the causal task.

### Congruent-minus-opposite balance

Per trial: mean activity of congruent units minus mean activity of opposite
units. Correlated with `post_c1`.

**Your results:**

| run | correlation | R² of a 1-D linear read-out |
|---|---|---|
| flagship | **0.500** | 0.250 |
| pcommon028 | **0.538** | 0.289 |
| pcommon07 | **−0.053** | 0.003 |
| pcommon1 | −0.002 | 0.000 |
| pcommon0 | 0.050 | 0.003 |

**How to read it — this one does not replicate cleanly.** Two of the three
causal runs show a solid correlation (0.50, 0.54); `pcommon07` shows nothing
(−0.05). The two controls correctly show nothing, as they should, since their
posterior is constant.

I would not present the balance result as established on this evidence. The
`pcommon07` null is a genuine inconsistency and needs explaining before the
claim is made. One candidate: `pcommon07` is the run that failed the
intermediate-posterior check (10.3%), so it has the fewest ambiguous trials for
the balance to track. That is testable — re-run 0.7 with a config tuned to
restore intermediate mass, or check whether the correlation appears when
restricted to its ambiguous trials.

### Lesion

Zeros a subpopulation's MSL activations before the read-out and re-measures
overall RMSE.

**Your results (RMSE, all four outputs pooled):**

| run | intact | no congruent | no opposite | no mixed |
|---|---|---|---|---|
| flagship | 1.480 | 10.206 | 7.545 | 8.882 |
| pcommon07 | 1.607 | 8.912 | 7.977 | 7.194 |
| pcommon028 | 1.313 | 9.765 | 7.481 | 6.552 |
| pcommon1 | 0.446 | 12.787 | 4.884 | 2.993 |
| pcommon0 | 0.586 | 11.742 | 8.305 | 5.417 |

**How to read it, carefully.** Every lesion is catastrophic — 5× to 20× the
intact error. That is *not* evidence that each subpopulation is specifically
important. Zeroing roughly a third of a densely-connected layer's activations
destroys the read-out's operating point regardless of which third you pick; the
remaining units' biases no longer sum to anything sensible.

The one comparison that does carry information is *within* a run: on
`pcommon1`, removing congruent units (12.79) hurts far more than removing
opposite ones (4.88), consistent with an always-fuse network relying on
congruent cells. On the flagship the gap is much narrower (10.21 vs 7.55),
consistent with both populations mattering.

To make this analysis properly interpretable it needs a **size-matched random
lesion** as a baseline — remove *n* random units and compare. That control is
not currently implemented.

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
| flagship | **0.013** |
| pcommon07 | 0.014 |
| pcommon028 | 0.017 |
| pcommon0 | −0.028 |
| pcommon1 | −0.103 |

**How to read it.** Median gains within ±0.02 of zero in the three causal runs
mean the MSL population codes position in spatial coordinates — the
reference-frame transformation is complete by that layer. This is the continuity
check with the published paper.

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

Splits trials by the network's *own* causal judgment (joint implied weight >
0.5, falling back to the analytical posterior where the weight is unreadable)
and plots bias against |disparity| separately for each branch.

Conditioning on "inferred two causes" selects trials whose noise happened to
exaggerate the disparity, which can produce a counter-intuitive **negative**
bias — a fingerprint of inference over causal structure that no fixed-weight
scheme produces.

**Your results** (`conditioned_bias_negative_seen`):

| run | negative bias seen in the separate-cause branch |
|---|---|
| flagship | **False** |
| pcommon07 | **False** |
| pcommon028 | **True** |

**How to read it.** The truncation effect appears only at the lowest prior. That
is the direction you would expect — a low prior produces many more
confidently-segregated trials, so the selection effect has more trials to act
on — but with one run showing it and two not, this is a lead rather than a
result. If you want to pursue it, `pcommon03` (not yet run) would tell you
whether the effect grows monotonically as the prior falls.

## 7.11 Transition fit

Fits `w = 1 / (1 + exp(k(|d| − d0)))` to the implied weight against
|disparity|, giving a midpoint `d0` (the disparity at which fusion gives way to
segregation) and a sharpness `k`. Also fitted to the analytical posterior for
comparison.

**Your results:**

| run | prior | network midpoint | analytical midpoint | network sharpness | analytical sharpness |
|---|---|---|---|---|---|
| pcommon028 | 0.28 | 5.90 deg | 5.29 deg | 0.392 | 0.301 |
| flagship | 0.5 | 7.72 deg | 8.04 deg | 0.357 | 0.350 |
| pcommon07 | 0.7 | 10.13 deg | 9.93 deg | 0.397 | 0.414 |

**How to read it.** This is the cleanest cross-prior result in your data. The
midpoint moves monotonically with the prior — 5.90 → 7.72 → 10.13 deg as the
prior goes 0.28 → 0.5 → 0.7 — and tracks the analytical prediction (5.29 →
8.04 → 9.93) closely at every point. A network that believes one cause is more
likely tolerates more disparity before segregating, by very close to the amount
Bayes says it should.

---

# Part 8 — The cross-prior comparison

The satellites exist to test one prediction: **implied weights should shift with
the prior, per Bayes**. Since the only difference between these configs is the
Bernoulli constant, any systematic movement is attributable to the prior.

| quantity | p = 0.28 | p = 0.5 | p = 0.7 | moves with prior? |
|---|---|---|---|---|
| transition midpoint (network) | 5.90 | 7.72 | 10.13 | **yes, monotone** |
| transition midpoint (analytical) | 5.29 | 8.04 | 9.93 | yes |
| position regression slope, visual | 0.797 | 0.856 | 0.927 | **yes, monotone** |
| reliability slope (§7.2) | 0.906 | 0.567 | 0.712 | no clear pattern |
| variance hump, var_vis (network) | 1.15 | 3.40 | 4.58 | **yes, monotone** |
| variance hump, var_vis (analytical) | 1.89 | 4.55 | 7.07 | yes |
| MSL posterior decoding R² | 0.927 | 0.958 | 0.948 | flat (all high) |
| balance-vs-posterior correlation | 0.538 | 0.500 | −0.053 | **no** |
| best strategy | averaging | averaging | averaging | consistent |

**What this supports.** Three independent measures — the transition midpoint,
the position-regression slope, and the variance hump — move monotonically with
the prior and in the direction Bayes predicts. The midpoint tracks the
analytical value quantitatively. Model averaging wins at every prior.

**What it does not support.** The congruent/opposite balance result does not
replicate at 0.7. The reliability slope moves but not monotonically. Neither
should be presented as established without more evidence.

**The gap.** `pcommon03` has a config but no run. Four points would be
substantially more convincing than three, particularly for the reliability slope
where the current pattern is unclear.

---

# Part 9 — Every figure, and the calculation behind it

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
with the early-stopping epoch marked. Your flagship stopped at epoch 298 with
best validation 0.2158.

**`02_per_output_loss.png`** — validation loss split by output column. A flat,
high curve for one output while the others fall is the scale-imbalance
signature; it is what `balance_loss` exists to prevent.

## 9.3 `figures/model/` — network versus observer

Fourteen figures. Figures 01–07 exist for every run; 08–14 are produced only for
runs with a causal head and a prior strictly between 0 and 1 — which is why
`pcommon1` and `pcommon0` have 7 figures and the other three have 14.

**`01_output_scatter.png`** — network against analytical, one panel per output,
with the identity line and R²/RMSE in each title. The first thing to look at.

**`02_error_histograms.png`** — distribution of `network − analytical` per
output, with mean bias in the title.

**`03_post_c1_vs_disparity.png`** — the analytical posterior binned by signed
disparity on the config's `disparity_grid`. The classic Körding-shaped curve:
high near zero disparity, falling away on both sides. This is a property of the
*generative model*, not of the network — it is the reference the next figure is
read against.

**`04_fusion_weight.png`** — the classical view. Per trial, solve
`estimate = w·fused + (1−w)·seg` for `w` (dropping trials where
|fused − seg| < `min_separation`), then bin by disparity. Plots the analytical
posterior (dashed) against the weight implied by `mu_vis` and by `mu_prop`.

Both output columns encode the same weight, so the two curves are two readings
of one quantity. Near zero disparity neither is readable — the estimator returns
NaN there — so the centre of the curve rests on few trials and swings. That is a
property of the estimator, not of the network, and is precisely why the
position-domain regression (figure 08) is the headline instead of this one.

**`05_fusion_weight_by_reliability.png`** — the same implied-weight curve
computed separately at each of the config's `reliability_levels` (nearest-match
on `sig2_vis`). The prediction: less reliable cues tolerate more disparity
before segregating, so the transition midpoint moves outward.

**`06_decoding.png`** — bar chart of held-out R² for decoding `post_c1` from
each hidden layer. Your flagship: 0.482 at SIL, 0.958 at MSL.

**`07_emergent_vs_imposed.png`** — the same bars for the causal model and the
always-fuse twin side by side. Your flagship: 0.482/0.958 against 0.082/0.189.
The contrast is the emergence argument.

**`08_position_regression.png`** — the headline. Scatter of
`(network − seg)` against `post_c1 · (fused − seg)`, both in degrees, with the
identity line (Bayes-optimal) dashed and the actual fit drawn with its CI in the
legend. Your flagship: slope 0.856 [0.830, 0.881], intercept −0.007.

The vertical spine of points at x = 0 is the ill-conditioned zone — trials where
the two hypotheses agree. They are visible, and correctly carry almost no
leverage on the fit.

**`09_variance_hump_vis.png`** and **`10_variance_hump_prop.png`** — four lines
against the analytical posterior in 10 bins: network Var output, analytical
mixture variance, the between-component term alone, and the best fixed-weight
prediction. The last is flat by construction and is the control.

Your flagship `var_vis`: the network peaks at 16.4 around posterior 0.15–0.35
and falls to 4.5 at 0.95, against a fixed-weight line pinned at 7.3–8.0
throughout. The hump is +3.40 for the network against +4.55 analytical.

**`11_model_comparison.png`** — two panels. Left: per-posterior-decile implied
weight for the network and all five strategies. This is the discriminating
panel — averaging predicts a smooth sigmoid tracking the posterior, selection
predicts a step at 0.5. Right: per-decile RMSE of each strategy against the
network.

Your flagship: the network tracks averaging through the intermediate deciles and
departs clearly from the selection step, then falls short of averaging in the
top three deciles (0.62, 0.62, 0.51 against 0.84, 0.88, 0.94).

**`12_behavioral_bias.png`** — two panels. Left: bias against signed disparity,
network versus Bayes-optimal (Körding Fig. 2e analog). Right: bias against
|disparity| split by the network's own causal judgment (Fig. 3b–c analog). Both
drop bins with fewer than 30 trials — without that filter the right panel swings
by several degrees on a handful of trials and reads as structure.

**`13_congruency.png`** — two panels. Left: histogram of the congruency index
across MSL units, with the ±0.5 classification thresholds marked. Right: scatter
of the congruent-minus-opposite activity balance against the analytical
posterior, titled with the correlation.

Your flagship: the index histogram is bimodal (17 congruent, 24 opposite), and
the balance correlates 0.500 with the posterior. On `pcommon07` the same left
panel looks similar (14/29) but the right panel shows nothing (−0.053) — the
inconsistency noted in §7.9.

**`14_rf_shifts.png`** — two histograms. Left: distribution of RF shift gain
across MSL units, with 0 (spatial code) and +1 (retinal code) marked. Right:
distribution of gain-field slopes. Your flagship median shift gain: 0.013.

---

# Part 10 — How to run things

```bash
make test                                       # 67 property tests, ~40 s
make calibrate CONFIG=configs/flagship.yaml     # the gate alone
make all       CONFIG=configs/flagship.yaml     # full pipeline, 50k trials
make quick     CONFIG=configs/flagship.yaml     # same, 8k trials / 60 epochs
```

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
residuals also contain any causal-inference misweighting, so σ_w comes out too
large, never too small.

To finish the satellite series:

```bash
make all CONFIG=configs/pcommon03.yaml          # the missing 0.3 point
```

---

# Part 11 — Things to be careful about when reading these results

**One seed, one architecture.** Everything above is `seed: 0`. None of the
numbers carry a run-to-run error bar. Before anything goes in a paper, the
headline quantities need several seeds.

**Two results do not replicate across priors.** The congruent/opposite balance
(0.50, 0.54, but −0.05) and the truncation effect (True at 0.28, False at 0.5
and 0.7). Both are currently single-run observations.

**The lesion analysis lacks its control.** Every lesion is catastrophic because
zeroing a third of a layer breaks the read-out's operating point. Without a
size-matched random lesion, only within-run comparisons between subpopulations
mean anything.

**The hand-vs-visual consistency correlation is not what it looks like.** Read
the mean absolute difference (0.12–0.18), not the correlation (≈0.11), for the
reason given in §7.4.

**`pcommon07` carries a calibration WARN.** 10.3% intermediate posterior mass
against a 15% floor. Its ambiguity-dependent analyses rest on fewer trials.

**The two controls' decoding R² of ≈0 is expected**, not a finding. Their
posterior is constant, so there is no variance to explain.

**The design document's §5 eye-transform formula disagrees with the code**, and
the code is the one that matches the exact posterior. The document is the thing
to amend.

**The consistent sub-optimality is real.** Position-regression slopes of
0.80–0.93 with CIs excluding 1, a variance hump at 60–75% of analytical, and a
top-decile weight shortfall all point the same way: the network's implicit
causal inference is close to Bayesian but systematically conservative. Since
every pipeline guard is verified and both controls are clean, this is
attributable to the network. Per the design document, that makes it a reportable
finding rather than a failure.
