# `causal_integration` — Complete Guide

**What this document is.** A walkthrough of the codebase, every config, every
analysis, and every figure — with the calculation behind each one and the number
your own runs actually produced.

**Provenance.** Every number below was read out of files in your repo:
`results/<run>/metrics.json`, `results/calibration/<config>/metrics.json`, and
`results/prior_sweep/sweep.json`, as they stood after your `make all` and
`make sweep SEEDS="0 1 2"` runs. Nothing here is estimated, remembered, or
carried over from anywhere else. Where a result is absent or degenerate I say so
rather than filling the gap.

**Start here if you are writing the paper.** Part 12 grades every analysis in
this document as a finding, a supporting result, a control, or something that
should not be presented as a result at all. It also flags one reproducibility
problem you need to fix first (§12.0).

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

**Where the σ_w filter must not be used.** Filtering is fine for *reporting a
per-trial weight*. It is **biased** for *estimating a weight within a bin*,
because the filter keeps preferentially large-|Δ| trials and those are
systematically the lowest-weight ones — so a binned average of filtered ratios
sits below the analytical posterior even for a perfectly Bayesian network. Any
binned weight curve should instead use the division-free least-squares form in
`analysis.binned_implied_weight` (§8.2), which uses every trial. §7.3 is the
same idea applied to the whole dataset at once.

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
| **green** | control: best possible *fixed* weight, `w̄·V_fused + (1−w̄)·V_seg` |

The x-axis is the analytical posterior, **not disparity**. Far left: certain
there were two causes. Far right: certain there was one. Middle: genuinely
unsure which.

The scalar summary is the **hump**: mean variance in the intermediate bins
(0.2 < centre < 0.8) minus mean variance in the confident bins.

### Your flagship, `var_vis`, per bin (from `metrics.json`)

| posterior bin | network | analytical mixture | between term | fixed-weight | n |
|---|---|---|---|---|---|
| 0.05 | 11.89 | 11.82 | 0.41 | 7.54 | 2588 |
| 0.15 | 16.39 | 18.07 | 7.30 | 7.89 | 137 |
| 0.25 | 16.27 | 18.89 | 8.91 | 7.93 | 122 |
| 0.35 | 16.09 | 18.73 | 9.46 | 7.97 | 139 |
| 0.45 | 13.88 | 15.99 | 7.96 | 7.64 | 148 |
| 0.55 | 13.15 | 14.08 | 6.62 | 7.92 | 208 |
| 0.65 | 10.62 | 10.80 | 4.22 | 7.83 | 346 |
| 0.75 | 7.81 | 7.67 | 1.89 | 7.99 | 883 |
| 0.85 | 5.49 | 5.25 | 0.40 | 7.59 | 2233 |
| 0.95 | 4.54 | 4.08 | 0.21 | 7.28 | 696 |

### Reading it: two effects are stacked

Splitting the mixture into its two brackets (`within = mixture − between`,
derived from the table above) is what makes the blue curve legible:

| posterior | 0.05 | 0.15 | 0.25 | 0.35 | 0.45 | 0.55 | 0.65 | 0.75 | 0.85 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|
| within | 11.41 | 10.76 | 9.98 | 9.27 | 8.03 | 7.46 | 6.57 | 5.78 | 4.84 | 3.87 |
| between | 0.41 | 7.30 | 8.91 | **9.46** | 7.96 | 6.62 | 4.22 | 1.89 | 0.40 | 0.21 |
| between as % of total | 3% | 40% | 47% | **51%** | 50% | 47% | 39% | 25% | 8% | 5% |

**The ramp.** `within` falls monotonically 11.41 → 3.87. Segregation is
expensive: if the cues are separate, the visual estimate is just the visual
measurement (variance 11.4), while fusion buys 3.9. The curve therefore slopes
downhill left-to-right even with zero causal uncertainty.

**The hump.** The dotted line rides on top of that ramp, peaking at 9.46.

Blue is the sum of the two, which is why it is high on the left, humped in the
middle, and low on the right.

**At the peak, half the network's reported uncertainty is causal uncertainty** —
9.46 of 18.73 at bin 0.35. That is the headline of the figure.

### Three features that look wrong and are not

**The peak sits at 0.35, not 0.5.** Two factors multiply. `w(1-w)` peaks at
0.5, but `d^2` is falling as `w` rises — mean |disparity| across the bins runs
31.4° at the left edge down to about 2° at the right. Rising times falling peaks
left of centre.

**The leftmost point is not part of the hump.** Bin 0.05 has the largest
disparity of any bin (mean 31.4°) — the hypotheses are further apart there than
anywhere else — yet its between term is the *smallest* (0.41). The reason is
that mean `w(1-w)` in that bin is 0.0052: the network is certain the causes are
separate. The between term measures **indecision, not disagreement**. That
bin's total of 11.9 is essentially all `within`.

**The green line is flat by construction, not by fit.** Any constant weight,
whatever value is chosen, gives a flat line, because fixing `w` removes the
`w(1-w)` factor. Green pinned at 7.28–7.99 across every bin is the null
hypothesis drawn on the plot.

### Hump values across your runs

| run | prior | var_vis network | var_vis analytical | var_prop network | var_prop analytical |
|---|---|---|---|---|---|
| flagship | 0.5 | **+3.40** | +4.55 | +0.94 | +1.24 |
| pcommon07 | 0.7 | **+4.58** | +7.07 | +1.09 | +1.59 |
| pcommon028 | 0.28 | **+1.15** | +1.89 | +0.29 | +0.49 |

The hump is present in every run and in both output channels. The network
reproduces it at roughly 60–75% of its analytical magnitude — the same
conservatism the position regression reports, appearing in the uncertainty
channel.

The network also slightly *over*-reports variance in the most confident bins
(4.54 vs 4.08 at centre 0.95) and *under*-reports in the ambiguous ones. That is
compression toward the mean, the signature of a read-out smoothing a sharply
peaked target.

### `var_prop` is the same figure at about quarter scale

Figure 10 is not an independent result; it is a consistency check. Its between
term peaks at 2.26 against 8.91 for `var_vis`, and its hump is +0.94 against
+3.40.

The reason is arithmetic. Recovering `|d|` per bin as
`sqrt(between / mean[w(1-w)])`, the vis/prop ratio is

```
1.95  1.99  1.99  2.11  1.93  1.97  1.92  2.07  1.97  1.95
```

— flat at about 1.97 across all ten bins. The between term goes as `d^2`, hence
roughly a quarter.

`d_prop` is half `d_vis` because **proprioception is the more reliable cue in
this setup** (segregated variance 5.58 against 11.41 for vision), so the fused
estimate sits nearer proprioception. Precision weighting predicts the ratio
should be `sigma^2_vis / sigma^2_prop = 2.05` (from the unrounded 11.4098 and
5.5773 the table shows as 11.41 and 5.58); the measured 1.97
differs by the prior term in the fused estimate.

### One caveat to carry

The middle bins are thin: 122–208 trials each, against 2588 at the left edge.
The shape of the hump is estimated from roughly 750 trials in total.

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

Ablates a subpopulation's MSL units and re-measures the read-out error. Two
things make this interpretable, and both were added after the first version of
this analysis gave misleading numbers:

**1. Mean-clamping, not zeroing.** An ablated unit is clamped to its *mean
activation across trials*. That removes the unit's information — it no longer
varies with the trial — while leaving the read-out's operating point intact.
Zeroing is wrong for a sigmoid layer: on the flagship the per-unit mean
activations run 0.013 to 0.990 (median 0.408), so forcing a unit to 0 does not
remove it, it injects a large constant perturbation the read-out's weights and
biases were never calibrated for. The damage then measures the size of that
perturbation, not the unit's role.

**2. A size-matched random baseline.** Ablating any *k* of 64 units costs
something. The question is whether ablating *these k* costs more than ablating
*k arbitrary ones*. For each subpopulation the code draws `n_random` random
subsets of the same size, ablates each, and reports where the real lesion falls
in that null distribution as a z-score and a percentile.

**Your flagship, both modes, 200 random draws** (intact RMSE 1.479):

| ablation | k | mode | RMSE | random null | z | percentile |
|---|---|---|---|---|---|---|
| congruent | 17 | zero | 10.209 | 7.928 ± 2.100 | +1.09 | 85% |
| opposite | 24 | zero | 7.546 | 8.876 ± 1.878 | −0.71 | 22% |
| mixed | 21 | zero | 8.882 | 8.467 ± 2.089 | +0.20 | 70% |
| **congruent** | 17 | **mean** | **8.696** | 4.986 ± 0.592 | **+6.27** | **100%** |
| **opposite** | 24 | **mean** | **3.979** | 6.480 ± 0.687 | **−3.64** | **0%** |
| mixed | 21 | mean | 5.902 | 5.845 ± 0.676 | +0.08 | 53% |

Per output, mean-clamp mode, z against the random baseline:

| ablation | mu_vis | var_vis | mu_prop | var_prop |
|---|---|---|---|---|
| no congruent | **+7.77** | −1.48 | **+3.11** | −0.88 |
| no opposite | −3.53 | −0.12 | −2.42 | −1.10 |
| no mixed | −0.39 | −1.64 | +1.10 | −1.45 |

**How to read it.** Under the corrected ablation the conclusion is specific and
it is *not* the one the raw numbers first suggested:

- **Congruent units are load-bearing, strongly and specifically** — z = +6.27
  overall, and the effect is concentrated in the two *mean* channels (+7.77 on
  `mu_vis`, +3.11 on `mu_prop`), not the variance channels.
- **Opposite units are not load-bearing for the read-out at all.** Removing them
  costs *less* than removing 24 random units (z = −3.64, 0th percentile).
- **Mixed units are exactly average** (z = +0.08) — unremarkable, which is what
  the label implies.

The same analysis on the always-fuse control `pcommon1` (37 congruent, 6
opposite) gives congruent z = +5.94 and opposite z = +0.31 — congruent units
matter there too, and its handful of opposite units do not.

**What this does and does not say.** It says the position estimates are carried
by congruent units, and that the network does not *need* opposite units to
produce its four outputs. It does **not** say opposite units carry nothing: the
balance analysis above shows their activity tracks the posterior at r = 0.50 on
this same network. A code can carry information redundantly, so that no single
subpopulation is necessary. Necessity and representation are different
questions, and this analysis only answers the first.

The earlier reading — "both subpopulations load-bearing" — was an artefact of
zeroing sigmoid units, and is withdrawn.

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

**A guard on the fit.** `curve_fit` can report convergence on a midpoint far
outside the data — on an undertrained smoke run it returned −48219°, which is
not a transition but a flat curve fitted by a logistic whose midpoint ran away.
The fit is therefore rejected (returning NaN) unless `k > 0` and `d0` lies
within a quarter-span of the observed disparity range. A NaN midpoint means "no
transition was measurable", which is information; a large finite one would be a
silent lie.

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

# Part 8 — The cross-prior experiments

Two things go under this heading. The **satellite runs** (§8.1) are three full
pipelines at three priors, each with the complete analysis suite. The **prior
sweep** (§8.2) is a dedicated experiment at nine priors and three seeds that
measures a smaller set of quantities. The sweep is the stronger evidence and is
what belongs in the manuscript; the satellites are what let you inspect any
individual network in depth.

## 8.1 The three satellite runs

The satellites test one prediction: **implied weights should shift with the
prior, per Bayes**. Since the only difference between these configs is the
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
the prior and in the direction Bayes predicts.

**What it does not support.** The congruent/opposite balance result does not
replicate at 0.7. Neither it nor the reliability slope should be presented as
established on this evidence.

Three points and one seed each is, however, a weak design for a claim about a
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

### How the per-bin weight is recovered, and why the obvious method is wrong

Within each disparity bin, regress `(network − seg)` on `Δ = fused − seg`
through the origin:

```
w_bin = Σ(Δ · (network − seg)) / Σ(Δ²)          SE = σ_out / √(Σ Δ²)
```

The intuitive alternative — take the per-trial ratio `(network − seg)/Δ`, filter
it by σ_w, and average within the bin — is **biased low**. The σ_w filter keeps
preferentially large-|Δ| trials, and within a disparity bin those are the trials
whose hypotheses are most separated, which are also the lowest-weight trials.
The filtered average therefore sits below the analytical posterior *even for a
perfectly Bayesian network*. The least-squares form uses every trial and has no
such selection. (`analysis.binned_implied_weight`.)

Two consequences are visible in Panel A and are deliberate:

- **Bins are coarse near zero disparity.** There the two hypotheses coincide,
  Σ(Δ²) collapses, and no amount of data identifies a weight. A fine grid there
  produces a spike that is an artefact of the estimator. Bins are also dropped
  when their slope SE exceeds 0.05.
- **Gaps break the line rather than being bridged.** At high priors many
  far-disparity bins hold too few trials to keep. Joining across them would draw
  a transition that was never measured.

### Your results — the aggregated table

From `results/prior_sweep/sweep.json`, `aggregated` block. All ± are **SEM
across the three seeds**, not within-run CIs. The distinction matters: a
within-run CI says how well *one network's* behaviour is pinned down; only the
across-seed spread says whether the result survives retraining.

| `p_common` | midpoint net | midpoint opt | posreg slope | hump net | hump opt | sharp net | sharp opt | MSL R² |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 2.72 ± 0.67 | 0.44 | 0.764 ± 0.035 | −0.65 ± 0.33 | −0.15 | 0.391 | 0.254 | 0.861 |
| 0.2 | 3.83 ± 0.23 | 3.68 | 0.826 ± 0.017 | 0.38 ± 0.06 | 0.75 | 0.346 | 0.281 | 0.929 |
| 0.3 | 5.70 ± 0.26 | 5.52 | 0.831 ± 0.017 | 1.54 ± 0.06 | 2.05 | 0.402 | 0.304 | 0.942 |
| 0.4 | 6.96 ± 0.15 | 6.88 | 0.820 ± 0.025 | 2.22 ± 0.05 | 3.12 | 0.365 | 0.329 | 0.958 |
| 0.5 | 8.06 ± 0.18 | 7.97 | 0.848 ± 0.006 | 3.26 ± 0.11 | 4.46 | 0.359 | 0.353 | 0.959 |
| 0.6 | 8.90 ± 0.42 | 8.97 | 0.906 ± 0.010 | 3.82 ± 0.05 | 5.71 | 0.356 | 0.381 | 0.953 |
| 0.7 | 9.82 ± 0.21 | 9.93 | 0.906 ± 0.013 | 4.30 ± 0.14 | 6.97 | 0.375 | 0.413 | 0.944 |
| 0.8 | 11.61 ± 0.11 | 10.96 | 0.950 ± 0.007 | 3.94 ± 0.05 | 7.93 | 0.420 | 0.455 | 0.924 |
| 0.9 | 12.65 ± 0.17 | 12.31 | 0.932 ± 0.005 | 2.83 ± 0.32 | 10.27 | 0.323 | 0.515 | 0.870 |

Read-out R² on `mu_vis` runs 0.988–0.995 across the whole sweep, so no network
is simply fitting the task worse at the extremes.

### The three headline numbers

**1. The midpoint tracks Bayes with no free parameters.** Regressing the network
midpoint on the analytical midpoint:

```
all nine priors     slope 0.882   intercept +1.27 deg   R² 0.968
                    mean |net − analytical| = 0.44 deg    max 2.28 deg (at p = 0.1)

p_common >= 0.2     slope 1.027   intercept -0.06 deg   R² 0.994
                    mean |net − analytical| = 0.21 deg    max 0.65 deg
```

Over a midpoint range of 0.4–12.7°, restricted to the nine-tenths of the range
where the measurement is well-posed, the network lands **within a fifth of a
degree of the Bayesian prediction on average, with a fitted slope of 1.03 and
R² = 0.994.** That is the strongest quantitative result in the project.

**2. Model averaging wins at every prior and every seed** — 27 of 27 — in the
five-way comparison against full integration, full segregation, model selection,
and the best fixed-weight model.

**3. A consistent, mild conservatism.** The position-regression slope rises
monotonically in trend from 0.764 at `p_common` = 0.1 to 0.950 at 0.8
(+0.213 per unit prior, r = 0.950), and **`slope + 1.96·SEM < 1` at all nine
priors** (worst case 0.963 at p = 0.8). The network applies slightly less pull
toward the fused solution than the ideal observer, at every prior, across every
seed. This is the same deviation the flagship position regression reports, now
established as a property of the model rather than of one training run.

### Why p_common = 0.1 is excluded from the second fit

Not to flatter the result. At `p_common` = 0.1 the analytical midpoint is 0.44°,
essentially at the origin, where a logistic midpoint is barely identifiable —
there is almost no fusion regime for the transition to be a transition *out of*.
The symptom is in the seed spread: the three midpoints are **2.34, 4.03, 1.80**,
an across-seed SEM of 0.67° against 0.11–0.42° everywhere else. It is the
noisiest point in the sweep by a factor of two, and it is the single point
driving the +1.27° intercept.

Report both fits. Excluding a point silently would be indefensible; excluding it
with the seed spread shown, and the reason stated, is ordinary practice.

## 8.3 What the sweep settles that three runs could not

| question | three satellites said | 27 networks say |
|---|---|---|
| does the midpoint track Bayes? | yes, 3 points | **yes, slope 1.03, R² 0.994 over p ≥ 0.2** |
| is the conservatism real? | slopes 0.80–0.93, one seed each | **yes — CI excludes 1 at all nine priors, 3 seeds** |
| does model averaging always win? | 3 for 3 | **27 for 27** |
| does the reliability slope move? | "no clear pattern" | **declining trend, r = −0.867** (see below) |
| does the balance result replicate? | no | *not measured by the sweep* |

**The reliability slope needs a second look.** With three runs it looked like
noise. Across 27 networks it declines with the prior — 0.951, 0.882, 0.805,
0.653, 0.639, 0.605, 0.672, 0.691, 0.456 — with r = −0.867 and a trend of
−0.477 per unit prior, and its CI excludes 1 at seven of the nine priors. So
there is a real pattern where there previously appeared to be none. What it
*means* is a separate question, and the honest position is that this analysis
was not designed to answer it: the statistic is computed on the subset of
disparity bins that survive a spread filter, that subset changes composition
with the prior, and no control rules out the change being a property of the
surviving bins rather than of the network. Treat it as a lead, not a result.

### The variance hump across the sweep

The hump is a real effect over the middle of the range and a measurement failure
at the top. The network-to-analytical ratio, `p_common` = 0.2 → 0.9:

```
0.51   0.75   0.71   0.73   0.67   0.62   0.50   0.28
```

It holds around two-thirds from 0.3 to 0.6 and then collapses. The reason is
trial counts, not mechanism: the hump statistic is defined on
intermediate-posterior trials (0.2 < posterior < 0.8), and at extreme priors
there are almost none. At `p_common` = 0.9 the flagship-scale run has on the
order of a couple of hundred such trials. At `p_common` = 0.1 the hump is
*negative* in both network and analytical values (−0.65 and −0.15) — there is no
hump to measure there at all.

**For the manuscript:** supplementary, restricted to `p_common` ∈ [0.2, 0.7],
with the trial counts stated. Do not present the full range as a monotone
result; it is not one, and a reader checking counts will notice.

---

# Part 9 — Every figure, and the calculation behind it

**Every figure is written to the PLOS ONE specification** (`cmsi/viz/style.py`
carries the numbers): drawn at its final printed size — 5.2 in wide for a
single panel, 7.5 in for a multi-panel figure, never taller than 8.75 in —
at 300 dpi, in Arial (Liberation Sans on Linux, which is metrically identical)
with every font between 8 and 12 pt, and lettered panels on every multi-panel
figure. Each figure is saved three times: `.png` to look at, `.tif` for
submission (flattened RGB, no alpha channel, LZW-compressed, 300 dpi metadata),
and `.svg` to edit. The sweep figures additionally get a `.pdf` for the LaTeX
draft. `save_figures` reads every file back and warns if it breaks a limit;
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

## 9.4 `results/prior_sweep/figures/` — the cross-prior experiment

Produced by `scripts/05_prior_sweep.py`, not by `04_figures.py`. These are not
per-run figures: each one summarises all 27 networks.

**`prior_sweep.png`** (with `.tif` for submission, `.svg` to edit and `.pdf` for
LaTeX) — the main figure, three panels, 7.5 in wide.

- **Panel A** — implied fusion weight against signed body-frame disparity, one
  curve per prior, colour on a sequential viridis ramp (light = low prior).
  Solid with error bars: the network, from `binned_implied_weight` (§8.2).
  Dashed: the analytical posterior for the same prior. The curves fan out in an
  orderly family — at a low prior the network abandons fusion within a few
  degrees; at a high prior it holds a near-complete weight past 10°.
  Error bars are `1.96 × SE` where `SE = σ_out / √(ΣΔ²)` within the bin.
  Broken lines are dropped bins, not missing data — see §8.2.
- **Panel B** — network transition midpoint against analytical midpoint, one
  point per prior, with an identity line. No parameter relates the two axes.
  With three or more seeds the points carry across-seed SEM bars. If every
  logistic fit is rejected by the §7.11 guard the panel prints "no usable
  transition fit" instead of crashing.
- **Panel C** — position-regression slope against prior, with the Bayes-optimal
  value of 1 marked. The legend states which kind of error bar is drawn:
  "within-run 95% CI" for a single seed, "mean ± 95% CI across N seeds"
  otherwise. Read the label — the two mean different things (§8.2).

**`variance_hump_vs_prior.png`** — supplementary. Mid-ambiguity variance
elevation, network against analytical, per prior. Interpret only over
`p_common` ∈ [0.2, 0.7]; outside that band the statistic is trial-count limited
rather than informative (§8.3).

---

# Part 10 — How to run things

```bash
make test                                       # 81 property tests, ~45 s
make calibrate CONFIG=configs/flagship.yaml     # the gate alone
make all       CONFIG=configs/flagship.yaml     # full pipeline, 50k trials
make quick     CONFIG=configs/flagship.yaml     # same, 8k trials / 60 epochs
make sweep     SEEDS="0 1 2"                    # the cross-prior sweep, ~45 min
make figures   RUN=flagship                     # redraw one run's figures
python scripts/05_prior_sweep.py --replot       # redraw the sweep, no training
```

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
residuals also contain any causal-inference misweighting, so σ_w comes out too
large, never too small.

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

To finish the satellite series:

```bash
make all CONFIG=configs/pcommon03.yaml          # the missing 0.3 point
```

This is now optional. The sweep covers 0.3 with three seeds; a full satellite
run at 0.3 would only add the unit-level and behavioural analyses, which the
sweep does not measure.

---

# Part 11 — Things to be careful about when reading these results

**One seed for everything except the sweep.** Every number in Parts 6, 7 and 8.1
is `seed: 0` and carries no run-to-run error bar. The prior sweep (§8.2) is the
exception: three seeds per prior, 27 networks, with across-seed SEMs. So the
transition midpoint, the position-regression slope and the model comparison
*are* seed-checked; the unit-level analyses, the behavioural curves and the
calibration gate are not.

**Two results do not replicate across priors.** The congruent/opposite balance
(0.50, 0.54, but −0.05) and the truncation effect (True at 0.28, False at 0.5
and 0.7). Both are currently single-run observations, and the sweep does not
measure either.

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
0.76–0.95 with `slope + 1.96·SEM < 1` at all nine priors across three seeds, a
variance hump at 60–75% of analytical over the well-measured range, and a
top-decile weight shortfall all point the same way: the network's implicit
causal inference is close to Bayesian but systematically conservative. Since
every pipeline guard is verified and both controls are clean, this is
attributable to the network. Per the design document, that makes it a reportable
finding rather than a failure.

**The lesion result answers necessity, not representation.** Mean-clamping plus
a size-matched random baseline (§7.9) shows congruent units are load-bearing
(z = +6.27) and opposite units are not (z = −3.64). That does not mean opposite
units carry nothing — their activity tracks the posterior at r = 0.50 on the
same network. A redundant code can make every subpopulation dispensable.

**The reliability slope is a lead, not a result.** It has a real declining trend
across 27 networks (r = −0.867), but the statistic is computed on a
prior-dependent subset of bins and has no control for that. §8.3 gives the
argument.
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

## 12.0 Read this before citing any unit-level number

**Every `metrics.json` in `results/` predates the lesion fix.** The five run
directories are dated 2026-08-24; `units.py` and `03_analyze.py` were corrected
on 2026-08-26. The stored `lesion` blocks therefore contain the **old
zero-ablation** numbers with no random baseline — the mean-clamp table in §7.9
(z = +6.27, −3.64) cannot currently be reproduced from any file in `results/`.

Stage 3 does not retrain. Regenerating is cheap:

```bash
for r in flagship pcommon07 pcommon028 pcommon0; do
  python scripts/03_analyze.py --run $r --twin ${r}_twin --control pcommon1
done
```

Do this before the lesion result goes anywhere near a manuscript. Everything
graded **A** below is unaffected — those come from the prior sweep and from
analyses whose stored values are current.

## 12.1 Grade A — the findings

**A1. The implied fusion weight depends on the prior exactly as Bayes requires.**
§8.2. Twenty-seven networks. Over `p_common` ≥ 0.2 the network transition
midpoint regresses on the analytical midpoint with slope **1.027**, intercept
**−0.06°**, **R² = 0.994**, mean deviation **0.21°** — with **no free parameter
relating the two axes**. This is the strongest result in the project and should
be the paper's central figure.

Its force is in what it rules out: a disparity heuristic predicts a
prior-independent midpoint, a fixed-weight scheme predicts no transition at all,
and model selection predicts a step with sharpness → ∞. All three are excluded
by the same measurement.

**A2. Model averaging, not model selection.** §7.7 and §8.2. The five-way
comparison selects averaging in **27 of 27** sweep networks plus the flagship
and both satellites. This is the Körding et al. (2007) question asked of a
network that was never given a causal read-out, and it answers cleanly.

**A3. The variance output carries the between-component signature of causal
ambiguity.** §7.6. The `w(1−w)d²` term is the part of the mixture variance that
**no fixed-weight model can produce**, and the network reproduces it — at the
peak, half its reported uncertainty (9.46 of 18.73 deg²) is causal uncertainty
rather than measurement noise. Present in both output channels and every run.
The flat fixed-weight control line makes the argument visually on the figure.

The claim is strongest as an *existence and shape* result. Its amplitude, and
its prior dependence, are weaker — see D4.

**A4. The posterior is emergent, not imposed.** §7.8. The trial-wise posterior
decodes linearly from the multisensory layer at R² = 0.93–0.96 while rising from
only 0.27–0.48 at the single-modality layer, and the always-fuse twin — same
architecture, same inputs, fused targets only — reaches just 0.19–0.21. The twin
is what turns this from an observation into a controlled contrast, and it should
be reported alongside, not separately.

**A5. A small, consistent conservatism.** §7.3 and §8.2. Position-regression
slopes run 0.764–0.950 and `slope + 1.96·SEM < 1` at **all nine priors across
three seeds** (worst case 0.963). The network applies slightly less pull toward
the fused solution than the ideal observer, everywhere.

**Frame this as a finding, not an apology.** It is a reproducible property of a
network trained on exactly Bayesian targets, it is visible in three independent
measures (position slope, variance-hump amplitude, top-decile weight shortfall),
and every pipeline guard and both controls are clean. A referee will ask whether
it is a bug; §12.0 aside, the answer is documented and it is not.

## 12.2 Grade B — supporting results

**B1. The disparity-heuristic alternative is excluded directly.** §7.5.
Within-disparity-bin slopes of 0.57–0.91, far from the 0 a heuristic predicts.
Supporting rather than headline because only 2–3 of 8 bins survive the spread
filter — past ~20° the posterior is numerically pinned and there is nothing to
test. A2 and A1 rule out the same alternative on much more data.

**B2. Congruent units are necessary for the position read-out; opposite units
are not.** §7.9, with mean-clamping and a size-matched random baseline
(congruent z = +6.27, opposite z = −3.64, mixed z = +0.08), and the effect
concentrated in the two mean channels. **Subject to §12.0** — regenerate before
citing.

State the scope precisely: this is a claim about **necessity**, not about
representation. Opposite units track the posterior at r = 0.50 on the same
network, and a redundant code can make any subpopulation dispensable.

**B3. The reference-frame transformation is complete by the multisensory
layer.** §7.9, median RF shift gain 0.013 (0.014 and 0.017 at the other priors).
This is the continuity check with Farahmandi et al. — it says the present
network reproduces the previous result before adding anything causal, which is
what licenses the comparison.

**B4. Congruent/opposite units emerge at all.** §7.9, a bimodal congruency-index
histogram (17 congruent, 24 opposite on the flagship). Descriptive, and the link
to Rideaux et al. (2021). The *balance* analysis built on it is D1 — the
classification is fine, the correlation is not.

**B5. The Körding Fig. 2e bias curve is reproduced.** §7.10, non-monotonic pull
that grows with disparity then collapses. A recognisable qualitative signature;
it is not independent evidence, since it is the same behaviour A1 measures
quantitatively.

**B6. Hand and visual channels imply the same weight.** §7.4, mean absolute
difference 0.12–0.18 on a quantity bounded in [0, 1]. Quote the MAD. Do not
quote the correlation — see D5.

## 12.3 Grade C — method verification and controls

None of these is a finding. They belong in Methods or a supplement, and they
matter — a referee who wants to know whether the pipeline is sound is asking for
exactly this list. But writing them up as discoveries reads as padding.

| item | § | what it establishes |
|---|---|---|
| the eight-check calibration gate | Part 6 | the dataset can support the analyses at all |
| read-out accuracy, R² 0.988–0.995 | §7.1 | the network learned the task; not a result about causal inference |
| σ_out measured on the `p_common` = 1 control | §7.2 | the noise scale is measured where the target is single-valued |
| the always-fuse twin | §7.8 | the contrast that makes A4 a controlled claim |
| `pcommon0` / `pcommon1` decoding R² ≈ 0 | §7.8 | **expected** — the posterior is constant, so there is no variance to explain |
| Poisson validity, range containment, anti-confound AUCs | Part 6 | the encoders behave as specified and no confound is available |
| the `transition_fit` range guard | §7.11 | a failed fit returns NaN instead of a fabricated midpoint |
| least-squares binned weight vs the filtered ratio | §7.2, §8.2 | the estimator is unbiased; the obvious alternative is not |

The last two are worth a Methods sentence each. They are the kind of detail that
pre-empts a reviewer question rather than inviting one.

## 12.4 Grade D — do not present these as findings

**D1. The congruent-minus-opposite balance correlation.** §7.9. 0.538 at
`p_common` = 0.28, 0.500 at 0.5, **−0.053** at 0.7. It does not replicate. One
seed each, and the sweep does not measure it. Either omit it, or run it across
the sweep's 27 networks and report whatever comes out — but do not report the
two positive values without the third.

**D2. The truncation / negative-bias effect.** §7.10,
`conditioned_bias_negative_seen`: True at 0.28, False at 0.5 and 0.7. One run
each. The direction is theoretically sensible, which makes it tempting and makes
it worse — a plausible single-run observation is exactly the thing that fails to
replicate. A lead for future work.

**D3. The reliability-within-disparity slope.** §8.3. It does have a real
declining trend across 27 networks (r = −0.867), so it is no longer noise. But
the statistic is computed on the subset of bins that survive a spread filter,
that subset changes composition with the prior, and there is no control
separating a property of the network from a property of the surviving bins.
Report it as an observation needing follow-up, or not at all.

**D4. The variance hump above `p_common` = 0.7, and at 0.1.** §8.3. The
network-to-analytical ratio holds near two-thirds from 0.3 to 0.6, then falls to
0.50 at 0.8 and **0.28** at 0.9 — because the statistic is defined on
intermediate-posterior trials and at extreme priors there are almost none. At
`p_common` = 0.1 the hump is negative in both network and analytical values,
so there is nothing there to reproduce. Restrict the supplementary panel to
[0.2, 0.7] and state the trial counts.

**D5. The hand-vs-visual consistency correlation (≈ 0.11).** §7.4. Low not
because the two disagree but because the underlying weight barely varies across
the trials that pass the filter, so the correlation has nothing to correlate.
Quoting it would understate the agreement. Use the MAD (B6).

**D6. The `p_common` = 0.1 transition midpoint.** §8.2. Across-seed SEM 0.67°
against 0.11–0.42° elsewhere; the three seeds give 2.34, 4.03, 1.80. The
analytical midpoint is 0.44°, essentially at the origin, where a logistic
midpoint is barely identifiable. Show the point, show the error bar, and report
the fit both with and without it.

**D7. Any lesion claim about causal behaviour specifically.** §7.9 measures
lesion effects on read-out RMSE. It does not establish that either subpopulation
is necessary for the *causal* computation as distinct from the position
estimate — and the per-output breakdown points the other way, with the congruent
effect concentrated in the mean channels (+7.77, +3.11) and absent from the
variance channels (−1.48, −0.88), which are where the causal signature lives
(A3). Do not extend B2 into a claim about causal inference.

## 12.5 The shortest honest version of the paper's claims

If the Results section had to be four sentences:

> A feedforward network trained only to report position and uncertainty, with no
> causal read-out, develops an implicit fusion-vs-segregation trade-off that
> tracks the Bayesian posterior over causal structure quantitatively across nine
> priors and three seeds (slope 1.03, R² 0.994, no free parameters). A five-way
> model comparison selects model averaging over model selection, full
> integration, full segregation and the best fixed-weight observer in all 27
> networks. The network's uncertainty output carries the between-component
> variance term that no fixed-weight scheme can produce. The posterior itself is
> linearly decodable from the multisensory layer, but not from an
> identically-structured network trained on fused targets alone.

Everything in 12.2 supports those four sentences. Everything in 12.3 belongs in
Methods. Everything in 12.4 belongs in a limitations paragraph or nowhere.
