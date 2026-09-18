# causal_integration

An additive feedforward network trained to perform Bayesian causal inference on
population-coded sensory cues, across reference frames. Two cues report where the
hand is — vision in retinal coordinates, proprioception in body coordinates — and
the network has to decide, implicitly, whether they share a cause.

## Setup

Requires Python 3.10–3.13 and [Poetry](https://python-poetry.org/docs/#installation).

```bash
poetry install        # or: make setup
```

That creates `.venv/` inside the project (configured in `poetry.toml`) and
installs the exact versions recorded in `poetry.lock` — not merely compatible
ones. `pyproject.toml` says what the project tolerates; the lock says what your
results were actually produced with, which is why it's committed rather than
ignored.

Run things either through Poetry or by activating the environment:

```bash
poetry run pytest              # one-off
poetry shell                   # or activate, then just: pytest
```

**PyCharm:** Settings → Project → Python Interpreter → Add → Poetry Environment,
and point it at `.venv/bin/python` (`poetry env info --path` prints it). Mark
`src/` as a Sources Root so imports resolve in the editor.

**Reproducing this environment elsewhere** — a cluster, a colleague's machine,
your own laptop in two years — is `git clone` then `poetry install`. Nothing else.
If you ever need a GPU build of torch, that comes from a different package index
and needs a `[[tool.poetry.source]]` entry; ask before adding it, because it
changes what the lock resolves to.

## Layout

```
pyproject.toml           dependencies, tool config
poetry.lock              exact pinned versions -- commit this
Makefile                 shortcuts: make help
GUIDE.md                 the complete guide: every analysis, every figure, and
                         the number the stored runs actually produced
CROSS_PRIOR_RESULT.md    write-up of the cross-prior sweep
configs/flagship.yaml    the calibrated p_common = 0.5 network -- start here
configs/pcommon{0,1}.yaml the two controls;  pcommon{028,03,07} the satellites
configs/default.yaml     every parameter, documented;  realistic, equal_n and
                         matlab_match are earlier parameter sets, kept to compare
src/cmsi/
  data/        generative.py   generative model + analytical Bayesian observer
               encoding.py     measurements -> population codes (network input)
               dataset.py      make_dataset, split_indices, subset
  models/      network.py      the network, predict, hidden_activations
               losses.py       the objective and why it is reweighted
               training.py     training loop with early stopping
  analysis/    accuracy.py     readout vs observer, per output
               causal.py       implied weight (per trial, per disparity bin,
                               per posterior bin), position regression,
                               transition fit, variance signature, five-way
                               model comparison, strategy fit
               calibration.py  the pre-training gate (SS8 / SS4 / SS9.4)
               units.py        congruent/opposite units, balance, lesion, RF shift
               behavior.py     Kording-style bias curves
               decoding.py     what the hidden layers carry
               todo.py         placeholders for analyses not yet written
  experiments/ implied_weight.py  the implied-weight investigation (06)
  viz/         inputs.py       what the network is shown
               training.py     did it converge
               results.py      network vs observer, and the manuscript panels
               prior_sweep.py  the cross-prior figures
               manuscript.py   the standard panel every manuscript figure is on
               style.py        PLOS figure defaults, save_figures, exact_frame
  utils/       config.py  paths.py  seed.py  io.py
scripts/       00_calibrate  01_generate_data  02_train  03_analyze  04_figures
               05_prior_sweep  run_all.sh
               06_implied_weight   side experiment, writes to results/experiments/
tests/         property tests on the maths, the stage boundaries and the
               figure contract
data/          generated datasets (.npz)      contents gitignored
results/       one folder per run, plus calibration/, manuscript/ and
               prior_sweep/                   contents gitignored
```

## Run it

```bash
make calibrate CONFIG=configs/flagship.yaml   # the gate -- run this first
make all       CONFIG=configs/flagship.yaml   # the real thing (50k trials)
make quick     CONFIG=configs/flagship.yaml   # ~2 min end to end
make sweep     SEEDS="0 1 2"                  # nine priors x three seeds, ~45 min
make figures   RUN=flagship                   # redraw one run's figures
make help                                     # everything else
```

Or one stage at a time — each reads what the previous one wrote, so you can
re-run any of them alone:

```bash
poetry run python scripts/00_calibrate.py     --config configs/flagship.yaml
poetry run python scripts/01_generate_data.py --config configs/flagship.yaml --name flagship
poetry run python scripts/02_train.py         --data flagship --run flagship
poetry run python scripts/03_analyze.py       --run flagship --twin flagship_twin \
                                              --control pcommon1
poetry run python scripts/04_figures.py       --run flagship --only model
                                              # --only: inputs | training | model | manuscript
```

The cross-prior sweep is a separate experiment with its own output folder:

```bash
poetry run python scripts/05_prior_sweep.py --priors 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
                                            --seeds 0 1 2 --control pcommon1
poetry run python scripts/05_prior_sweep.py --replot                 # redraw, no training
poetry run python scripts/05_prior_sweep.py --replot --min-count 25 --max-se 0.05
                                              # re-bin the stored per-trial curves
```

It trains one network per (prior, seed), changing only the Bernoulli constant,
writes after every seed so an interrupted run leaves usable output, and keeps
the per-trial arrays of the first seed in `curves.npz` so the weight curves can
be re-binned without retraining.

Stage 0 is a gate, not a report: it exits non-zero if the config fails a design
criterion, so `run_all.sh` stops before spending a training run on a dataset
whose targets are miscalibrated. Run it on any config you edit.

`--control pcommon1` hands stage 3 the p_common = 1 network's residual spread as
`sigma_out`, which is what makes the per-trial `sigma_w = sigma_out / |Delta|`
filter meaningful. Without it stage 3 falls back to the run's own residuals,
which on the flagship also contain any causal-inference misweighting and are
about three times larger. `metrics.json` records both the value used
(`sigma_out`) and where it came from (`sigma_out_source`), and stage 4 reads the
value back rather than the run's own `residual_std`.

## Side experiments

`scripts/06_implied_weight.py` is an investigation, not a pipeline stage. It
writes everything under `results/experiments/implied_weight/<name>/` and never
touches `results/<run>/`, so it is the place to try other network sizes or
input encodings before changing the global config:

```bash
poetry run python scripts/06_implied_weight.py figures     --run flagship   # 05, 15, 16 for one run
poetry run python scripts/06_implied_weight.py sigma       --run flagship   # sigma_out and per-trial sigma_w
poetry run python scripts/06_implied_weight.py reliability --run flagship --levels 5
poetry run python scripts/06_implied_weight.py train --hidden 16 32 64 128 256 --name units
poetry run python scripts/06_implied_weight.py train --hidden 64 --set rf_width=4 --name rf4
poetry run python scripts/06_implied_weight.py compare --name units
```

`train` trains every variant together with its own p_common = 1 control, so
each gets a `sigma_out` measured at its own architecture, and then compares
the implied weight, `sigma_out` and `sigma_w` across variants. The script's
docstring documents the sub-commands and the output layout
(`src/cmsi/experiments/implied_weight.py` holds the analysis and figures).

## Tests

```bash
make test                    # 94 tests, ~45 seconds
```

They are property tests, not regression tests: each states something that must
hold for any correct implementation, so they survive changes to the parameters
or the internals. The Bayes factor falls with disparity; the posterior is
monotone in it and saturates at 0 and 1; model averaging hits the fused estimate
at p=1 and the segregated one at p=0; the mixture variance exceeds both
components when they disagree; encoders are reproducible from the seed; splits
are disjoint; a reloaded checkpoint predicts identically.

Four are worth knowing about specifically:

- `test_targets_match_brute_force_integration` checks the closed-form observer
  against a numerical integration of the full posterior over (C, source, eye)
  on a grid. It certifies the whole target derivation at once — likelihoods,
  the eye-position transform, the prior terms, the mixture variance — so if the
  targets are ever subtly wrong, this is what says so.

- `test_observer_never_touches_the_true_sources` perturbs the hidden truth and
  asserts the targets don't move. If it ever fails, the labels leak information
  the network could not have, and every result is inflated.
- `test_subset_keeps_trials_aligned` guards against comparing one trial's
  prediction to another trial's ground truth — the failure that produces
  plausible-looking nonsense rather than an error.
- `test_strategy_fit_identifies_the_strategy_that_made_the_data` plants each
  decision strategy in turn and checks the analysis names it.

Run them after touching anything in `data/` or `analysis/`.

## What a run produces

```
results/calibration/<config>/     stage 0: the gate's verdict and its figures
results/<run>/
  config.yaml        the exact parameters that produced this run
  dataset.txt        which dataset it was trained on
  model.pt           weights + config + the train/val/test split
  metrics.json       every number the analysis computed
  analysis.npz       per-trial arrays the figures are drawn from
  figures/
    inputs/          tuning curves, population heatmaps, gain, latent distributions
    training/        loss curves, per-output loss
    model/           01 output scatter, 02 errors, 03 p(C=1|x) vs disparity,
                     04 fusion weight, 05 reliability dependence, 06 decoding,
                     07 emergent vs imposed, 08 position regression,
                     09/10 variance hump, 11 five-way model comparison,
                     12 behavioural bias, 13 congruency, 14 RF shifts,
                     15/16 implied weight vs posterior by the two estimators
results/manuscript/<figure>/      the manuscript figures built from the flagship
                                  run, one folder per figure, on the standard
                                  panel; results/manuscript_<run>/ for any other run
results/prior_sweep/
  sweep.json         per-(prior, seed) rows and the per-prior aggregate
  curves.npz         per-trial arrays of the first seed, for re-binning
  figures/           prior_sweep (A weight curves, B midpoints, C slopes) and
                     variance_hump_vs_prior
  manuscript/        the same sweep figure on the standard panel
```

Every figure is written as `.png`, `.tif` and `.svg` side by side, at its
printed size and to the PLOS ONE specification (`viz/style.py`); manuscript and
sweep figures also get a `.pdf`.

Two stage boundaries earn their keep. Numbers are separated from figures, so you
can re-plot without refitting decoders and diff `metrics.json` between runs. And
the split is stored in the checkpoint, so every analysis runs on exactly the
trials the model was validated against.

## Use it from a notebook

```python
import sys; sys.path.insert(0, "src")
from cmsi.utils import load_config
from cmsi.data import make_dataset, subset
from cmsi.models import train, predict, hidden_activations
from cmsi import analysis

cfg  = load_config()
d    = make_dataset(cfg, n=20000)
model, history, splits = train(d, cfg)

test = subset(d, splits["test"])        # slices X, Y and every latent together
pred = predict(model, test["X"])
analysis.print_accuracy(analysis.accuracy(pred, test["Y"], d["target_names"]))
```

A dataset is one flat dict of numpy arrays, so masking is the natural way to ask
a sub-question:

```python
reliable = subset(test, test["sig2_vis"] < 2)
```

One naming trap worth knowing: `d["post_c1"]` is the **trial-wise posterior**
`p(C=1|x)`, while `cfg["generative"]["p_common"]` is the **prior**. They used to
share a name, which is exactly the confusion the design document warns about;
datasets written before the rename are remapped on load.

`tweak(cfg, p_common=0.8, epochs=100)` copies a config with values replaced; it
finds the key in whichever section owns it and raises on typos, which matters
when you're sweeping.

## The idea in one paragraph

Two cues report where the hand is: vision, in **retinal** coordinates, so it
needs the eye-position signal before it can be read in body coordinates; and
proprioception, already in **body** coordinates. They may share one source
(`C=1`) or come from two separate ones (`C=2`). A Bayesian observer weighs the
evidence, gets a graded posterior `p(C=1|x)`, and reports the model-averaged
estimate `p·(fused) + (1−p)·(single-cue)`. The network is trained on those four
numbers — `mu_vis, var_vis, mu_prop, var_prop` — from population codes alone. It
never sees `p(C=1)`, the disparity, the true sources, or `C`, so any causal
inference in it has to be built internally.

## What the analyses ask

**`accuracy`** — does the readout match the observer, output by output.

**`position_regression`** — the headline. Regress `(estimate − seg)` on
`w_opt · Δ` across trials; Bayes-optimal model averaging predicts slope 1,
intercept 0. No division, so every trial enters with its natural leverage and
the small-`|Δ|` trials — where behaviour *cannot* reveal the weight — carry
almost none. Read this before `fusion_weight`.

**`fusion_weight`** — solve `estimate = w·fused + (1−w)·segregated` for `w`.
Optimal averaging predicts `w == p(C=1|x)`, so `w` against disparity is the
fusion→segregation transition, and `transition_fit` gives its midpoint and
sharpness. Trials where the two references nearly coincide are dropped: the
denominator goes to zero there and a handful of trials would otherwise dominate
every summary. `sigma_w` makes that principled — per-trial `sigma_out / |Δ|`,
filtered at a stated criterion. `joint_fusion_weight` pools both outputs, which
share one `w`; their agreement on well-conditioned trials is an internal
coherence test. Per-trial R² on `w` looks bad even when the binned curve tracks
the optimum closely — single-trial `w` is a noisy ratio, so read the slope and
the curve, not R².

**`binned_implied_weight` / `implied_weight_by_posterior`** — the weight
*within a bin*, by least squares through the origin:
`w_bin = Σ Δ·(estimate − seg) / Σ Δ²`, with `SE = sigma_out / √ΣΔ²`. This is
the estimator behind every binned weight curve, including the prior sweep. The
intuitive alternative — filter the per-trial ratio by `sigma_w` and average it
within the bin — is biased low, because the filter keeps preferentially
large-`|Δ|` trials and within a bin those are the lowest-weight ones.
`implied_weight_by_posterior` computes both estimators side by side so the bias
can be seen (figures 15 and 16); the filtered one also loses almost every trial
above a posterior of 0.6, where `|Δ|` is small.

**`transition_fit`** — a logistic in `|disparity|` fitted to a weight curve,
returning the midpoint (where fusion gives way to segregation) and the
sharpness. The fit is guarded: it returns NaN unless the sharpness is positive
and the midpoint falls within a quarter-span of the observed disparity range,
so a flat curve reports "no transition" rather than a fabricated midpoint.

**`reliability_within_disparity`** — the Bayes-vs-heuristic test. At matched
disparity the optimal weight still varies with the cue reliabilities; a pure
disparity heuristic predicts a within-bin slope of 0, Bayes predicts 1.

**`variance_signature`** — the uncertainty channel. The full mixture variance
carries a between-component term `w(1−w)(fused − seg)²` that humps at
intermediate ambiguity; no fixed-weight model can produce it. This is the
analysis that covers the ambiguous zone where the weight recovery is blind.

**`model_comparison`** — network against averaging, full integration, full
segregation, model selection, and the best fixed-weight model, binned by
posterior decile. Per bin the implied weight is a least-squares slope, not a
mean of signed biases — the latter cancels within a bin because Δ is signed.
Averaging tracks the posterior smoothly; selection steps at 0.5.

**`congruency` / `balance` / `lesion`** — classify MSL units by the correlation
of their visual- and proprioceptive-sweep tuning (Rideaux's congruency logic),
then ask whether the congruent−opposite activity balance carries `p(C=1|x)`,
and what the readout loses when each subpopulation is silenced.

**`bias_vs_disparity` / `conditioned_bias`** — the Körding Fig. 2e and 3b–c
analogs, the second conditioned on the network's own causal judgment.

**`decode` / `decode_by_layer`** — ridge-decode a latent from a hidden layer.
The causal target is `post_c1`, the **trial-wise posterior** `p(C=1|x)` — never
the prior `p_common`, which is one number per network and whose "decoding" would
just read out a disparity confound.
Independent of the readout, so it asks "is this represented?" rather than "was it
trained to output this?". Weak in layer0 and strong in the last layer means the
network is building it. Running it on the always-fuse twin, which was never asked
for anything causal, is the emergent-vs-imposed test.

**`strategy_fit`** — model averaging vs model selection vs probability matching.

`analysis/todo.py` holds one-line placeholders for the two analyses still to be
designed: per-unit additivity indices and population geometry.

## Two things that matter for training

`standardize_inputs` and `balance_loss` are both on by default. The three input
groups differ ~30× in magnitude and the outputs live on different scales (means
≈ ±10 deg, variances ≈ 5–25 deg²). Without both, the shared trunk learns
`var_prop` fine and starves `mu_prop`, even though a dedicated decoder reaches
R²≈0.97 on it. `figures/training/02_per_output_loss.png` is where that shows up.
Use Adam, not Rprop — Rprop is a full-batch method and misbehaves on mini-batches
(`check` in `utils/config.py` warns if you configure that combination).

## History

This grew out of an earlier `causal_msi` package. The observer maths was ported
line by line and verified against it on identical draws — worst disagreement
2e-13, i.e. floating-point noise — before that package was retired; the property
tests in `tests/test_generative.py` now guard the same equations without needing
the old code present.

Dropped along the way: a pydantic config layer, a typer CLI with `scripts/`
wrappers around it, frozen dataclasses wrapping every group of arrays
(`LatentBatch`, `Measurements`, `ObserverTargets`, `Dataset`, `Encoders`, …), and
a nine-module `analysis/` split. Added: the four-stage pipeline, the per-run
results convention, and the input and training figure groups.
