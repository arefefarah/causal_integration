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
configs/optimal_sweep.yaml the configuration the readability sweeps converged on
                         (IMPLIED_WEIGHT_RESULT.md): small domain, relu 256x256,
                         gain_K 720, lr 3e-4, 100k trials; run it with 06 train
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
               causal.py       the implied weight (the hybrid read of each
                               channel, on every trial), its binnings by
                               disparity and by posterior, the position- and
                               variance-domain regressions, transition fit,
                               variance signature, five-way model comparison,
                               strategy fit
               calibration.py  the pre-training gate (SS8 / SS4 / SS9.4)
               units.py        congruent/opposite units, balance, lesion, RF shift
               behavior.py     Kording-style bias curves
               decoding.py     what the hidden layers carry
               todo.py         placeholders for analyses not yet written
  experiments/ implied_weight.py  the implied-weight figures of a run, the weight per
                                  reliability level, and `train` (06)
  viz/         inputs.py       what the network is shown
               training.py     did it converge
               results.py      network vs observer, and the manuscript panels
               prior_sweep.py  the cross-prior figures
               manuscript.py   the standard panel every manuscript figure is on
               style.py        PLOS figure defaults, save_figures, exact_frame
  utils/       config.py  paths.py  seed.py  io.py
scripts/       00_calibrate  01_generate_data  02_train  03_analyze  04_figures
               05_prior_sweep  run_all.sh
               06_implied_weight   the weight figures and the reliability analysis of a run;
                                   `train` runs a config through the pipeline with its control
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
`sigma_out`, the read-out noise per output, which sets the nominal per-trial
uncertainty of the implied weight. Without it stage 3 falls back to the run's
own residuals, which on the flagship also contain any causal-inference
misweighting and are about three times larger. `metrics.json` records both the
value used (`sigma_out`) and where it came from (`sigma_out_source`).

**How the weight is read.** The network never outputs a weight; on every trial
it is *implied* from the outputs of one channel by the hybrid read
(`analysis.hybrid_weight`). A model-averaging observer's variance is the
mixture variance `v(w) = w·var_fus + (1 − w)·var_seg + w(1 − w)Δ²`, a
downward parabola in `w` whose peak sits at `½ − c/(2Δ²)` with
`c = var_seg − var_fus`. Where `Δ² ≤ c` (which includes zero disparity) the
parabola is monotone on [0, 1], the channel's variance output has one root,
and that root is the weight; where `Δ² > c` the variance would have two roots,
so the weight is the channel's own position ratio `(mu − seg)/Δ`, well
conditioned exactly there (`|Δ| > √c`). Every trial gets a weight, nothing is
filtered or clipped, and each trial's flag says which output it came from
(`hybrid_flags_vis/prop` in `analysis.npz`). Stage 3 stores the visual and
the proprioceptive read as `fusion_weight` and `fusion_weight_prop` with their
nominal sd (`σ_out(var)/|dv/dw|` on root trials, `σ_out(mu)/|Δ|` on ratio
trials), and every downstream analysis that needs a weight — the transition
fit, the reliability-within-disparity test, the behavioural conditioning, the
reliability curves, the prior sweep — uses it.

## Experiments

One script investigates the implied weight outside the numbered stages. It
never changes `results/<run>/`: an analysis of a run writes only under
`results/experiments/implied_weight/<name>/`. When a configuration turns out to
be the one you want, it goes into `configs/` and the pipeline is re-run
through the numbered stages.

### `06_implied_weight.py` — analyses of a run

```bash
poetry run python scripts/06_implied_weight.py figures     --run flagship
poetry run python scripts/06_implied_weight.py reliability --run flagship --levels 5
poetry run python scripts/06_implied_weight.py reliability --run flagship --inputs vis prop eye \
                                                           --levels 1.5 2.5 3.5 4.5 5.5 6.5
```

Common flags: `--run` (default `flagship`), `--control` (the p_common = 1 run
whose residuals give σ_out; left out, it is read from the run's `metrics.json`,
where stage 3 recorded it as `sigma_out_source`, falling back to `pcommon1`),
`--name` (experiment folder, default the run name). Output goes to
`results/experiments/implied_weight/<name>/figures/<sub-command>/` plus one
block per sub-command in `metrics.json`.

- **`figures`** — the pipeline's weight figures for one run, exactly as stage
  4 draws them: `04` the weight against disparity per channel, `05` by
  reliability level, `08v` the variance-domain regression beside the weight on
  the posterior, `15` the weight in posterior bins, `16` its distribution on
  every trial against the posterior's. The console prints, per channel, the
  share of trials read from the position ratio, the fraction outside [−1, 2],
  the sd against the posterior (and its root and ratio parts), the weight's
  regression on the posterior, the two headline regressions and the
  transition midpoint, plus the vis–prop consistency.
- **`reliability`** — per input (`--inputs vis prop eye`), the weight against
  disparity at each reliability level, per channel (mean and 95 % interval per
  bin), with the analytical posterior at that level dashed, and the transition
  midpoint against the level. `--levels N` makes N quantile bins (equal trial
  counts); a list of numbers makes nearest-centre levels like the config.

Nothing is filtered anywhere: the hybrid read exists on every trial. The one
guard that leaves data out of a *figure* is the per-bin rule on the curves: a
bin with too few trials, or whose mean is too uncertain, is left blank and the
line breaks there.

### `06_implied_weight.py train` — a new configuration through the pipeline

```bash
poetry run python scripts/06_implied_weight.py train --config configs/exp_weight.yaml --name lownoise
poetry run python scripts/06_implied_weight.py train --set 'sigma2_vis_range=[1.2,1.8]' --name lownoise
poetry run python scripts/06_implied_weight.py train --config configs/realistic.yaml --name real --quick
```

Runs stages 0–4 exactly as `run_all.sh` does, into `results/<name>/`, and
also trains a p_common = 1 control of the *same* configuration into
`results/<name>_pcommon1/`, analysed first so stage 3 can use it. σ_out must
be measured on a control trained with the same encoding, noise ranges and
architecture — the flagship's `pcommon1` is right for the satellites and wrong
for anything else, which is why every configuration gets its own. The run
then has everything a pipeline run has (config, checkpoint, `metrics.json`,
`analysis.npz`, all figure groups), every analysis in the project applies to
it, and the configs it used are written to `configs/experiments/<name>.yaml`
and `<name>_pcommon1.yaml` — so keeping it later is
`make all CONFIG=configs/experiments/<name>.yaml`. The calibration gate is run
and reported but does not stop an experiment; read its verdict in
`results/calibration/<name>/`. Flags: `--set key=value` (repeatable) changes
any config key by name, in whichever section owns it — quote values with
brackets, because zsh treats `[ ]` as a glob; `--config` picks the base config
(default `configs/flagship.yaml`); `--quick` is 8,000 trials and 60 epochs;
`--n`, `--epochs`, `--seed` override the config; `--twin` also trains the
always-fuse twin (needed for figure 07); `--force` retrains into an existing
name.

```bash
poetry run python scripts/06_implied_weight.py figures --run lownoise     # control found automatically
```

### What the experiment writes

```
configs/experiments/<name>.yaml, <name>_pcommon1.yaml   the configs 06 train used
results/experiments/implied_weight/<name>/
    metrics.json           one block per sub-command
    figures/figures/       04, 05, 08v, 15, 16
    figures/reliability/   one pair of figures per input
```

The configuration sweeps that led to the current read (hidden sizes, fixed
variances, one config key over several values, the trial-level readability
screening) were retired on 2026-09-29 once the approach was settled; their
results stay under `results/experiments/{architecture,fixed_variance,sweep}/`
and `data/experiments/`, and the story is in `IMPLIED_WEIGHT_RESULT.md`.

## Tests

```bash
make test                    # 97 tests, ~45 seconds
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
                     04 the implied weight vs disparity per channel,
                     05 reliability dependence, 06 decoding, 07 emergent vs
                     imposed, 08 position-domain regression, 08v variance-
                     domain regression + the weight on the posterior,
                     09/10 variance hump, 11 five-way model comparison,
                     12 behavioural bias, 13 congruency, 14 RF shifts,
                     15 the weight in posterior bins, 16 its distribution on
                     every trial, per channel, against the posterior's
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
almost none. `variance_regression` is the same headline in the variance
domain: `var_out − var_seg` on `v_opt − var_seg` (Eq. 10), slope 1 for the
optimal mixture.

**`hybrid_weight`** — the implied weight of one channel on every trial: the
root of the mixture-variance parabola of the channel's variance output where
that root is unique (`Δ² ≤ c`, including zero disparity), the channel's own
position ratio `(estimate − seg)/Δ` where the variance has two roots
(`Δ² > c`, where the ratio is well conditioned). Optimal averaging predicts
`w == p(C=1|x)`, so `w` against disparity is the fusion→segregation transition,
`transition_fit` gives its midpoint and sharpness, `weight_regression` its
slope on the posterior, `weight_consistency` the agreement of the two
channels' reads trial by trial, and `variance_weight` is the solver the
hybrid calls. Nothing is filtered: the nominal per-trial sd
(`σ_out(var)/|dv/dw|` or `σ_out(mu)/|Δ|`) is reported, not applied.

**`binned_weight` / `weight_by_posterior`** — the weight averaged within a
bin of disparity (on a grid) or of the posterior, with the standard error of
the bin mean; every trial enters. This is what every binned weight curve is
drawn from, including the prior sweep's. A bin with too few trials, or (given
`max_se`) too uncertain a mean, is dropped and the line breaks there.

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
posterior decile, from the position output alone (its own per-bin
least-squares weight, no division). Averaging tracks the posterior smoothly;
selection steps at 0.5.

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

The implied weight went through three readings (September 2026, the story in
`IMPLIED_WEIGHT_RESULT.md`): the per-trial position ratio with a σ_w filter,
which is blind where the two hypotheses coincide; the mixture-variance root of
the variance outputs, sharp exactly there but ambiguous at large disparity;
and the hybrid of the two per channel, which is what the code reads now. The
ratio, the σ_w filter, the least-squares per-bin weight, the joint read across
both position outputs, the variance-only read and the configuration sweeps
that led here (`07_architecture`, `08_fixed_variance`, `09_sweep`, `06 sigma /
compare / design`) were removed on 2026-09-29.
