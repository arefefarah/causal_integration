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
  experiments/ implied_weight.py  the implied-weight analyses and figures (06)
               design.py          readability of the weight at the level of the trials:
                                  screening before training, runs side by side (06 design / compare)
               architecture.py    variants at other hidden sizes, and their comparison (07)
               fixed_variance.py  variants with each input's noise pinned to one value (08)
               sweep.py           variants over the values of any one config key (09)
  viz/         inputs.py       what the network is shown
               training.py     did it converge
               results.py      network vs observer, and the manuscript panels
               prior_sweep.py  the cross-prior figures
               manuscript.py   the standard panel every manuscript figure is on
               style.py        PLOS figure defaults, save_figures, exact_frame
  utils/       config.py  paths.py  seed.py  io.py
scripts/       00_calibrate  01_generate_data  02_train  03_analyze  04_figures
               05_prior_sweep  run_all.sh
               06_implied_weight   implied-weight analyses; `train` runs a config through the pipeline
               07_architecture     the same at other hidden sizes -> results/experiments/architecture/
               08_fixed_variance   every input at one fixed noise level -> results/experiments/fixed_variance/
               09_sweep            one config key over several values   -> results/experiments/sweep/
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

## Experiments

Four scripts investigate the implied weight outside the numbered stages.
They never change `results/<run>/`: an analysis of a run writes only under
`results/experiments/`, and the three that train networks keep their networks
there too. When a configuration turns out to be the one you want, it goes into
`configs/` and the pipeline is re-run through the numbered stages.

| script | what it does | writes to |
|---|---|---|
| `06_implied_weight.py` | `figures`, `sigma`, `reliability`: analyses of any pipeline run. `compare`: several runs side by side. `design`: what a configuration's trials allow, before training. `train`: a new configuration through the whole pipeline | `results/experiments/implied_weight/<name>/`; `train` → `results/<name>/` |
| `07_architecture.py` | the same task at other hidden-layer sizes, compared | `results/experiments/architecture/<name>/` |
| `08_fixed_variance.py` | each input's noise pinned to one value, swept over values, compared | `results/experiments/fixed_variance/<name>/` |
| `09_sweep.py` | one config key (a prior width, a noise range, anything) over several values, each with its own datasets and control, compared | `results/experiments/sweep/<name>/` |

All four share the analysis and figures in `src/cmsi/experiments/implied_weight.py`;
`design.py` holds the trial-level readability analysis they all report.

### Rules that apply to all of them

- **Every configuration gets its own p_common = 1 control.** σ_out (the
  read-out noise that sets σ_w and every error bar) is measured on a control
  trained with the same encoding, noise ranges and architecture. The flagship's
  `pcommon1` is right for the satellites; it is wrong for anything that changes
  the encoding, the noise or the network, which is why `06 train`, `07` and
  `08` all train one control per configuration. The analyses find the control
  by themselves: `--control` can be left out and is read from the run's
  `metrics.json` (stage 3 records it as `sigma_out_source`), falling back to
  `pcommon1`.
- **Nothing is filtered.** The σ_w criterion appears in the figures only as a
  reference line, and figure 15 exists to show what applying it *would* do next
  to figure 16, which uses every trial. The one guard that leaves data out is
  the per-bin SE rule on least-squares curves: a bin whose weight is not
  identifiable is left blank and the line breaks there.
- **`--set key=value`** (repeatable) changes any config key by name, in whichever
  section owns it: `--set rf_width=4 --set n_vis=100`. Quote values with
  brackets, because zsh treats `[ ]` as a glob: `--set 'sigma2_vis_range=[1,4]'`.
- **`--config`** picks the base config (default `configs/flagship.yaml`);
  **`--quick`** is 8,000 trials and 60 epochs; **`--n`**, **`--epochs`**,
  **`--seed`** override the config.
- **Datasets are cached per experiment name.** `07`, `08` and `09` draw their
  datasets once under `data/experiments/<experiment>/` and reuse them for every
  later variant of the same `--name`, so variants stay comparable (`08` and
  `09` draw one pair per variant, since their variants change the generative
  model). Asking for a
  different config under an existing name stops with a message; use a new
  `--name`, or `--regen` to redraw (which warns if variants already exist).
- **`compare --name a b c`** puts the variants of several experiments on one
  set of figures, labelled `<experiment>/<variant>`, written to a
  `compare_a+b+c/` folder so nothing inside any one experiment is overwritten.

### `06_implied_weight.py` — analyses of a run

```bash
poetry run python scripts/06_implied_weight.py figures     --run flagship
poetry run python scripts/06_implied_weight.py sigma       --run flagship
poetry run python scripts/06_implied_weight.py reliability --run flagship --levels 5
poetry run python scripts/06_implied_weight.py reliability --run flagship --inputs vis prop eye \
                                                           --levels 1.5 2.5 3.5 4.5 5.5 6.5
```

Common flags: `--run` (default `flagship`), `--control` (see above), `--name`
(experiment folder, default the run name). Output goes to
`results/experiments/implied_weight/<name>/figures/<sub-command>/` plus one
block per sub-command in `metrics.json`.

- **`figures`** — the per-run figures this investigation grew out of: `04`
  fusion weight vs disparity, `05` by reliability level (the pipeline's
  version), `15` weight vs posterior by the σ_w-filtered ratio, `16` the same by
  least squares on every trial — and the pipeline's `04` and `08` with the
  weight read from the variance outputs instead of the position outputs:
  `04v_fusion_weight_variance_read` (the transition curve on every trial,
  through zero disparity, with the pipeline's ratio curves faded for
  comparison) and `08v_variance_regression` (A: the no-division headline
  statistic in the variance domain, `var_out − var_seg` against
  `v_opt − var_seg`, slope 1 = optimal; B: the variance read against the
  posterior with its fit). The console prints the per-trial table and both
  regressions; `metrics.json` carries them under `per_trial`.
- **`sigma`** — seven figures, all unfiltered: `01` the run's per-trial residuals
  (network − analytical) on every test trial, with the control's as an outline
  and both stds; `02` |error| and σ_w against |Δ|; `03` σ_w per posterior bin
  and the fraction the criterion would keep; `04` fraction kept as the criterion
  is varied; `05` the distribution of the per-trial weight `(network − seg)/Δ`
  on every trial, against the analytical posterior's; `06` the distribution of
  `network − seg` next to `Δ = fused − seg`; `07` the per-trial weight grouped
  by |Δ|, with each group's share of the trials, of the least-squares weight
  (ΣΔ²) and of the out-of-range ratios; `08` the weight on every trial read
  four ways (below). The console also prints the `readability` block and the
  per-trial table.
- **`reliability`** — per input (`--inputs vis prop eye`), the least-squares
  weight curve at each reliability level read from `mu_vis` and from `mu_prop`,
  with the analytical posterior at that level dashed, and the transition
  midpoint against the level. `--levels N` makes N quantile bins (equal trial
  counts); a list of numbers makes nearest-centre levels like the config.

### The weight on every trial: the variance outputs

The ratio `(network − seg)/Δ` divides by Δ, so it cannot be read where the
two hypotheses nearly coincide — the confident-fusion trials, exactly where
the weight is highest. The network's *variance* outputs carry the same weight
without that blind spot. A model-averaging observer's variance is the mixture
variance, `v(w) = w·var_fus + (1 − w)·var_seg + w(1 − w)Δ²`, so a variance
output can be solved for the weight the network used (a quadratic;
`implied_weight.variance_weight`). Its sensitivity `|dv/dw| = |Δ²(1 − 2w) − c|`
with `c = var_seg − var_fus` does not vanish at Δ = 0 — there it equals `c`,
the variance fusion saves — so the read stays sharp on every trial. The two
roots that exist when Δ² > c are told apart by the position read, which is
precise on exactly those large-|Δ| trials; a variance above any mixture is
flagged. `weight_analysis` now returns a `per_trial` block with, on every
trial, the ratio from each position output, the least-squares read across
both position outputs (`joint`: `Σ_k Δ_k(est_k − seg_k) / Σ_k Δ_k²`, sd
`σ_out/√(Δ_vis² + Δ_prop²)`), the read from both variance outputs
(`variance`) and the inverse-variance combination of all four (`combined`),
each with its fraction outside [−1, 2], its sd against the analytical
posterior (all trials, |Δ| < 1°, |Δ| > 4°), its correlation with the
posterior, its per-trial σ_w and its binned mean per posterior decile — plus a
`coherence` table: per posterior bin, the position read and the variance read
on the trials where the position read is precise. A network that mixes both
outputs with one weight puts that comparison on the identity; where it leaves
it, the network reports fusion in its uncertainty that it does not perform in
its estimate (or the reverse). On the runs analysed so far the variance read
puts 0 % of trials outside [−1, 2], sits within sd 0.03–0.11 of the posterior
on the |Δ| < 1° trials, and agrees with the position read to within 0.01 on
average. Figure `sigma/08_weight_from_variance_all_trials` shows the four
distributions, the variance read trial by trial, the per-trial σ_w of the
position and variance reads against |Δ|, and the coherence table.

### `06_implied_weight.py compare` — trained runs side by side

```bash
poetry run python scripts/06_implied_weight.py compare --runs flagship exp2 exp2_128units exp2_128_mean
```

Runs the same analysis on every run named (each with its own control, found
as above) and puts them on one set of figures under
`results/experiments/implied_weight/compare_<runs>/figures/compare/`: `A` the
fraction of per-trial ratios outside [−1, 2] observed, beside what a Gaussian
error of the run's own residual sd would give, what σ_out alone would give (an
optimal network with that read-out noise) and the observed fraction re-weighted
to a flat posterior histogram; `B` σ_out, the run's own residual sd and the
decoding floor, with the position-regression slope; `C` the distribution of
|Δ| / own sd — the dimensionless quantity the ratio depends on; `D` the
posterior histograms; `E`, `F` the least-squares weight against posterior and
disparity, one curve per run. The console table has the same numbers per run.

The identity behind these: on a network that is Bayes-optimal up to a position
error *e*, the per-trial ratio is exactly `w = p + e/Δ`, so the spikes in
figure 05 are decided by two things only — how the trials distribute |Δ| (a
property of the configuration, fixed before training) and how large *e* is
(a property of the trained network). Every run's `metrics.json` block from
`weight_analysis` now carries a `readability` entry per read with these
pieces: `own_sd` (the run's error, which exceeds σ_out by its misweighting),
`frac_absdelta_lt1` (the trials no network can be read on), `outside`,
`outside_predicted_own_sd`, `outside_predicted_sigma_out`,
`outside_if_flat_posterior`, `outside_share_from_absdelta_lt1`, and Δ and the
error by true cause. `--no-floor` skips the decoding floor, the slow part.

### `06_implied_weight.py design` — what a configuration allows, before training

```bash
poetry run python scripts/06_implied_weight.py design --config configs/flagship.yaml
poetry run python scripts/06_implied_weight.py design --param sigma0_sq --values 100 169 425                                                       --set 'hidden=[128,128]' --name s0_design
poetry run python scripts/06_implied_weight.py design --config configs/exp2_128_mean.yaml                                                       --param sigma2_prop_range --values '[2,2.5]' '[8,9]'
poetry run python scripts/06_implied_weight.py design --config a.yaml b.yaml --mark flagship exp2_128_mean
poetry run python scripts/06_implied_weight.py design --config configs/flagship.yaml --balance 0.5
```

Draws the trials of each configuration (no network) and reports what they
allow: `01` the posterior histogram with the calibration masses; `02` the
distribution of |Δ| per read; `03` the fraction of ratios that would fall
outside [−1, 2] against the read-out error sd — solid as drawn, dashed
re-weighted to a flat posterior — and the fraction the σ_w criterion would
keep, with each configuration's **decoding floor** dotted (the error an ideal
decoder of the spike counts would make: no network's σ_out can be below it)
and, with `--mark`, the errors trained runs actually achieved as grey lines to
read the curves at. This is the screening step of a systematic range test:
which values of a key are worth training on, read off one figure. `--config`
takes one or more yamls, `--set` applies to all of them, `--param/--values`
varies one key of the first; `--n` draws that many trials (default the
config's `n_trials`, at most 20,000). `--balance KEEP` thins the trials to a
flat posterior histogram first (see 09) and prints what the anti-confound
audit says about the thinned set.

### `06_implied_weight.py train` — a new configuration through the pipeline

```bash
poetry run python scripts/06_implied_weight.py train --config configs/exp_weight.yaml --name lownoise
poetry run python scripts/06_implied_weight.py train --set 'sigma2_vis_range=[1.2,1.8]' --name lownoise
poetry run python scripts/06_implied_weight.py train --config configs/realistic.yaml --name real --quick
```

Runs stages 0–4 exactly as `run_all.sh` does, into `results/<name>/`, and
also trains a p_common = 1 control of the *same* configuration into
`results/<name>_pcommon1/`, analysed first so stage 3 can use it. The run then
has everything a pipeline run has (config, checkpoint, `metrics.json`,
`analysis.npz`, all figure groups), every analysis in the project applies to
it, and the configs it used are written to `configs/experiments/<name>.yaml`
and `<name>_pcommon1.yaml` — so keeping it later is
`make all CONFIG=configs/experiments/<name>.yaml`. The calibration gate is run
and reported but does not stop an experiment; read its verdict in
`results/calibration/<name>/`. Flags: `--twin` also trains the always-fuse twin
(needed for figure 07), `--force` retrains into an existing name.

```bash
poetry run python scripts/06_implied_weight.py sigma --run lownoise     # control found automatically
```

### `07_architecture.py` — network size

```bash
poetry run python scripts/07_architecture.py train --hidden 16 32 64 128 256 --name units
poetry run python scripts/07_architecture.py train --hidden 32x128 128x32   --name asym
poetry run python scripts/07_architecture.py train --hidden 32 64 --config configs/exp_weight.yaml \
                                                   --set 'sigma2_vis_range=[1.2,1.8]' --name lownoise_units
poetry run python scripts/07_architecture.py compare --name units lownoise_units
```

A hidden size is `N` (both layers) or `AxB` (SIL x MSL). For each size the
causal network and its control are trained on the experiment's shared
datasets, the per-variant figures are drawn (`04`, `05`, `15`, `16`, the six
`sigma_*` figures and the `reliability_*` figures; `--inputs`, `--levels` as
in 06), and the variants are compared: `A` weight vs disparity, `B` weight vs
posterior, `C` σ_out and σ_w, `D` transition midpoint (both estimators),
position-regression slope, read-out R² and fraction readable against units.

### `08_fixed_variance.py` — every input at one noise level

```bash
poetry run python scripts/08_fixed_variance.py train --vis 3 --prop 5.7 --eye 8.25 --name mid
poetry run python scripts/08_fixed_variance.py train --vis 1.5 3 6 --name vis_sweep
poetry run python scripts/08_fixed_variance.py train --vis 1.5 3 6 --prop 2 8 --name grid
poetry run python scripts/08_fixed_variance.py compare --name vis_sweep grid
```

Pins each input's measurement variance to a single value instead of a range —
`sigma2_vis_range: [v, v]`, which every stage accepts unchanged — so
reliability is not a variable at all. Values are in deg², like the ranges in
the config; an input not given keeps the midpoint of its range in the base
config (3.9 / 5.7 / 8.25 for the flagship). Every combination of the values
given becomes one variant (`v3_p5.7_e8.25`) with its own datasets and its own
control, gets the per-variant figures (`04`, `05`, `15`, `16`, `sigma_*`; no
reliability figures, since reliability is constant), and the variants are
compared with the same `A`–`D` figures as 07, each variant's own analytical
curve dashed in its colour. The calibration gate would fail its
reliability-coverage check on such a dataset by construction, so it is not
run here; if you take a fixed-variance config through the full pipeline with
`06 train --set 'sigma2_vis_range=[3,3]' …` instead, expect that check to say
FAIL and ignore it.

### `09_sweep.py` — one config key, several values

```bash
poetry run python scripts/09_sweep.py train --param sigma0_sq --values 100 169 250 425                                             --set 'hidden=[128,128]' --name s0
poetry run python scripts/09_sweep.py train --param sigma2_prop_range --values '[2,2.5]' '[4,4.5]' '[8,9]'                                             --config configs/exp2_128_mean.yaml --name propnoise
poetry run python scripts/09_sweep.py train --param sigma0_sq --values 100 425 --balance 0.5 --name s0_flat
poetry run python scripts/09_sweep.py train --param rf_width --values 4 6 9 --name rf --quick
poetry run python scripts/09_sweep.py compare --name s0 s0_flat
```

The systematic range test with training: any key `utils.tweak` can reach —
`sigma0_sq`, `eye_sigma_sq`, a `sigma2_*_range`, `hidden`, `rf_width`,
`gain_K`, `epochs`, `lr` — over the values given (parsed as yaml, so lists
work; quote them), everything else held at the base config. Each value
becomes one variant (`sigma0_sq=100`) with its own datasets under
`data/experiments/sweep/` and its own control, gets the per-variant figures
(`04`, `05`, `15`, `16`, `sigma_*`, `reliability_*`), and the variants are
compared with the `A`–`D` figures of 07 plus `E`, the readability of the
per-trial ratio per variant (observed and predicted fraction outside [−1, 2],
own sd / σ_out, share of trials with |Δ| < 1°). `06 design` screens the same
values without training first.

`--balance KEEP` keeps a fraction KEEP of every variant's causal trials,
chosen so that the histogram of the analytical posterior is as flat as that
allows (`design.balance_posterior`; the control is left alone). The rule sees
a trial only through its posterior, a function of the measurements, so the
acceptance cancels in p(C | x): the targets stay exactly right and no cue to
C is created that the observer does not already use — what changes is how
many ambiguous trials the network trains on. It is the safe reading of "a
balanced weight distribution". Two things to know before using it: the
calibration gate is not run on these variants, and the single-channel AUCs
of its anti-confound audit do move on the thinned set (they are marginal
statistics of a distribution selected on the posterior; `06 design --balance`
prints them), which is expected and is not a leak — but a run trained this
way should be described as trained on posterior-balanced trials.

### What the experiments write

```
configs/experiments/<name>.yaml, <name>_pcommon1.yaml   the configs 06 train used
data/experiments/<experiment>/                          the datasets 07, 08 and 09 trained on
results/experiments/
  implied_weight/<name>/
    metrics.json           one block per sub-command
    figures/figures/       04, 04v, 05, 08v, 15, 16
    figures/sigma/         01-08
    figures/reliability/   one pair of figures per input
    figures/design/        01-03 (design; <name> defaults to "design")
  implied_weight/compare_<runs>/
    figures/compare/       A-F (compare)
  architecture/<name>/,  fixed_variance/<name>/  and  sweep/<name>/
    config.yaml            the base configuration
    metrics.json           the comparison table
    figures/compare/       A, B, C, D, E
    variants/<variant>/    model.pt, control.pt, config.yaml, metrics.json,
                           arrays.npz, figures/
  <experiment>/compare_<a>+<b>/   a comparison across experiments
```

## Tests

```bash
make test                    # 104 tests, ~45 seconds
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
