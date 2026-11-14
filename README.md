# causal_integration

Code and figures for

> Farahmandi A, Blohm G. *Causal inference in scalable feedforward networks
> for multisensory integration across reference frames.*

A feedforward network is trained to report the position and the uncertainty
of a seen object and of the hand from three population-coded signals: vision
in retinal coordinates, proprioception of the hand in body coordinates, and
proprioception of eye orientation. On each trial the seen object either is
the hand or is not, and the network is never told which. The question the
code answers is whether a network trained only on four numbers, the four
model-averaged estimates of a Bayesian observer, comes to weight fusion
against segregation by the posterior probability of a common cause, how that
weighting depends on the prior, and where in the network the decision is made.

## Setup

Python 3.10–3.13 and [Poetry](https://python-poetry.org/docs/#installation).

```bash
poetry install          # or: make setup
poetry run pytest       # or: make test
```

`poetry install` creates `.venv/` inside the project and installs the exact
versions recorded in `poetry.lock`, which is committed for that reason.
Every command below runs as `poetry run python …`, or directly after
`poetry shell`; the `Makefile` wraps the common ones (`make` lists them).

PyTorch is needed by the pipeline stages (training, and reading the
checkpoint in stages 3 and 4); the unit-level and transformation analyses
run from a saved checkpoint in numpy (`scripts/neuronal/npmodel.py`
reimplements the forward pass), so they need no torch.

## The pipeline

One configuration is one run. The stages write files, so any stage can be
rerun on its own, and `scripts/run_all.sh` chains them for a configuration
together with its always-fuse twin (the same network trained on fused
targets only, the control for the decoding analyses):

```bash
bash scripts/run_all.sh --config configs/flagship.yaml       # = make all
bash scripts/run_all.sh --config configs/flagship.yaml --quick
```

| stage | script | what it does | writes |
|---|---|---|---|
| 0 | `00_calibrate.py --config C` | draws a dataset from the configuration and checks that it can support the analyses (realized prior, posterior histogram, reliability coverage, hypothesis separation, anti-confound audit, range containment, Poisson validity); a FAIL stops the pipeline | `results/calibration/<config>/` |
| 1 | `01_generate_data.py --config C --name D` | samples trials from the generative model, runs the analytical observer, encodes the measurements as Poisson population codes | `data/<D>.npz` (carries its config) |
| 2 | `02_train.py --data D --run R` | trains the network (`--seed k` changes the initialisation and the split, nothing else) | `results/<R>/model.pt`, `config.yaml` |
| 3 | `03_analyze.py --run R [--twin T] [--control pcommon1]` | compares the network with the observer on the stored test split: read-out accuracy, the implied fusion weight on every trial, transition fit, regressions, variance signature, model comparison, decoding, unit classes, lesions, RF shifts | `results/<R>/metrics.json`, `analysis.npz` |
| 4 | `04_figures.py --run R [--only model\|manuscript]` | every figure of the run, and the manuscript figures built from it | `results/<R>/figures/`, `results/manuscript/` |
| 5 | `05_prior_sweep.py --priors … --seeds …` | one network per prior per seed, the Bernoulli constant being the only difference; the midpoint of each network's transition against the analytical one | `results/prior_sweep/`, Fig. 6 |
| 7 | `07_eye_sweep.py --param eye_sd\|eye_mu` | one network per eye-position spread (or mean), the reference-frame transformation being the only difference | `results/eye_sweep/`, `figS_eye_sweep` |
| 8 | `08_behavioral.py [analysis] --run R` | the behavioral analyses of a trained run, read from its outputs alone: the relative bias against true disparity at two priors, the hand bias by inferred cause, the bias, one-cause share and spread on probe trials of fixed true disparities; no PyTorch needed | `results/<R>/figures/model/behavioral_analysis/`, Fig. 4 |

`06_implied_weight.py` is not a stage: it draws the implied-weight figures of a
run at several reliability levels and can push a new configuration through
the whole pipeline (`train --set key=value`).

The analysis block of a configuration (`disparity_grid`,
`reliability_levels`) is not a training parameter: stages 3 and 4 take it
from `--config` when given, else from `results/<run>/config.yaml`, else from
the checkpoint, so a change of binning never needs retraining.

### The paper's networks

```bash
make calibrate                                   # the gate on the flagship
bash scripts/run_all.sh --config configs/pcommon1.yaml    # the p = 1 control FIRST:
                                                          # its residuals are the read-out
                                                          # noise every other run uses
bash scripts/run_all.sh --config configs/flagship.yaml    # the main network, p = 0.5
bash scripts/run_all.sh --config configs/pcommon0.yaml    # the p = 0 control
bash scripts/run_all.sh --config configs/pcommon03.yaml   # the satellites
bash scripts/run_all.sh --config configs/pcommon07.yaml
make sweep SEEDS="0 1 2"                         # nine priors x three seeds
make neuronal RUN=flagship                       # the unit-level analyses of one network
make seeds RUN=flagship N=100                    # the same over 100 seeds (trains 99 networks)
make eye_sweep SEEDS="0 1 2"                     # the eye-position-spread sweep
```

The transformation study (Fig. 10) trains its own level networks and then
analyses them without torch:

```bash
python scripts/transformation/train_levels.py --param eye_noise --levels 0.3 1 2 4 8 1000 --seeds 0 1 2 --twin
python scripts/transformation/train_levels.py --param eye_sd    --levels 1 3 9 18 27     --seeds 0 1 2 --twin
make transformation            # run_stored, run_levels, frames_by_k, report_tables, manuscript_figures
```

## The figures

Every manuscript figure is a folder under `results/manuscript/` named as the
paper numbers it, holding the composed figure under the same name as `.png`
(300 dpi), `.tif` (flattened RGB, LZW) and `.svg` (text kept as text), its
panels as `A_…`, `B_…`, and a flat copy of the png beside the folder, so
`results/manuscript/*.png` is the set the paper reads. All of them are drawn
on one standard (`src/cmsi/viz/manuscript.py`: fixed cells of 2.5 × 2.5,
3.75 × 2.5 and 7.5 × 2.5 in with absolute margins, 8/9-pt text, 12-pt panel
letters, the PLOS ONE specification of `viz/style.py`), and that file is the
one place a figure is numbered (`FIG_*`).

| figure | folder | made by |
|---|---|---|
| 1 | `fig1_task_model_network` | drawn by hand (task, generative model, encoding, network); stored here only |
| 2 | `fig2_output_scatter` | `04_figures.py --run flagship --only manuscript` |
| 3 | `fig3_fusion_weight` | `04_figures.py` (panels A and B) |
| 4 | `fig4_bias_disparity` | `scripts/08_behavioral.py fig4` (three panels: A the relative bias against true disparity, as in Körding et al. 2007, Fig. 2e, on probe trials of fixed true positions, at two priors (p_common 0.3, 0.5) of the run's configuration (`bias-disparity`; for the paper `pcommon03` and `flagship`); B the hand bias against the signed true disparity on probe trials, as in Dokka et al. 2019, Fig. 1F (panel B of `dokka-probe`); C the hand bias by the cause the network inferred, as in Dokka et al. 2019, Fig. 4C, and Rideaux et al. 2021, Fig. 2D-E (`decision-bias`)) |
| 5 | `fig5_variance_hump` | `04_figures.py` |
| 6 | `fig6_prior_sweep` | `05_prior_sweep.py` (`--replot` redraws from the stored sweep) |
| 7 | `fig7_decoding_rf_shift` | `04_figures.py` |
| 8 | `fig8_units_seeds` | `scripts/neuronal/seeds.py units` |
| 9 | `fig9_lesion_seeds` | `scripts/neuronal/seeds.py lesion` |
| 10 | `fig10_transformation_decision` | `scripts/transformation/manuscript_figures.py` |
| S1, S2 | `figS_lesion_seeds`, `figS_lesion_pairs_seeds` | `scripts/neuronal/seeds.py si` |
| supporting | `figS_lesion_networks`, `figS_units_network` | `scripts/neuronal/run.py r5`, `units` (one and two networks) |
| supporting | `figS_unit_counts_seeds`, `figS_lesion_clamp_seeds`, `figS_mixed_patterns` | `scripts/neuronal/seeds.py` |
| supporting | `figS_eye_sweep`, `figS_eye_mu_sweep` | `07_eye_sweep.py --param eye_sd`, `--param eye_mu` |
| supporting | `figS_decision_frame` | `scripts/transformation/manuscript_figures.py` |

Rendering a run other than the flagship writes to `results/manuscript_<run>/`,
so a satellite never overwrites the paper's figures.

### Commands to generate the manuscript figures

There is no single command. The figures are made per run (a trained config,
`results/<run>/`), by different scripts. For the paper, `<run>` is `flagship`.

If the config is not trained yet, run the pipeline first (it trains, analyses
and already draws Figs. 2, 3, 5 and 7):

```bash
bash scripts/run_all.sh --config configs/<run>.yaml
```

Figures that depend on one run:

```bash
python scripts/04_figures.py --run <run> --only manuscript            # Figs. 2, 3, 5, 7
python scripts/08_behavioral.py fig4 --run <run>                      # Fig. 4 (runs bias-disparity, decision-bias, dokka-probe)
python scripts/neuronal/run.py units r5 --run <run>                   # figS_units_network, figS_lesion_networks (a few minutes)
python scripts/neuronal/seeds.py units  --run <run>                   # Fig. 8
python scripts/neuronal/seeds.py lesion --run <run>                   # Fig. 9
python scripts/neuronal/seeds.py figure --run <run>                   # Figs. S1, S2, figS_unit_counts_seeds, figS_lesion_clamp_seeds
python scripts/neuronal/seeds.py patterns --run <run>                 # figS_mixed_patterns
```

- `python scripts/08_behavioral.py --run <run>` without an analysis name runs
  all of its analyses (decision-bias, dokka-probe, bias-disparity, fig4);
  `python scripts/neuronal/run.py --run <run>` runs `units` and `r5`.
- Panel A of Fig. 4 draws two priors, p_common = 0.3 and 0.5, of the run's
  own configuration: the run's config with only p_common changed. For the
  flagship these are `pcommon03` and `flagship`; for a run `<run>` whose
  config has another p_common they are `<run>_p03` and `<run>_p05`. A member that is not trained yet is left out of the panel and
  the script names it. To write the missing config files and see what to
  train: `python scripts/08_behavioral.py prior-configs --run <run>`, then
  `bash scripts/run_all.sh --config configs/<member>.yaml` for each member.
  Panels B and C are always the run's own.
- The `seeds.py` figures need the 100 seeds of the run. They are drawn from
  the stored numbers (`seeds.py si` draws only Figs. S1 and S2). For a run that has no seeds yet,
  `make seeds RUN=<run> N=100` trains and analyses them first (hours).

Figures that do not belong to one run (no `--run`):

```bash
python scripts/05_prior_sweep.py --replot                  # Fig. 6, from the stored sweep
python scripts/transformation/manuscript_figures.py        # Fig. 10, figS_decision_frame
python scripts/07_eye_sweep.py --param eye_sd --replot     # figS_eye_sweep
python scripts/07_eye_sweep.py --param eye_mu --replot     # figS_eye_mu_sweep
```

Fig. 1 is drawn by hand and is never rendered.

One model figure that is not a manuscript figure compares the network with
Dokka et al. (2019, Fig. 1E-G) on probe trials of fixed true disparities:

```bash
python scripts/08_behavioral.py dokka-probe --run <run>   # results/<run>/figures/model/behavioral_analysis/dokka_probe/
```

## Configurations

`configs/flagship.yaml` is the main network (p_common = 0.5; a hand prior of
sd 10 deg, an eye-position prior of sd 9 deg, narrow per-trial noise ranges,
three populations of 76 Poisson units, two sigmoid layers of 64, a linear
read-out of four units; its header says why each value was chosen).
`pcommon0.yaml`, `pcommon03.yaml`, `pcommon07.yaml` and `pcommon1.yaml` are
the same file with only the Bernoulli constant changed: the two controls, at
which there is no causal inference to perform, and the two satellites. The
prior sweep and the eye sweep derive their configurations from the flagship
in code (`cmsi.utils.tweak`). `default.yaml` is what the tests and the
scripts fall back to; `optimal_sweep.yaml` is an alternative training recipe
explored during development and not used in the paper.

## Layout

```
src/cmsi/
  data/        generative model and analytical observer (generative.py),
               Poisson population codes (encoding.py), datasets and splits
  models/      the network (network.py), the balanced loss, training
  analysis/    accuracy; the implied weight, transition fit, regressions,
               variance signature and model comparison (causal.py); decoding;
               unit classes, lesions and RF shifts (units.py); bias curves
               (behavior.py); the calibration gate (calibration.py)
  experiments/ the implied-weight experiment behind 06_implied_weight.py
  viz/         house style (style.py), the manuscript standard and the figure
               registry (manuscript.py), the figures (results.py, inputs.py,
               training.py, prior_sweep.py)
  utils/       configs, paths, i/o, seeding
scripts/
  00-07_*.py, run_all.sh      the pipeline stages above
  neuronal/    the unit-level analyses of a run (run.py), the 100-seed study
               (seeds.py), a read-out of the network's own decision
               (decision_weights.py), the numpy forward pass (npmodel.py)
  transformation/   the transformation-to-decision study: train the level
               networks (train_levels.py), analyse the stored and the level
               networks (run_stored.py, run_levels.py, frames_by_k.py), the
               tables (report_tables.py) and the figures
tests/         property tests of the observer, the encoding, the analyses
               and the figure standard
configs/       the configurations above
data/, results/   generated; ignored by git except for the placeholders
```

What a run leaves behind:

```
results/<run>/
  config.yaml      the exact parameters (training from stage 2, analysis
                   block as stage 3 last used it)
  dataset.txt      which dataset it was trained on
  model.pt         weights + config + the train/val/test split
  metrics.json     every number the analysis computed
  analysis.npz     the per-trial arrays the figures are drawn from
  figures/
    inputs/        tuning curves, population heatmaps, gain, latent distributions
    training/      loss curves
    model/         01 read-out scatter … 16 the weight's distribution, one
                   figure per analysis; behavioral_analysis/ for the bias
                   analyses of stage 8 and neuronal_level_analysis/ for the
                   unit-level analyses, with every panel and a numbers.json
```

Numbers and figures are kept apart (re-plot without re-fitting, diff
`metrics.json` between runs), and the split lives in the checkpoint, so every
analysis runs on exactly the trials the network was validated against.
Comments that cite `SS n.m` refer to sections of the project's internal design
specification.

## Tests

```bash
make test        # 105 property tests
```

They state what must hold for any correct implementation rather than pin
numbers: the Bayes factor falls with disparity, the posterior is monotone in
it, model averaging reaches the fused estimate at p = 1 and the segregated one
at p = 0, the closed-form observer matches a numerical integration of the
full posterior, the observer never sees the true sources, a reloaded
checkpoint predicts identically, and every manuscript figure is on the
standard (cell size, margins, font set) with its name taken from the registry.

## From a notebook

```python
import sys; sys.path.insert(0, "src")
from cmsi.utils import load_config
from cmsi.data import make_dataset, subset
from cmsi.models import train, predict
from cmsi import analysis

cfg = load_config("configs/flagship.yaml")
d = make_dataset(cfg, n=20000)
model, history, splits = train(d, cfg)
test = subset(d, splits["test"])
pred = predict(model, test["X"])
analysis.print_accuracy(analysis.accuracy(pred, test["Y"], d["target_names"]))
```

`d["post_c1"]` is the trial-wise posterior p(C = 1 | x); `cfg["generative"]["p_common"]`
is the prior. `tweak(cfg, p_common=0.8, epochs=100)` copies a configuration
with values replaced, wherever the key lives.

## Citation

If you use this code, please cite the paper above.
