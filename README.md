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
| 5 | `05_prior_sweep.py --priors … --seeds …` | one network per prior per seed, the Bernoulli constant being the only difference; the midpoint of each network's transition against the analytical one | `results/prior_sweep/`, Fig. 7 |
| 7 | `07_eye_sweep.py --param eye_sd\|eye_mu` | one network per eye-position spread (or mean), the reference-frame transformation being the only difference | `results/eye_sweep/`, `figS_eye_sweep` |

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

The transformation study (Fig. 11) trains its own level networks and then
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
| 3 | `fig3_error_histograms` | `04_figures.py` |
| 4 | `fig4_fusion_weight` | `04_figures.py` |
| 5 | `fig5_decision_bias` | `scripts/neuronal/run.py r6` |
| 6 | `fig6_variance_hump` | `04_figures.py` |
| 7 | `fig7_prior_sweep` | `05_prior_sweep.py` (`--replot` redraws from the stored sweep) |
| 8 | `fig8_decoding_rf_shift` | `04_figures.py` |
| 9 | `fig9_units_seeds` | `scripts/neuronal/seeds.py units` |
| 10 | `fig10_lesion_seeds` | `scripts/neuronal/seeds.py lesion` |
| 11 | `fig11_transformation_decision` | `scripts/transformation/manuscript_figures.py` |
| S1, S2 | `figS_lesion_seeds`, `figS_lesion_pairs_seeds` | `scripts/neuronal/seeds.py si` |
| supporting | `figS_lesion_networks`, `figS_units_network` | `scripts/neuronal/run.py r5`, `units` (one and two networks) |
| supporting | `figS_unit_counts_seeds`, `figS_lesion_clamp_seeds`, `figS_mixed_patterns` | `scripts/neuronal/seeds.py` |
| supporting | `figS_eye_sweep`, `figS_eye_mu_sweep` | `07_eye_sweep.py --param eye_sd`, `--param eye_mu` |
| supporting | `figS_decision_frame` | `scripts/transformation/manuscript_figures.py` |

Rendering a run other than the flagship writes to `results/manuscript_<run>/`,
so a satellite never overwrites the paper's figures.

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
                   figure per analysis; neuronal_level_analysis/ for the
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
