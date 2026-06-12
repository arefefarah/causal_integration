# causal_msi — Bayesian causal inference across reference frames

An additive feedforward neural network trained to perform **Bayesian causal
inference** on population-coded sensory cues. The network receives three
population-code input groups — **visual hand position**, **proprioceptive hand
position**, and a **proprioceptive eye-position** signal — and learns to estimate
each cue's source and uncertainty (in a common body frame) and to infer the
probability that the two cues share a **common cause**. It extends a prior
multisensory-integration model with a causal-inference read-out.

> Units convention: all spatial quantities are in **degrees**; all variances are
> in **degrees²** (deg²).

---

## The four network outputs

The `causal` head emits **four** outputs (variances via softplus, means via
identity) — the Bayesian causal-inference **optimal position estimates** of
Körding et al. (2007):

| # | Output     | Meaning                                                          | Körding eq. |
|---|------------|------------------------------------------------------------------|-------------|
| 1 | `mu_vis`   | Optimal **visual** estimate (model-averaged), body frame (deg)   | Eq. 9       |
| 2 | `var_vis`  | Posterior (mixture) variance of the visual estimate (deg²)       | —           |
| 3 | `mu_prop`  | Optimal **proprioceptive** estimate (model-averaged) (deg)       | Eq. 10      |
| 4 | `var_prop` | Posterior (mixture) variance of the prop estimate (deg²)         | —           |

Each optimal estimate is the cost-minimising (posterior-mean) combination of the
common-cause estimate (Eq. 12, fused) and the separate-cause estimate (Eq. 11,
segregated), weighted by the common-cause posterior `p(C=1|x)` (Eq. 2):

```
mu_i = p(C=1|x)·ŝ_{i,C=1} + (1 − p(C=1|x))·ŝ_{i,C=2}        (Eqs. 9/10)
```

**`p(C=1|x)` is computed internally** from the Gaussian evidences (Eqs. 2/4/6) to
form these estimates, but it is **neither an input nor an output** — the network
must infer it implicitly. The paired variance is the variance of the same
2-component mixture posterior (law of total variance).

A second `integration_only` head (2 outputs: `mu_fused`, `var_fused`) is a
Project-1-style always-fuse control twin (Eq. 12) used for the emergent-vs-imposed
analysis.

---

## Generative model

- **Spatial prior:** `source ~ Normal(mu0, sigma0²)`.
- **Causal latent:** `C ~ Bernoulli(p_common)`. If `C=1`, one shared source `s`;
  if `C=2`, two independent sources `s1` (vision), `s2` (proprioception).
- **Eye position** `e` from its own distribution; per-trial reliabilities
  `σ²_vis, σ²_prop, σ²_eye` drawn from configured ranges.
- **Reference-frame transform:** vision is **retinal** (`retinal = s − e`),
  proprioception is **body**; body-frame visual = `retinal + e`.
- **Noisy scalar measurements:** `x_vis ~ N(retinal, σ²_vis)`,
  `x_eye ~ N(e, σ²_eye)`, `x_prop ~ N(s_or_s2, σ²_prop)`.

The **analytical observer** (the label pipeline) operates only on the noisy scalar
measurements — never the true sources — and produces the four optimal-estimate
targets: segregated single-cue posteriors (Eq. 11), the forced-fusion posterior
(Eq. 12), the Gaussian Bayes factor / common-cause posterior (Eqs. 2/4/6), and the
model-averaged optimal estimates + variances (Eqs. 9/10) — Körding et al. (2007).

---

## Implementation status

The core science is **implemented and tested**; several downstream analysis
metrics remain as scaffolded `TODO(science)` stubs for you to author.

**✅ Implemented (and runnable end-to-end)**

- `generative.py` — generative model + full analytical observer
  (`transform_visual_to_body`, `segregated_estimate`, `fused_estimate`,
  `log_bayes_factor`, `common_cause_posterior`, `analytical_observer`,
  `build_dataset`).
- `encoding.py` — Gaussian-RF and push-pull encoders, reliability gain, Poisson noise.
- `models.py`, `losses.py`, `training.py`, `io.py`, `utils.py`, `config.py`, `viz.py`.
- `analysis/integration.py` — **all** functions (model averaging, likelihood
  inversion / reconstruction, population decoding, fusion-weight curve, disparity
  sweep, reliability dependence, decision-strategy fit).
- `analysis/bayes_factor.py` — **all** functions (implied log-BF inversion,
  comparison, population tracking).
- `analysis/decoding.py` — layer-wise p(C=1) decoding + emergent-vs-imposed twin.
- `analysis/performance.py` — per-output regression, error distributions,
  calibration, generalization report.

**🛠️ Still `TODO(science)` (raise `NotImplementedError`; author against the tests)**

- `analysis/indices.py` — AI / RE / RA / gain-index formulae.
- `analysis/geometry.py` — `pc_axis_explicitness`, `reference_frame_of_causality`.
- `analysis/rf_shifts.py` — `rf_shift_gain`.
- `analysis/congruent_opposite.py` — `classify_units`, `balance_predicts_readout`.
- `analysis/ablation.py` — `ablation_dissociation`.
- `analysis/performance.py` — `proportion_common_vs_disparity`,
  `proportion_common_vs_reliability` (curve summaries; the raw binning is shown in
  `visualize` / notebook 03).

Find them all with:

```bash
grep -rn "raise NotImplementedError" src/
```

---

## Setup

Requires Python 3.11+ and [Poetry](https://python-poetry.org/).

```bash
poetry install              # creates the venv + registers console scripts
poetry run pytest           # 16 passed, 14 xpassed (the science property tests)
```

> **Note on console scripts.** `generate-data`, `train`, `run-analysis`, and
> `visualize` are entry points declared in `pyproject.toml` under
> `[project.scripts]`. They are only created during `poetry install`. If you add a
> new script and get "Command not found", re-run `poetry install` (or call the
> module directly, e.g. `poetry run python scripts/visualize.py ...`).

---

## How to run (complete workflow)

### 1. Generate a dataset

```bash
poetry run generate-data --config configs/default.yaml --out data/dataset.npz
```

Samples latents → renders noisy measurements → encodes the three population-code
input groups → runs the analytical observer for the targets. Writes a compressed
`.npz` (inputs, targets, latents, measurements).

### 2. Train

```bash
# Main model (5-output causal head)
poetry run train --head-type causal --dataset-path data/dataset.npz

# Control twin (always-fuse, 2 outputs) for the emergent-vs-imposed analysis
poetry run train --head-type integration_only --dataset-path data/dataset.npz
```

Trains `training.n_seeds` models with the chosen optimizer (rprop/adam), early
stopping on validation loss, and per-output logging; saves
`checkpoints/model_seed{N}.pth`. Omit `--dataset-path` to generate data on the fly.

### 3. Visualize (diagnostics → `results/`)

```bash
# Quick-train a small model and render all figures
poetry run visualize --n-trials 8000 --epochs 100

# …or visualize a checkpoint you already trained
poetry run visualize --checkpoint checkpoints/model_seed0.pth
```

Produces 10 figures under `results/`:

| File | Group | Shows |
|------|-------|-------|
| `01_encoding_tuning_curves` | encoding | RF bumps + push-pull tuning |
| `02_encoding_population_vectors` | encoding | single-trial code at 3 stimuli |
| `03_encoding_heatmaps` | encoding | code vs stimulus (bump tracks stimulus) |
| `04_encoding_gain_poisson` | encoding | gain ∝ 1/variance; Poisson mean=λ |
| `05_training_loss` | quality | train/val loss + per-component |
| `06_output_r2_bars` | quality | per-output R² + common-cause accuracy |
| `07_output_decoding_scatter` | output | network-vs-analytical optimal estimate (4 outputs) |
| `08_output_error_histograms` | output | per-output error distributions |
| `09_proportion_common_vs_disparity` | analysis | analytical p(C=1) vs disparity (internal latent) |
| `10_integration_estimates` | analysis | reconstructed/decoded vs analytical + fusion weight |

### 4. Analysis library

The implemented analyses (`integration`, `bayes_factor`, `decoding`,
`performance`) are exercised in `notebooks/03_analysis.ipynb` and from the
`visualize` command, and are importable directly, e.g.:

```python
from causal_msi.analysis import integration as integ
analytical    = integ.analytical_model_averaged_estimate(dataset.targets)
reconstructed = integ.network_reconstructed_estimate(pred, mu0, sigma0_sq)
fit           = integ.decision_strategy_fit(pop_estimate, p_common, fused, seg)
```

> The `run-analysis` CLI exposes subcommands (`decoding`, `integration`,
> `mechanism`, `indices`, `geometry`, `rf_shifts`) as **scaffolds** — they print a
> TODO until you wire the chosen analyses + result-saving for your study. Use the
> notebook / `visualize` paths above for the already-implemented analyses.

### Make targets

```bash
make setup          # poetry install
make lint           # ruff + black --check + mypy
make test           # pytest with coverage
make generate-data  # build a dataset from configs/default.yaml
make train          # train the causal model
make analyze        # run-analysis (scaffold)
make visualize      # render figures into results/
```

---

## Training note (input/target scaling)

The four outputs live on different scales (means ≈ ±10 deg vs variances ≈ 5–25
deg²), and the three input groups differ ~30× in magnitude. A diagnostic study
found that without scaling, the shared `[64,64]` network learns some outputs (e.g.
`var_prop`) but starves others (notably `mu_prop`) — even though each is easily
learnable by a dedicated decoder (R²≈0.97). The robust recipe is:

1. **Standardize the inputs** (z-score per feature on the train set).
2. **Balance the loss across outputs** (normalize each component by its target
   variance) so no output dominates.
3. **Use Adam** for mini-batch training (Rprop is a full-batch method).

These remain modelling choices for you to wire in; the diagnostics confirmed all
four outputs then reach high R².

---

## Layout

```
configs/default.yaml          all parameters (generative / encoding / model / training / analysis)
src/causal_msi/
  utils.py        seeding (numpy + torch) and device helpers
  config.py       pydantic config models + loader
  generative.py   generative model + analytical observer  [implemented]
  encoding.py     population-code encoders (3 input groups)
  models.py       FeedforwardMSI (causal / integration_only heads; SIL/MSL hooks)
  losses.py       causal_loss + integration_loss
  training.py     train loop, early stopping, multi-seed runner
  io.py           dataset / checkpoint / results persistence
  viz.py          figure helpers
  analysis/       performance, decoding, integration, bayes_factor (done);
                  indices, geometry, rf_shifts, congruent_opposite, ablation (TODO)
  cli/            typer entry points (generate_data, train, run_analysis, visualize)
scripts/          thin CLI wrappers
tests/            property tests (generative + integration now pass as xpass)
notebooks/        01_explore_generative · 02_train_and_test · 03_analysis
results/          generated figures (gitignored)
```

## Data / artefacts

`data/`, `checkpoints/`, `results/`, and `outputs/` are gitignored. Datasets are
`.npz`, checkpoints are torch `.pth` (state dict + config + metrics), results are JSON.
