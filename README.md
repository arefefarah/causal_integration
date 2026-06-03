# causal_msi — Bayesian causal inference across reference frames

An additive feedforward neural network trained to perform **Bayesian causal
inference** on population-coded sensory cues. The network receives three
population-code input groups — **visual hand position**, **proprioceptive hand
position**, and a **proprioceptive eye-position** signal — and learns to estimate
each cue's source and uncertainty (in a common body frame) and to infer the
probability that the two cues share a **common cause**. It extends a prior
multisensory-integration model with a causal-inference read-out.

This repository is a **typed, tested skeleton**. The core Bayesian derivations are
intentionally left unimplemented as `TODO(science)` blocks (each stating the
intended formula), with matching **failing/xfail** unit tests that encode the
expected properties. You implement the derivations against the tests and keep
authorship of the science.

> Units convention: all spatial quantities are in **degrees**; all variances are
> in **degrees²** (deg²).

---

## The five network outputs

The `causal` head emits five outputs (variances via softplus, p(C=1) via sigmoid,
means via identity):

| # | Output      | Meaning                                                    |
|---|-------------|------------------------------------------------------------|
| 1 | `mu_vis`    | Visual source estimate, **body frame** (deg)               |
| 2 | `var_vis`   | Visual estimate variance, body frame (deg²; uses σ²_vis+σ²_eye) |
| 3 | `mu_prop`   | Proprioceptive source estimate (deg)                       |
| 4 | `var_prop`  | Proprioceptive estimate variance (deg²)                    |
| 5 | `p(C=1\|x)` | **Graded** common-cause posterior (regression / BCE)       |

**Integration is left implicit:** there is no fused / averaged output. The fused
and model-averaged estimates are *reconstructed in analysis* from the five
outputs. Output 5 is trained on the **graded analytical posterior** `p(C=1|x)`,
not on the binary causal label `C`.

A second `integration_only` head (2 outputs: `mu_fused`, `var_fused`) is a
Project-1-style always-fuse control twin used for the emergent-vs-imposed analysis.

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
measurements — never the true sources — and produces the five targets via the
segregated single-cue posteriors, the forced-fusion posterior, the Gaussian Bayes
factor, and the common-cause posterior (Körding et al., 2007). These closed forms
are the `TODO(science)` you implement.

---

## Setup

Requires Python 3.11+ and [Poetry](https://python-poetry.org/).

```bash
poetry install
poetry run pytest        # science tests xfail by design; the rest pass
```

### Make targets

```bash
make setup          # poetry install
make lint           # ruff + black --check + mypy
make test           # pytest with coverage
make generate-data  # build a dataset from configs/default.yaml
make train          # train the causal model (multi-seed)
make analyze        # run the analysis suite on a checkpoint
```

The same commands are exposed as Poetry scripts: `generate-data`, `train`,
`run-analysis`.

---

## Workflow

**Train**

1. `poetry run generate-data --config configs/default.yaml --out data/dataset.npz`
   (dense disparity × reliability sweep, ~50k trials, train/val/test split).
2. `poetry run train --head-type causal` → main model (softplus vars, sigmoid pc;
   weighted estimate-MSE + pc loss; rprop/adam; early stopping; per-output logging;
   ≥5 seeds).
3. `poetry run train --head-type integration_only` → the control twin.

**Test (gate before analysis)** — per-output regression decoded-vs-analytical
(slope, R²), pc calibration, proportion-common-cause vs disparity (sharpening with
reliability), fused-at-low-disparity / segregated-at-high-disparity, and
generalization to out-of-range disparities/reliabilities.

**Analyze** (`run-analysis` subcommands)

- `decoding` — layer-wise p(C=1) decodability (SIL vs MSL) + emergent-vs-imposed twin.
- `integration` — the full implicit-integration suite (reconstructed vs analytical
  vs population-decoded estimates; fusion-weight curve; disparity sweep; reliability
  dependence; model averaging vs selection vs probability matching).
- `mechanism` — congruent/opposite units + Bayes-factor tracking + ablation.
- `indices` — AI / RE / RA + gain (divisive-normalization comparison).
- `geometry` — RSA / dimensionality + reference-frame-of-causality.
- `rf_shifts` — receptive-field shift vs eye position per layer.

---

## Layout

```
configs/default.yaml          all parameters (generative / encoding / model / training / analysis)
src/causal_msi/
  utils.py        seeding (numpy + torch) and device helpers
  config.py       pydantic config models + loader
  generative.py   generative model + analytical observer  [TODO(science): closed forms]
  encoding.py     population-code encoders (3 input groups)
  models.py       FeedforwardMSI (causal / integration_only heads; SIL/MSL hooks)
  losses.py       causal_loss + integration_loss
  training.py     train loop, early stopping, multi-seed runner
  io.py           dataset / checkpoint / results persistence
  viz.py          figure helpers
  analysis/       performance, decoding, integration, congruent_opposite,
                  bayes_factor, indices, geometry, rf_shifts, ablation
  cli/            typer entry points (generate_data, train, run_analysis)
scripts/          thin CLI wrappers
tests/            property tests (generative + integration xfail by design)
notebooks/        01_explore_generative, 02_train_and_test, 03_analysis
```

## Where to start implementing

Search the codebase for `TODO(science)`:

```bash
grep -rn "TODO(science)" src tests
```

Each block states the intended formula in its docstring and points at the test
that pins down its expected behaviour.
