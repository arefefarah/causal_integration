# research/

A working rewrite of `src/causal_msi/` — same model, same equations, same
numbers, organised as a research project rather than as a software package. The
original still runs and is untouched; `scripts/check_observer.py` verifies the
two observers agree to floating-point noise.

## Layout

```
configs/default.yaml     every parameter, in five sections
src/cmsi/
  data/        generative.py   generative model + analytical Bayesian observer
               encoding.py     measurements -> population codes (network input)
               dataset.py      make_dataset, split_indices, subset
  models/      network.py      the network, predict, hidden_activations
               losses.py       the objective and why it is reweighted
               training.py     training loop with early stopping
  analysis/    accuracy.py     readout vs observer, per output
               causal.py       implied fusion weight, curves, decision strategy
               decoding.py     what the hidden layers carry
               todo.py         placeholders for analyses not yet written
  viz/         inputs.py       what the network is shown
               training.py     did it converge
               results.py      network vs observer
               style.py        shared figure defaults
  utils/       config.py  paths.py  seed.py  io.py
scripts/       01_generate_data  02_train  03_analyze  04_figures
               run_all.sh  check_observer.py
tests/         property tests on the maths and the stage boundaries
data/          generated datasets (.npz)      gitignored
results/       one folder per run             gitignored
```

## Run it

```bash
cd research
bash scripts/run_all.sh quick     # ~2 min: data, train, twin, analysis, figures
bash scripts/run_all.sh           # the real thing (50k trials)
```

Or one stage at a time — each reads what the previous one wrote, so you can
re-run any of them alone:

```bash
python scripts/01_generate_data.py --name main --n 50000
python scripts/02_train.py         --data main --run baseline
python scripts/03_analyze.py       --run baseline --twin twin
python scripts/04_figures.py       --run baseline --only model
```

## Tests

```bash
pytest                       # 48 tests, ~2 seconds
```

They are property tests, not regression tests: each states something that must
hold for any correct implementation, so they survive changes to the parameters
or the internals. The Bayes factor falls with disparity; the posterior is
monotone in it and saturates at 0 and 1; model averaging hits the fused estimate
at p=1 and the segregated one at p=0; the mixture variance exceeds both
components when they disagree; encoders are reproducible from the seed; splits
are disjoint; a reloaded checkpoint predicts identically.

Three are worth knowing about specifically:

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
results/baseline/
  config.yaml        the exact parameters that produced this run
  dataset.txt        which dataset it was trained on
  model.pt           weights + config + the train/val/test split
  metrics.json       every number the analysis computed
  analysis.npz       per-trial arrays the figures are drawn from
  figures/
    inputs/          tuning curves, population heatmaps, gain, latent distributions
    training/        loss curves, per-output loss
    model/           output scatter, errors, p(C=1) vs disparity, fusion weight,
                     reliability dependence, decoding, emergent vs imposed
```

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

**`fusion_weight`** — solve `estimate = w·fused + (1−w)·segregated` for `w`.
Optimal averaging predicts `w == p(C=1)`, so `w` against disparity is the
fusion→segregation transition, and `transition_fit` gives its midpoint and
sharpness. Trials where the two references nearly coincide are dropped: the
denominator goes to zero there and a handful of trials would otherwise dominate
every summary. Per-trial R² on `w` looks bad even when the binned curve tracks
the optimum closely — single-trial `w` is a noisy ratio, so read the slope and
the curve, not R².

**`decode` / `decode_by_layer`** — ridge-decode a latent from a hidden layer.
Independent of the readout, so it asks "is this represented?" rather than "was it
trained to output this?". Weak in layer0 and strong in the last layer means the
network is building it. Running it on the always-fuse twin, which was never asked
for anything causal, is the emergent-vs-imposed test.

**`strategy_fit`** — model averaging vs model selection vs probability matching.

`analysis/todo.py` holds one-line placeholders for the analyses still to be
designed: unit indices, population geometry, RF shifts across eye position,
congruent/opposite unit classification, and MSL ablation.

## Two things that matter for training

`standardize_inputs` and `balance_loss` are both on by default. The three input
groups differ ~30× in magnitude and the outputs live on different scales (means
≈ ±10 deg, variances ≈ 5–25 deg²). Without both, the shared trunk learns
`var_prop` fine and starves `mu_prop`, even though a dedicated decoder reaches
R²≈0.97 on it. `figures/training/02_per_output_loss.png` is where that shows up.
Use Adam, not Rprop — Rprop is a full-batch method and misbehaves on mini-batches
(`check` in `utils/config.py` warns if you configure that combination).

## Relation to the original

The observer maths is a line-by-line port; `scripts/check_observer.py` runs both
implementations on identical draws and asserts they agree (worst difference so
far: 2e-13). What was dropped: the pydantic config layer, the typer CLI and its
`scripts/` wrappers, the frozen dataclasses wrapping every group of arrays
(`LatentBatch`, `Measurements`, `ObserverTargets`, `Dataset`, `Encoders`, ...),
and the nine-module `analysis/` split. What was added: the four-stage pipeline,
the per-run results convention, and input/training figure groups the original
didn't have.
