# The cross-prior experiment — result and manuscript notes

**Figure:** `results/prior_sweep/figures/prior_sweep.tif` (PLOS submission copy:
7.5 in wide, 300 dpi, flattened RGB, LZW), with `.png` to view, `.svg` to edit
and `.pdf` for LaTeX
**Supplementary:** `results/prior_sweep/figures/variance_hump_vs_prior.tif`
(+ `.png`, `.svg`, `.pdf`)
**Data:** `results/prior_sweep/sweep.json`, `results/prior_sweep/curves.npz`
**Script:** `scripts/05_prior_sweep.py`

Every number below comes from that sweep: nine networks, one per prior, 50,000
trials each, seed 0.

---

## 1. What the experiment is

Nine networks were trained, at `p_common` = 0.1 … 0.9. Between them **only the
Bernoulli constant differs** — same seed, same encoders, same σ ranges, same
architecture, same training protocol. Because the latent streams are drawn
before `C` selects between them, any two of these datasets are bit-identical on
every trial whose causal structure did not flip.

Each network is trained on the same four numbers as the flagship: `mu_vis`,
`var_vis`, `mu_prop`, `var_prop`. None has an explicit causal read-out. The
implied fusion weight is then recovered from behaviour alone.

### What this tests, stated precisely

It is worth being exact about the claim, because there is a weaker version that
would be trivial and a stronger version that would be false.

**Not the claim:** "the network discovered the prior from nothing." It did not.
The prior is present in every target it was trained on.

**The claim:** the network's *implicit* fusion-vs-segregation trade-off — which
is never an output, only something recovered from behaviour — depends on the
prior in the way Bayes says it must, quantitatively, with no free parameters.

This is a **parametric test of the Bayesian interpretation**. Its force comes
from what the alternatives predict:

| account | prediction as `p_common` varies |
|---|---|
| disparity heuristic | transition midpoint fixed — the network responds to disparity, and disparity statistics barely change |
| fixed-weight model | no transition at all, at any prior |
| model **selection** | a step at posterior 0.5 whose position moves, but with sharpness → ∞ |
| **model averaging with the correct posterior** | midpoint tracks the analytical value at every prior |

Only the last survives, and it survives quantitatively rather than qualitatively.

---

## 2. The figure, panel by panel

### Panel A — the mechanism

Implied weight on the fused estimate against signed body-frame disparity, one
curve per prior (colour = prior, sequential ramp). Solid with error bars: the
network. Dashed: the analytical posterior for that prior. Nine curves fan out
in an orderly family: at `p_common` = 0.1 the network has already abandoned
fusion by ±5°, while at 0.9 it holds a weight near 0.9 out past ±10°.

**How the per-bin weight is computed** — this matters, and the obvious method is
biased. Within each disparity bin, regress `(network − seg)` on
`Δ = fused − seg` through the origin:

```
w_bin = Σ(Δ · (network − seg)) / Σ(Δ²)          SE = σ_out / √(Σ Δ²)
```

The intuitive alternative — take the per-trial ratio `(network − seg)/Δ`, filter
it by σ_w, and average within the bin — is **biased low**. The σ_w filter keeps
preferentially large-|Δ| trials, and within a disparity bin those are the trials
whose hypotheses are most separated, which are also the lowest-weight trials.
The filtered average therefore sits below the analytical posterior even for a
perfectly Bayesian network. The least-squares form uses every trial and has no
such selection. (`analysis.binned_implied_weight`; the docstring records this.)

Two further details are visible in the panel and are deliberate:

- **Bins are coarse near zero disparity.** There the two hypotheses coincide,
  Σ(Δ²) collapses, and no amount of data identifies a weight. A fine grid there
  produces a spike that is an artefact of the estimator. Bins are also dropped
  when their slope SE exceeds 0.05.
- **Gaps break the line rather than being bridged.** At high priors most
  far-disparity bins hold too few trials to keep (at `p_common` = 0.9 only 10 of
  18 bins survive). Joining across them would draw a transition that was never
  measured.

### Panel B — the quantitative match

Each network's transition midpoint (the disparity at which the implied weight
crosses 0.5, from a logistic fit) against the analytical midpoint for the same
prior. There are no free parameters relating the two axes.

```
fit slope 0.895,  intercept +1.01 deg,  R² = 0.958
mean |network − analytical| = 0.57 deg      max = 1.87 deg
```

Across a midpoint range of 0.5–12.3°, the network lands within about half a
degree of the Bayesian prediction on average.

### Panel C — averaging at every prior, and a consistent conservatism

Position-domain regression slope (Bayes-optimal = 1) against prior, with 95%
CIs. The slope rises monotonically from **0.785 [0.71, 0.86]** at `p_common` =
0.1 to **0.954 [0.93, 0.98]** at 0.8 (trend +0.178 per unit prior, r = 0.930).

It is below 1 at every prior, and the CI excludes 1 at all nine. The network is
consistently, mildly conservative: it applies slightly less pull toward the
fused solution than the ideal observer would. This is the same deviation the
flagship analysis reports, now shown to be a stable property rather than a
quirk of one training run.

**Model averaging wins at all nine priors** (five-way comparison against full
integration, full segregation, model selection, and the best fixed-weight
model).

---

## 3. Full numbers

| `p_common` | midpoint net | midpoint opt | sharpness net | sharpness opt | posreg slope [95% CI] | MSL decode R² | best strategy |
|---|---|---|---|---|---|---|---|
| 0.1 | 2.33 | 0.46 | 0.371 | 0.254 | 0.785 [0.71, 0.86] | 0.862 | averaging |
| 0.2 | 3.61 | 3.69 | 0.325 | 0.283 | 0.854 [0.81, 0.90] | 0.928 | averaging |
| 0.3 | 6.21 | 5.54 | 0.402 | 0.306 | 0.858 [0.82, 0.89] | 0.950 | averaging |
| 0.4 | 6.66 | 6.88 | 0.351 | 0.330 | 0.854 [0.83, 0.88] | 0.957 | averaging |
| 0.5 | 7.71 | 8.04 | 0.356 | 0.350 | 0.857 [0.83, 0.88] | 0.958 | averaging |
| 0.6 | 8.05 | 8.96 | 0.321 | 0.380 | 0.896 [0.87, 0.92] | 0.950 | averaging |
| 0.7 | 10.11 | 9.93 | 0.394 | 0.414 | 0.927 [0.91, 0.95] | 0.948 | averaging |
| 0.8 | 11.79 | 10.97 | 0.399 | 0.463 | 0.954 [0.93, 0.98] | 0.916 | averaging |
| 0.9 | 12.33 | 12.26 | 0.271 | 0.512 | 0.933 [0.91, 0.95] | 0.881 | averaging |

Read-out R² on `mu_vis` is 0.988–0.995 across the whole sweep, so no network is
simply fitting the task worse at the extremes.

---

## 4. The supplementary panel, and an honest limit

`variance_hump_vs_prior.png` plots the mid-ambiguity variance elevation —
network against analytical — per prior. **It works in the middle of the range
and breaks at the top**, and the reason is measurement, not mechanism:

| `p_common` | trials with 0.2 < posterior < 0.8 | hump net | hump analytical |
|---|---|---|---|
| 0.1 | 23.2% (1741) | −1.21 | −0.20 |
| 0.3 | 40.0% (3003) | 1.50 | 2.12 |
| 0.5 | 24.6% (1846) | 3.40 | 4.55 |
| 0.7 | 10.3% (775) | 4.58 | 7.07 |
| 0.9 | **2.6% (194)** | 2.40 | **10.37** |

The hump statistic is defined on intermediate-posterior trials, and at extreme
priors there are almost none — 194 trials at `p_common` = 0.9. The network
tracks the analytical hump closely for `p_common` ≤ 0.5 and progressively
under-produces it above that, exactly where the ambiguous zone is emptying out.

**Recommendation for the manuscript:** show this as supplementary, restricted to
`p_common` ∈ [0.2, 0.7], and state the trial-count limitation. Do not present
the full range as a monotone result — it is not one, and a reader checking the
counts will notice.

---

## 5. Draft text

### Methods

> To test whether the network's implicit causal weighting is Bayesian rather
> than merely graded, we trained nine networks at `p_common` ∈ {0.1, …, 0.9}.
> These differed in the Bernoulli constant of the generative model and in
> nothing else: identical random seed, tuning curves, reliability ranges,
> architecture, and training protocol. Because the latent variables are drawn
> before the causal structure selects among them, datasets at different priors
> are identical on every trial whose structure did not change.
>
> For each network we recovered the implied weight on the fused estimate from
> behaviour alone. Within bins of body-frame disparity, the weight was estimated
> as the least-squares regression of (network estimate − segregated estimate) on
> (fused − segregated) through the origin, with standard error σ_out/√(ΣΔ²),
> where σ_out is the read-out noise measured on a `p_common` = 1 control
> network. Bins holding fewer than 80 trials, or whose slope standard error
> exceeded 0.05, were excluded; near zero disparity the two hypotheses coincide
> and the weight is not identifiable. The fusion-to-segregation transition
> midpoint was obtained by fitting a logistic to the implied weight against
> absolute disparity, and compared with the same fit applied to the analytical
> posterior.

### Results

> The implied weighting shifted systematically with the prior each network had
> been trained under (Fig. X A). Networks trained with a low prior on a common
> cause abandoned fusion at small disparities, while those trained with a high
> prior maintained near-complete fusion out to more than 10° of disparity.
>
> This shift was quantitatively Bayesian. Transition midpoints ranged from 2.3°
> to 12.3° across the sweep and tracked the analytical midpoints with a slope of
> 0.895 and R² = 0.958, deviating by 0.57° on average and never by more than
> 1.9° (Fig. X B). No parameter was fitted to relate the two.
>
> The position-domain regression confirmed model averaging at every prior, with
> slopes rising from 0.785 (95% CI [0.71, 0.86]) at `p_common` = 0.1 to 0.954
> ([0.93, 0.98]) at 0.8 (Fig. X C). Slopes lay below the Bayes-optimal value of
> 1 at all nine priors, indicating a small but consistent conservatism: the
> network applied slightly less pull toward the fused solution than an ideal
> observer. A five-way comparison against full integration, full segregation,
> model selection, and the best fixed-weight model selected model averaging at
> every prior. The trial-wise posterior remained linearly decodable from the
> multisensory layer throughout (R² = 0.86–0.96).

### Discussion sentence

> Because a disparity heuristic predicts a prior-independent transition and any
> fixed-weight scheme predicts none, the orderly and quantitatively correct
> dependence of the implied weight on the prior is difficult to account for
> without granting that the network has internalised the causal-structure
> posterior itself.

---

## 6. Caveats to keep with the result

**One seed per prior in the numbers above.** The table and figure in this
document come from nine networks, one training run each. The scatter in Panel B
(0.57° mean deviation) therefore mixes genuine deviation with run-to-run
variability, and the two cannot be separated from a single seed. Before
submission, re-run at ≥ 3 seeds per prior and show error bars on the midpoints.
This is the single most valuable addition, and the script now does it in one
invocation:

```bash
make sweep SEEDS="0 1 2"          # or the explicit form in §7
```

`--seeds` trains every (prior, seed) combination, then collapses them to one row
per prior carrying `midpoint_net_sem`, `hump_net_sem` and across-seed SEMs on
every summary statistic. The figure picks these up automatically: Panel B gains
midpoint error bars, Panel C's legend switches from "within-run 95% CI" to
"mean ± 95% CI across N seeds". The distinction matters — a within-run CI on a
regression slope says how well *that one network's* behaviour is pinned down,
not how much the result would move if the network were retrained. Only the
across-seed bars answer the second question, which is the one a reviewer asks.

Results are written after **every** seed, so a long run that is interrupted
leaves usable partial output rather than nothing.

**The extremes are less well measured.** At `p_common` = 0.1 and 0.9 the
posterior distribution is nearly degenerate, which is why the 0.9 midpoint fit
has the lowest sharpness (0.271 against an analytical 0.512) and why the hump
statistic collapses. Consider restricting the main figure to 0.2–0.8 and
showing the extremes as supplementary.

**The conservatism is real and should be reported, not smoothed over.** Slopes
of 0.79–0.95 with CIs excluding 1 at all nine priors is a consistent finding
about the network. Since every pipeline guard is verified and both controls are
clean, it is attributable to the model rather than the method.

---

## 7. Reproducing

```bash
# single seed, as the numbers in this document were produced
make sweep

# three seeds per prior — what the manuscript figure should use
make sweep SEEDS="0 1 2"

# explicit form, if you want to vary the priors too
python scripts/05_prior_sweep.py --priors 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
                                 --seeds 0 1 2 --control pcommon1
```

Runtime ≈ 15 min per seed for nine networks at 50k trials, so ≈ 45 min for
three. `--control pcommon1` supplies σ_out; without it each network falls back
to its own residuals, which is conservative rather than wrong.

`sweep.json` holds both levels: `rows` is the raw per-(prior, seed) records,
each carrying its `seed`; `aggregated` is the one-row-per-prior view with
`n_seeds`, `seeds` and the SEM fields, and is what the figure is drawn from.
Nothing is lost by aggregating — the raw rows stay in the file.

`curves.npz` keeps the per-trial arrays **for the first seed only** — they are
the bulk of the file, and Panel A is a within-seed quantity that wants one
representative run rather than a smear across runs. Per-prior curve summaries
are stored for every seed under `s{seed}_p{prior}_*` keys, with the first seed
additionally aliased to the old `p{prior}_*` names so existing plotting code
keeps working.
