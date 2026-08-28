"""Unit-level analyses of the hidden layers (design SS7.5).

Congruent/opposite classification follows the logic of Rideaux, Storrs,
Maiello & Welchman (2021): sweep the visual and the proprioceptive position
independently, correlate each unit's two tuning curves, and call a unit
congruent when they co-vary (same preferred direction in both cues) and
opposite when they anti-vary. The secondary claim of the design is that the
congruent-minus-opposite activity balance carries p(C=1|x); `balance` tests
it and `lesion` asks what the readout loses without each subpopulation.

`rf_shift` is the continuity analysis from the published paper: sweep visual
position at several eye positions and measure how much each unit's preferred
SPATIAL position shifts with the eye. In spatial coordinates a retinal code
shifts with gain -1, a body/spatial code with gain 0; partial shifts are the
interesting intermediate case. Peak response vs eye position is the gain field.

All sweeps are built on noiseless rates (poisson_noise off) at a fixed,
mid-range reliability: tuning is a property of the deterministic mapping, and
Poisson noise would only blur the correlation.
"""

import numpy as np
import torch

from cmsi.data.encoding import encode


def _sweep_dict(x_vis, x_prop, x_eye, sig2):
    n = max(np.size(x_vis), np.size(x_prop), np.size(x_eye))

    def full(v):
        return np.full(n, v, float) if np.size(v) == 1 else np.asarray(v, float)

    return {
        "x_vis": full(x_vis), "x_prop": full(x_prop), "x_eye": full(x_eye),
        "sig2_vis": np.full(n, sig2["vis"]), "sig2_prop": np.full(n, sig2["prop"]),
        "sig2_eye": np.full(n, sig2["eye"]),
    }


def _activations(model, X):
    from cmsi.models.network import hidden_activations
    return hidden_activations(model, X)


def _mid_sigmas(gen):
    return {"vis": float(np.mean(gen["sigma2_vis_range"])),
            "prop": float(np.mean(gen["sigma2_prop_range"])),
            "eye": float(np.mean(gen["sigma2_eye_range"]))}


def tuning_curves(model, encoders, cfg, positions=None, layer="msl"):
    """Each hidden unit's noiseless response to a visual sweep and to a
    proprioceptive sweep (the other cue held at 0, eye at 0).

    Returns (positions, resp_vis, resp_prop), responses (n_positions, n_units).
    """
    gen, enc = cfg["generative"], cfg["encoding"]
    if positions is None:
        span = 2.0 * np.sqrt(gen["sigma0_sq"])
        positions = np.linspace(-span, span, 41)
    sig2 = _mid_sigmas(gen)
    clean = dict(enc, poisson_noise=False)
    rng = np.random.default_rng(0)  # unused when poisson_noise is off

    d_vis = _sweep_dict(positions, 0.0, 0.0, sig2)
    d_prop = _sweep_dict(0.0, positions, 0.0, sig2)
    d_prop["x_vis"] = np.zeros(len(positions))
    resp_vis = _activations(model, encode(d_vis, encoders, clean, rng))[layer]
    resp_prop = _activations(model, encode(d_prop, encoders, clean, rng))[layer]
    return np.asarray(positions), resp_vis, resp_prop


def congruency(model, encoders, cfg, positions=None, layer="msl", threshold=0.5):
    """Classify units as congruent / opposite / untuned (SS7.5).

    Congruency index = Pearson correlation between a unit's visual-sweep and
    proprioceptive-sweep tuning curves (Rideaux's congruency logic): > threshold
    is congruent, < -threshold opposite. Units whose response barely moves over
    either sweep are untuned and excluded from the balance.
    """
    positions, rv, rp = tuning_curves(model, encoders, cfg, positions, layer)
    n_units = rv.shape[1]
    index = np.zeros(n_units)
    tuned = np.zeros(n_units, bool)
    for u in range(n_units):
        sv, sp = rv[:, u].std(), rp[:, u].std()
        tuned[u] = (sv > 1e-4) and (sp > 1e-4)
        if tuned[u]:
            index[u] = np.corrcoef(rv[:, u], rp[:, u])[0, 1]
    classes = np.where(~tuned, "untuned",
                       np.where(index > threshold, "congruent",
                                np.where(index < -threshold, "opposite", "mixed")))
    return {"index": index, "classes": classes, "tuned": tuned,
            "n_congruent": int((classes == "congruent").sum()),
            "n_opposite": int((classes == "opposite").sum()),
            "n_mixed": int((classes == "mixed").sum()),
            "n_untuned": int((classes == "untuned").sum()),
            "positions": positions}


def balance(acts_layer, classes, post):
    """Does the congruent-minus-opposite activity balance carry p(C=1|x)?

    Per trial: mean activity of congruent units minus mean activity of opposite
    units; correlate with the analytical posterior. Returns the correlation,
    the slope of post on balance, and r2 of a 1-D linear read-out.
    """
    con = classes == "congruent"
    opp = classes == "opposite"
    if con.sum() == 0 or opp.sum() == 0:
        return {"n_congruent": int(con.sum()), "n_opposite": int(opp.sum()),
                "corr": np.nan, "r2": np.nan, "slope": np.nan}
    bal = acts_layer[:, con].mean(1) - acts_layer[:, opp].mean(1)
    post = np.asarray(post)
    corr = float(np.corrcoef(bal, post)[0, 1])
    X = np.column_stack([bal, np.ones(len(bal))])
    beta, *_ = np.linalg.lstsq(X, post, rcond=None)
    pred = X @ beta
    ss_res = float(((post - pred) ** 2).sum())
    ss_tot = float(((post - post.mean()) ** 2).sum())
    return {"n_congruent": int(con.sum()), "n_opposite": int(opp.sum()),
            "corr": corr, "slope": float(beta[0]),
            "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
            "balance": bal}


@torch.no_grad()
def msl_activations(model, X, batch_size=4096):
    """Last-hidden-layer activations, as a numpy array."""
    model.eval()
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.as_tensor(np.asarray(X[i:i + batch_size]),
                             dtype=torch.float32, device=device)
        h = (xb - model.x_mean) / model.x_std
        for layer in model.layers:
            h = layer(h)
        out.append(h.cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def lesion(model, X, unit_mask, mode="mean", clamp_to=None, batch_size=4096):
    """Predictions with a subset of LAST-hidden-layer (MSL) units ablated.

    unit_mask: boolean, True = keep, False = ablate.

    mode="mean" (default): each ablated unit is CLAMPED TO ITS MEAN activation
    across the trials in X. This removes the unit's *information* -- it no
    longer varies with the trial -- while leaving the read-out's operating
    point intact.

    mode="zero": each ablated unit is set to 0. Kept for comparison, but it is
    the wrong ablation for a sigmoid layer and will overstate every effect. A
    sigmoid unit's resting output is nowhere near 0 (in practice the per-unit
    means run 0.01-0.99, median ~0.4), so forcing it to 0 does not remove the
    unit -- it injects a large constant perturbation that the read-out's
    weights and biases were never calibrated for. The damage then reflects the
    size of that perturbation, not the unit's role.

    clamp_to: optional per-unit values to clamp to instead of the mean of X.
    """
    model.eval()
    device = next(model.parameters()).device
    keep = torch.as_tensor(np.asarray(unit_mask, bool), device=device)

    if mode == "mean":
        ref = msl_activations(model, X, batch_size).mean(0) if clamp_to is None \
            else np.asarray(clamp_to, float)
        ref_t = torch.as_tensor(ref, dtype=torch.float32, device=device)
    elif mode == "zero":
        ref_t = torch.zeros(int(keep.numel()), dtype=torch.float32, device=device)
    else:
        raise ValueError(f"unknown lesion mode {mode!r}")

    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.as_tensor(np.asarray(X[i:i + batch_size]),
                             dtype=torch.float32, device=device)
        h = (xb - model.x_mean) / model.x_std
        for layer in model.layers:
            h = layer(h)
        h = torch.where(keep, h, ref_t.expand_as(h))
        o = model.readout(h)
        cols = list(o.unbind(dim=1))
        for c in model.var_cols:
            cols[c] = torch.nn.functional.softplus(cols[c])
        out.append(torch.stack(cols, dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def _rmse(pred, target):
    return float(np.sqrt(((pred - target) ** 2).mean()))


def lesion_comparison(model, X, classes, target, mode="mean",
                      n_random=200, seed=0):
    """Ablate each subpopulation, against a SIZE-MATCHED RANDOM baseline (SS7.5).

    The baseline is what makes this interpretable. Ablating any k of the 64 MSL
    units costs something; the question is whether ablating *these* k costs more
    than ablating k arbitrary ones. For each subpopulation of size k we draw
    `n_random` random subsets of the same size, ablate each, and report where
    the real lesion falls in that null distribution:

        z          (rmse - null_mean) / null_sd
        percentile fraction of random lesions that hurt LESS

    A subpopulation is only "special" if it sits well out in the upper tail.
    Without this baseline the raw damage number says nothing: it is dominated
    by how many units were removed.
    """
    from cmsi.models.network import predict
    rng = np.random.default_rng(seed)
    n_units = len(classes)
    ref = msl_activations(model, X).mean(0)

    base = _rmse(predict(model, X), target)
    out = {"intact": {"rmse": base, "mode": mode, "n_random": int(n_random)}}

    null_cache = {}
    for label in ("congruent", "opposite", "mixed"):
        idx = np.flatnonzero(classes == label)
        k = len(idx)
        if k == 0:
            out[f"no_{label}"] = {"n_units": 0, "skipped": "no units in class"}
            continue

        keep = np.ones(n_units, bool)
        keep[idx] = False
        pred = lesion(model, X, keep, mode=mode, clamp_to=ref)
        rmse = _rmse(pred, target)

        if k not in null_cache:
            draws = []
            for _ in range(n_random):
                rkeep = np.ones(n_units, bool)
                rkeep[rng.choice(n_units, size=k, replace=False)] = False
                draws.append(_rmse(lesion(model, X, rkeep, mode=mode, clamp_to=ref),
                                   target))
            null_cache[k] = np.array(draws)
        null = null_cache[k]

        out[f"no_{label}"] = {
            "n_units": int(k),
            "rmse": rmse,
            "null_mean": float(null.mean()),
            "null_sd": float(null.std()),
            "z": float((rmse - null.mean()) / null.std()) if null.std() > 0 else np.nan,
            "percentile": float((null < rmse).mean()),
            "rmse_per_output": np.sqrt(((pred - target) ** 2).mean(0)).tolist(),
        }
    return out


def rf_shift(model, encoders, cfg, eye_positions=None, vis_positions=None,
             layer="msl"):
    """Preferred-position shift with eye position, in SPATIAL coordinates.

    For each eye position e the visual sweep is presented in spatial
    coordinates s and encoded retinally (x_vis = s - e, x_eye = e). A unit
    whose preferred s does not move with e carries a spatial (body) code
    (shift gain 0); one whose preferred s moves as +e keeps a retinal code
    (gain +1 in spatial terms). Also returns the gain field: peak response vs
    eye position, per unit.
    """
    gen, enc = cfg["generative"], cfg["encoding"]
    if eye_positions is None:
        sd = np.sqrt(gen["eye_sigma_sq"])
        eye_positions = np.linspace(-sd, sd, 5)
    if vis_positions is None:
        span = 1.5 * np.sqrt(gen["sigma0_sq"])
        vis_positions = np.linspace(-span, span, 61)
    sig2 = _mid_sigmas(gen)
    clean = dict(enc, poisson_noise=False)
    rng = np.random.default_rng(0)

    prefs, peaks = [], []
    for e in eye_positions:
        d = _sweep_dict(np.asarray(vis_positions) - e, 0.0, e, sig2)
        resp = _activations(model, encode(d, encoders, clean, rng))[layer]
        # preferred spatial position: response-weighted mean over the sweep
        r = resp - resp.min(0, keepdims=True)
        denom = r.sum(0)
        denom[denom < 1e-9] = np.nan
        prefs.append((np.asarray(vis_positions)[:, None] * r).sum(0) / denom)
        peaks.append(resp.max(0))
    prefs, peaks = np.array(prefs), np.array(peaks)      # (n_eye, n_units)

    e = np.asarray(eye_positions, float)
    gains = np.full(prefs.shape[1], np.nan)
    gainfield = np.full(prefs.shape[1], np.nan)
    for u in range(prefs.shape[1]):
        if np.isfinite(prefs[:, u]).all():
            gains[u] = np.polyfit(e, prefs[:, u], 1)[0]
        if np.ptp(peaks[:, u]) > 1e-9:
            gainfield[u] = np.polyfit(e, peaks[:, u], 1)[0]
    return {"eye_positions": e, "shift_gain": gains, "gain_field": gainfield,
            "preferred": prefs, "peak": peaks,
            "median_shift_gain": float(np.nanmedian(gains))}
