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
def lesion(model, X, unit_mask, batch_size=4096):
    """Predictions with a subset of LAST-hidden-layer (MSL) units silenced.

    unit_mask: boolean, True = keep, False = zero the unit's activation before
    the readout. Zeroing the activation removes exactly that unit's
    contribution W_out[:, u] * h_u.
    """
    model.eval()
    device = next(model.parameters()).device
    mask = torch.as_tensor(np.asarray(unit_mask, float),
                           dtype=torch.float32, device=device)
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.as_tensor(np.asarray(X[i:i + batch_size]),
                             dtype=torch.float32, device=device)
        h = (xb - model.x_mean) / model.x_std
        for layer in model.layers:
            h = layer(h)
        o = model.readout(h * mask)
        cols = list(o.unbind(dim=1))
        for c in model.var_cols:
            cols[c] = torch.nn.functional.softplus(cols[c])
        out.append(torch.stack(cols, dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def lesion_comparison(model, X, classes, target):
    """RMSE of the readout with each subpopulation silenced (SS7.5 optional).

    Silencing congruent units should hurt the fused regime; silencing opposite
    units should hurt where segregation (and the causal read-out) matters.
    """
    from cmsi.models.network import predict
    base = predict(model, X)
    rows = {"intact": base}
    for label in ("congruent", "opposite", "mixed"):
        keep = classes != label
        rows[f"no_{label}"] = lesion(model, X, keep)
    out = {}
    for label, pred in rows.items():
        err = pred - target
        out[label] = {"rmse_per_output": np.sqrt((err ** 2).mean(0)).tolist(),
                      "rmse": float(np.sqrt((err ** 2).mean()))}
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
