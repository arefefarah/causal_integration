r"""Generative model + the analytical Bayesian observer (Kording et al. 2007).

Task: visuo-proprioceptive localisation across eye positions.

    C ~ Bernoulli(p_common)
        C=1 -> one shared source s drives both cues
        C=2 -> independent sources s_vis, s_prop, each ~ N(mu0, sigma0_sq)
    e ~ N(eye_mu, eye_sigma_sq)                      eye position
    vision is RETINAL:  retinal = s_vis - e
    measurements:  x_vis  ~ N(retinal, sig2_vis)     (retinal frame)
                   x_eye  ~ N(e,       sig2_eye)
                   x_prop ~ N(s_prop,  sig2_prop)    (body frame)

The observer sees only the noisy measurements -- never the true sources or C --
and produces the four supervised targets:

    mu_vis, var_vis, mu_prop, var_prop

each the model-averaged optimal estimate (Eqs. 9/10) and the variance of the
same two-component mixture posterior. The trial-wise posterior p(C=1|x) is
computed on the way there (stored as "post_c1") but is neither an input nor an
output: the network has to infer it implicitly.

Trials are one flat dict of arrays, so you can mask them freely:
    hi = d["sig2_vis"] < 2
    analysis.accuracy(pred[hi], d["Y"][hi], names)
"""

import numpy as np
from scipy.special import expit


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def _draw_trials(n, gen, rng):
    """One raw draw of n trials -- the strict sampling order of the design (SS3)."""

    # creates an array named C containing 500 random elements consisting of either 1 or 2, based on a 50 / 50 probability.\
    C = np.where(rng.random(n) < gen["p_common"], 1, 2)

    sigma0 = np.sqrt(gen["sigma0_sq"])
    shared = rng.normal(gen["mu0"], sigma0, n)
    s_vis = np.where(C == 1, shared, rng.normal(gen["mu0"], sigma0, n))
    s_prop = np.where(C == 1, shared, rng.normal(gen["mu0"], sigma0, n))

    eye = rng.normal(gen["eye_mu"], np.sqrt(gen["eye_sigma_sq"]), n)
    sig2_vis = rng.uniform(*gen["sigma2_vis_range"], n)
    sig2_prop = rng.uniform(*gen["sigma2_prop_range"], n)
    sig2_eye = rng.uniform(*gen["sigma2_eye_range"], n)

    retinal = s_vis - eye
    return {
        "C": C, "s_vis": s_vis, "s_prop": s_prop, "eye": eye, "retinal": retinal,
        "sig2_vis": sig2_vis, "sig2_prop": sig2_prop, "sig2_eye": sig2_eye,
        "x_vis": rng.normal(retinal, np.sqrt(sig2_vis)),
        "x_eye": rng.normal(eye, np.sqrt(sig2_eye)),
        "x_prop": rng.normal(s_prop, np.sqrt(sig2_prop)),
    }


def sample_trials(n, gen, rng, contain=None):
    """Sample latents and render noisy measurements. `gen` is cfg["generative"].

    contain (design SS3 range containment / failure mode #9): optional
    (lo, hi) bounds on the RETINAL VISUAL MEASUREMENT x_vis -- normally the
    encoded visual field pulled in by a 2*rf_width margin. Trials whose x_vis
    falls outside are rejected and redrawn through the identical code path, so
    the rejection rule is structurally the same for C=1 and C=2 (the marginal
    of x_vis is identical under both, so rejection cannot become a C cue).
    The realised rejection rate is returned under "rejection_rate" (length-1
    array); keep it below a few percent -- truncation slightly deforms the
    effective prior relative to the Gaussian the targets assume (SS3 note).
    """
    d = _draw_trials(n, gen, rng)
    if contain is None:
        d["rejection_rate"] = np.array([0.0])
        return d

    lo, hi = contain
    n_rejected, n_drawn = 0, n
    for _ in range(100):                       # safety cap; never reached in practice
        bad = (d["x_vis"] < lo) | (d["x_vis"] > hi)
        if not bad.any():
            break
        n_rejected += int(bad.sum())
        n_drawn += int(bad.sum())
        redraw = _draw_trials(int(bad.sum()), gen, rng)
        for key, value in redraw.items():
            d[key][bad] = value
    else:
        raise RuntimeError("containment resampling did not converge -- "
                           "the bounds exclude too much of the x_vis distribution")
    d["rejection_rate"] = np.array([n_rejected / n_drawn])
    return d


def containment_bounds(enc, n_widths=2.0):
    """(lo, hi) for x_vis: the encoded visual field pulled in by n_widths*rf_width.

    Returns None when the field already contains everything worth keeping
    (margin >= half the field would reject everything).
    """
    lo, hi = enc["visual_field"]
    margin = n_widths * enc["rf_width"]
    if hi - margin <= lo + margin:
        raise ValueError("visual_field too narrow for the containment margin")
    return lo + margin, hi - margin


# --------------------------------------------------------------------------- #
# The observer, one step per function
# --------------------------------------------------------------------------- #
def to_body_frame(x_vis, x_eye, sig2_vis, sig2_eye, eye_mu, eye_sigma_sq):
    """Retinal visual measurement -> body frame (Eqs. 1/2 of the MSI paper).

    This is the reference-frame transformation

    The eye measurement is first combined with its own prior. That is not a
    refinement -- it is what the sufficient statistic actually is. Because
    ``x_vis = s - e + noise`` and ``x_eye = e + noise`` both depend on ``e``,
    the two are correlated, and the raw sum ``x_vis + x_eye`` is unbiased but
    not efficient. Shrinking toward the eye prior gives

        var_eye = 1 / (1/sig2_eye + 1/eye_sigma_sq)
        k       = var_eye / sig2_eye = eye_sigma_sq / (eye_sigma_sq + sig2_eye)
        eye_hat = k * x_eye + (1 - k) * eye_mu

    Pass ``eye_sigma_sq = np.inf`` for a flat prior on eye position, which
    recovers the plain ``x_vis + x_eye``, ``sig2_vis + sig2_eye`` of Eqs. 1/2.
    Both arguments are required rather than defaulted, so the assumption about
    eye position is always visible at the call site.
    """
    prior_precision = 0.0 if np.isinf(eye_sigma_sq) else 1.0 / eye_sigma_sq
    var_eye = 1.0 / (1.0 / sig2_eye + prior_precision)
    k = var_eye / sig2_eye
    eye_hat = k * x_eye + (1.0 - k) * eye_mu
    return x_vis + eye_hat, sig2_vis + var_eye


def single_cue_posterior(x, var, mu0, sigma0_sq):
    """One cue + prior: the C=2 'segregated' estimate (Eq. 11)."""
    precision = 1 / var + 1 / sigma0_sq
    mu = (x / var + mu0 / sigma0_sq) / precision
    return mu, 1 / precision


def fused_posterior(x_vis_body, var_vis_body, x_prop, var_prop, mu0, sigma0_sq):
    """Both cues + prior, assuming one shared source: the C=1 estimate (Eq. 12)."""
    precision = 1 / var_vis_body + 1 / var_prop + 1 / sigma0_sq
    mu = (x_vis_body / var_vis_body + x_prop / var_prop + mu0 / sigma0_sq) / precision
    return mu, 1 / precision


def log_bayes_factor(x_vis_body, var_vis_body, x_prop, var_prop, mu0, sigma0_sq):
    """log p(x|C=1) - log p(x|C=2), Gaussian closed form (nats).

    Driven by the body-frame disparity d = x_vis_body - x_prop relative to the
    combined noise: large and positive as d -> 0, negative as |d| grows.
    """
    sv, sp, s0 = var_vis_body, var_prop, sigma0_sq

    # C=1: marginalise the single shared source over the prior.
    sigma_c = sv * sp + sv * s0 + sp * s0
    e1 = ((x_vis_body - x_prop) ** 2 * s0
          + (x_vis_body - mu0) ** 2 * sp
          + (x_prop - mu0) ** 2 * sv) / sigma_c
    log_c1 = -0.5 * np.log(sigma_c) - 0.5 * e1

    # C=2: two independent cues, each marginalised over its own prior.
    v_vis, v_prop = sv + s0, sp + s0
    e2 = (x_vis_body - mu0) ** 2 / v_vis + (x_prop - mu0) ** 2 / v_prop
    log_c2 = -0.5 * np.log(v_vis * v_prop) - 0.5 * e2

    return log_c1 - log_c2   # the shared -log(2*pi) cancels


def common_cause_posterior(log_bf, p_common):
    """p(C=1|x) = logistic(log_bf + logit(prior)). Monotone in log_bf."""
    pc = np.clip(p_common, 1e-12, 1 - 1e-12)
    return expit(log_bf + np.log(pc) - np.log1p(-pc))


def model_average(p, fused_mu, fused_var, seg_mu, seg_var):
    """Mix the C=1 and C=2 estimates by p(C=1) (Eqs. 9/10).

    Mean = mixture mean; variance = mixture variance (law of total variance),
    which is why it stays large when the two components disagree.
    """
    mu = p * fused_mu + (1 - p) * seg_mu
    second = p * (fused_var + fused_mu ** 2) + (1 - p) * (seg_var + seg_mu ** 2)
    return mu, second - mu ** 2


def observer(d, gen):
    """Run the whole observer on a trial dict; returns the keys it computes.

    Order matters and follows the analytical decomposition: the retinal cue is
    first brought into the body frame using eye position (Eqs. 1/2), and only
    then integrated with proprioception (Eq. 3). The common-cause decision is
    likewise made after the transform -- both cues have to be in one frame
    before their disagreement means anything.
    """
    x_vis_body, var_vis_body = to_body_frame(
        d["x_vis"], d["x_eye"], d["sig2_vis"], d["sig2_eye"],
        gen["eye_mu"], gen["eye_sigma_sq"],
    )
    mu0, s0 = gen["mu0"], gen["sigma0_sq"]

    seg_vis_mu, seg_vis_var = single_cue_posterior(x_vis_body, var_vis_body, mu0, s0)
    seg_prop_mu, seg_prop_var = single_cue_posterior(d["x_prop"], d["sig2_prop"], mu0, s0)
    fused_mu, fused_var = fused_posterior(
        x_vis_body, var_vis_body, d["x_prop"], d["sig2_prop"], mu0, s0
    )

    log_bf = log_bayes_factor(
        x_vis_body, var_vis_body, d["x_prop"], d["sig2_prop"], mu0, s0
    )
    p = common_cause_posterior(log_bf, gen["p_common"])

    mu_vis, var_vis = model_average(p, fused_mu, fused_var, seg_vis_mu, seg_vis_var)
    mu_prop, var_prop = model_average(p, fused_mu, fused_var, seg_prop_mu, seg_prop_var)

    return {
        # the four supervised targets
        "mu_vis": mu_vis, "var_vis": var_vis, "mu_prop": mu_prop, "var_prop": var_prop,
        # intermediates, kept for analysis (not network outputs)
        "x_vis_body": x_vis_body, "var_vis_body": var_vis_body,
        "disparity": x_vis_body - d["x_prop"],
        "seg_vis_mu": seg_vis_mu, "seg_vis_var": seg_vis_var,
        "seg_prop_mu": seg_prop_mu, "seg_prop_var": seg_prop_var,
        "fused_mu": fused_mu, "fused_var": fused_var,
        # NOTE the name: post_c1 is the TRIAL-WISE POSTERIOR p(C=1|x), not the
        # prior p_common in the config. Decoding analyses must target this
        # (design SS7.5 / failure mode #1: decode p(C=1|x), never p_common).
        "log_bf": log_bf, "post_c1": p,
    }
