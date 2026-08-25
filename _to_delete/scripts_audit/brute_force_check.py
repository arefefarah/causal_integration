"""Brute-force certification of the analytical targets (design SS5 mandatory test).

True generative model per trial:
  C=1: s ~ N(mu0, s0);  e ~ N(mu_e, se0)
       x_vis ~ N(s - e, sv); x_eye ~ N(e, sE); x_prop ~ N(s, sp)
  C=2: s_v, s_p indep ~ N(mu0, s0); e as above
       x_vis ~ N(s_v - e, sv); x_eye ~ N(e, sE); x_prop ~ N(s_p, sp)

Compute by numerical integration on grids:
  p(C=1|x), E[s_hand|x], Var[s_hand|x], E[s_vis|x], Var[s_vis|x]
where hand = s (C=1) or s_p (C=2); vis-source = s (C=1) or s_v (C=2).
Compare to (a) the code's closed-form targets (eye-prior shrinkage) and
(b) the design doc's plain-sum targets (x_vis + x_eye, sv + sE).
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cmsi.data.generative import sample_trials, observer, to_body_frame, \
    single_cue_posterior, fused_posterior, log_bayes_factor, common_cause_posterior, model_average
from cmsi.utils.config import load_config

cfg = load_config(str(Path(__file__).resolve().parents[2] / "configs" / "default.yaml"))
gen = cfg["generative"]
rng = np.random.default_rng(7)
n = 100
d = sample_trials(n, gen, rng)
code = observer(d, gen)

mu0, s0 = gen["mu0"], gen["sigma0_sq"]
mu_e, se0 = gen["eye_mu"], gen["eye_sigma_sq"]

# grids
gs = np.linspace(mu0 - 6*np.sqrt(s0), mu0 + 6*np.sqrt(s0), 801)      # source grid
ge = np.linspace(mu_e - 6*np.sqrt(se0), mu_e + 6*np.sqrt(se0), 801)  # eye grid
ds_, de_ = gs[1]-gs[0], ge[1]-ge[0]

def norm(x, m, v): return np.exp(-0.5*(x-m)**2/v)/np.sqrt(2*np.pi*v)

res = {k: np.zeros(n) for k in
       ["pC1","mu_prop","var_prop","mu_vis","var_vis"]}
for i in range(n):
    sv, sp, sE = d["sig2_vis"][i], d["sig2_prop"][i], d["sig2_eye"][i]
    xv, xp, xE = d["x_vis"][i], d["x_prop"][i], d["x_eye"][i]
    S, E = np.meshgrid(gs, ge, indexing="ij")
    prior_s = norm(gs, mu0, s0); prior_e = norm(ge, mu_e, se0)
    # C=1 joint over (s,e)
    J1 = (norm(xv, S - E, sv) * norm(xE, E, sE) *
          prior_s[:,None] * prior_e[None,:]) * norm(xp, gs, sp)[:, None]
    L1 = J1.sum() * ds_ * de_
    ps1 = J1.sum(1) * de_          # unnormalised posterior over s under C=1
    m1 = (gs*ps1).sum()/ps1.sum(); v1 = (gs**2*ps1).sum()/ps1.sum() - m1**2
    # C=2: prop branch is 1-D, vis branch is 2-D over (s_v, e)
    ps_p = norm(xp, gs, sp) * prior_s
    Lp = ps_p.sum()*ds_
    mp = (gs*ps_p).sum()/ps_p.sum(); vp = (gs**2*ps_p).sum()/ps_p.sum() - mp**2
    J2v = norm(xv, S - E, sv) * norm(xE, E, sE) * prior_s[:,None]*prior_e[None,:]
    ps_v = J2v.sum(1)*de_
    Lv = ps_v.sum()*ds_
    mv = (gs*ps_v).sum()/ps_v.sum(); vv = (gs**2*ps_v).sum()/ps_v.sum() - mv**2
    L2 = Lp * Lv
    pc = gen["p_common"]
    w = pc*L1 / (pc*L1 + (1-pc)*L2)
    res["pC1"][i] = w
    res["mu_prop"][i] = w*m1 + (1-w)*mp
    res["var_prop"][i] = w*(v1+m1**2) + (1-w)*(vp+mp**2) - res["mu_prop"][i]**2
    res["mu_vis"][i] = w*m1 + (1-w)*mv
    res["var_vis"][i] = w*(v1+m1**2) + (1-w)*(vv+mv**2) - res["mu_vis"][i]**2

def report(label, a, b):
    a, b = np.asarray(a), np.asarray(b)
    print(f"{label:>28}: max|diff| = {np.abs(a-b).max():.3e}   rms = {np.sqrt(np.mean((a-b)**2)):.3e}")

print("=== brute force vs CODE targets (eye-prior shrinkage) ===")
report("p(C=1|x)", res["pC1"], code["p_common"])
report("mu_prop", res["mu_prop"], code["mu_prop"])
report("var_prop", res["var_prop"], code["var_prop"])
report("mu_vis", res["mu_vis"], code["mu_vis"])
report("var_vis", res["var_vis"], code["var_vis"])

# design-doc plain-sum targets: x_tilde = x_vis + x_eye, var = sv + sE
xt, vt = d["x_vis"] + d["x_eye"], d["sig2_vis"] + d["sig2_eye"]
seg_v = single_cue_posterior(xt, vt, mu0, s0)
seg_p = single_cue_posterior(d["x_prop"], d["sig2_prop"], mu0, s0)
fus   = fused_posterior(xt, vt, d["x_prop"], d["sig2_prop"], mu0, s0)
lbf   = log_bayes_factor(xt, vt, d["x_prop"], d["sig2_prop"], mu0, s0)
p     = common_cause_posterior(lbf, gen["p_common"])
mv_, vv_ = model_average(p, *fus, *seg_v)
mp_, vp_ = model_average(p, *fus, *seg_p)
print("\n=== brute force vs DESIGN-DOC plain-sum targets ===")
report("p(C=1|x)", res["pC1"], p)
report("mu_prop", res["mu_prop"], mp_)
report("var_prop", res["var_prop"], vp_)
report("mu_vis", res["mu_vis"], mv_)
report("var_vis", res["var_vis"], vv_)
