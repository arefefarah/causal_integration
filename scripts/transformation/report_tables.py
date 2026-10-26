"""The report's tables and every derived number its prose quotes, generated
from the stored JSON so that nothing in the report is typed from memory.

    python scripts/transformation/report_tables.py

Reads  results/transformation_investigation/stored/{seeds.json, numbers_flagship.json}
       results/transformation_investigation/levels/{aggregated.json, rows.json, frames_by_k.json}
Writes results/transformation_investigation/tables.md       (T1-T5, Markdown)
       results/transformation_investigation/derived.json    (the derived numbers, named)
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from inv_common import OUT, save_json  # noqa: E402

S = json.loads((OUT / "stored" / "seeds.json").read_text())
F = json.loads((OUT / "stored" / "numbers_flagship.json").read_text())
A = json.loads((OUT / "levels" / "aggregated.json").read_text())
R = json.loads((OUT / "levels" / "rows.json").read_text())
FR = json.loads((OUT / "levels" / "frames_by_k.json").read_text())
ag = S["aggregated"]


def g(k, nd=3):
    a = ag[k]
    return f"{a['mean']:.{nd}f} ± {a['sd']:.{nd}f}"


def gv(k, nd=3):
    return f"{ag[k]['mean']:.{nd}f}"


def lv(param):
    return sorted([a for a in A if a["param"] == param], key=lambda a: a["level"])


def v(a, k, nd=3):
    x = a.get(k)
    return "—" if x is None else f"{x:.{nd}f}"


def vs(a, k, nd=3):
    x = a.get(k)
    return "—" if x is None else f"{x:.{nd}f} ± {a.get(k + '_sd', 0):.{nd}f}"


fr = {}
for rec in FR:
    fr.setdefault((rec["param"], rec["level"]), []).append(rec)


def frm(param, level, key, nd=2):
    vals = [r.get(key) for r in fr.get((param, level), []) if r.get(key) is not None]
    return f"{np.nanmean(vals):.{nd}f}" if vals else "—"


def frv(param, level, key):
    vals = [r.get(key) for r in fr.get((param, level), []) if r.get(key) is not None]
    return float(np.nanmean(vals)) if vals else float("nan")


# --------------------------------------------------------------------------- #
T = []
P = T.append
P("### T1. Stored flagship configuration, 100 seeds (mean ± sd over networks)\n")
P("| statistic | network | own observer R0 | true-eye observer R1 |\n|---|---|---|---|")
P(f"| causal accuracy against the true C | {g('R0.accuracy_net')} | {g('R0.accuracy_R0')} | {g('R1.accuracy_R1')} |")
P(f"| fusion window (0.5-crossing, deg) | {g('R0.crossing_net',2)} | {g('R0.crossing_R0',2)} | {g('R1.crossing_R1',2)} |")
P(f"| transition midpoint (logistic fit, deg) | {g('R0.midpoint_net',2)} | {g('R0.midpoint_R0',2)} | {g('R1.midpoint_R1',2)} |")
P(f"| weight on vision inside the fused estimate | — | {gv('R1.fused_visual_weight_R0')} | {gv('R1.fused_visual_weight_R1')} |")
P(f"| decisions flipped, R0 vs R1 | network sides with R0 on {g('R1.net_sides_with_R0_on_flips',2)} of them | {g('R1.flip_rate_R0_vs_R1',3)} of trials | — |")
P(f"| fusion of true common-cause trials, \\|t_err\\| < 1° | {g('terr.fusion_true_net_0_1')} | {g('terr.fusion_true_R0_0_1')} | {g('terr.fusion_true_R1_0_1')} |")
P(f"| … \\|t_err\\| 3–6° | {g('terr.fusion_true_net_3_6')} | {g('terr.fusion_true_R0_3_6')} | {g('terr.fusion_true_R1_3_6')} |")
P(f"| slope of that fusion on \\|t_err\\| (per deg) | {g('terr.fusion_true_slope_on_abs_terr_net')} | {g('terr.fusion_true_slope_on_abs_terr_R0')} | 0 |")
P(f"| coefficient of (w − p) on x_eye / on eye | {g('terr.residual_on_xeye',4)} / {g('terr.residual_on_eye',4)} | 0 / 0 | — |")
P(f"| accuracy at \\|eye\\| < 3° / ≥ 12° | {gv('terr.acc_net_eye_0_3')} / {gv('terr.acc_net_eye_12_60')} | {gv('terr.acc_R0_eye_0_3')} / {gv('terr.acc_R0_eye_12_60')} | — |")
P(f"| \\|μ_vis error\\| at \\|eye\\| < 3° / ≥ 12° (deg) | {gv('terr.abs_err_muvis_eye_0_3')} / {gv('terr.abs_err_muvis_eye_12_60')} | — | — |")
P(f"| fusion window vs body-frame / retinal disparity (deg) | {g('frame.crossing_body_net',2)} / {g('frame.crossing_retinal_net',2)} | {g('frame.crossing_body_R0',2)} / {g('frame.crossing_retinal_R0',2)} | — |")
P(f"| R² of the weight from \\|body\\| / \\|retinal\\| disparity | {g('frame.lookup_r2_body_net',2)} / {g('frame.lookup_r2_retinal_net',2)} | {g('frame.lookup_r2_body_R0',2)} / {g('frame.lookup_r2_retinal_R0',2)} | — |")
P(f"| effective eye gain (ratio of partials) | {g('partials.eye_gain_net_median')} | k = {gv('partials.eye_gain_k_observer')} | 1 |")
P(f"| ∂μ_hand/∂x_eye on segregated trials (median; mean \\|·\\|) | {g('partials.leak_d_muprop_dxeye_segregated_median')}; {gv('partials.leak_d_muprop_dxeye_segregated_absmean')} | 0 | 0 |")
P(f"| visual / hand error on t_err, segregated trials | {g('partials.shared_error_muvis_on_terr_segregated',2)} / {g('partials.shared_error_muprop_on_terr_segregated',2)} | {gv('partials.shared_error_muvis_on_terr_observer',2)} / 0 | — |")
P(f"| injected ε = +10°: fusion of common-cause trials / accuracy | {g('eps.fusion_true_net_+10')} / {g('eps.accuracy_net_+10')} | {g('eps.fusion_true_R0eps_+10')} / {g('eps.accuracy_R0eps_+10')} | — |")
P(f"| ε = +10°: location of maximal fusion (deg; −k·ε = {-F['eps']['k'] * 10:.2f}) | {g('eps.peak_w_vs_true_disparity_net_+10',2)} | {g('eps.peak_w_vs_true_disparity_R0eps_+10',2)} | — |")
P(f"| gain β = 0.5: accuracy / bias of μ_vis per deg of eye | {g('beta.accuracy_net_0.5')} / {g('beta.bias_muvis_on_eye_net_0.5')} | {g('beta.accuracy_R0beta_0.5')} / {g('beta.bias_muvis_on_eye_R0beta_0.5')} | — |")
P(f"| eye signal silenced: fusion of common-cause trials / accuracy | {g('silence.fusion_true_net')} / {g('silence.accuracy_net')} | {g('silence.fusion_true_R0silent')} / {g('silence.accuracy_R0silent')} | — |")
P(f"| MSL units congruent / opposite / mixed | {gv('units.n_congruent',1)} / {gv('units.n_opposite',1)} / {gv('units.n_mixed',1)} | | |")
P(f"| displacement index, opposite / congruent / mixed (median of usable units) | {g('units.di_median_opposite',2)} / {g('units.di_median_congruent',2)} / {g('units.di_median_mixed',2)} | | |")

P("\n### T2. The eye-signal noise sweep (channel A; 3 seeds per level, mean ± sd; sd 0.3 = the near-perfect signal R2, sd 1000 = the uninformative one R3)\n")
P("| eye noise sd (deg) | k | var_vis_body (deg²) | window net / R0 / R1 (deg) | accuracy net / R0 / R1 | weight on vision in fusion, R0 / R1 | fusion (\\|d\\| ≤ 2°) net / R0 | hump net / R0 (deg²) | weight-on-posterior slope | effective gain net (k) | opposite units net (twin) | Bayes-frame share of MSL units net (twin) |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
for a in lv("eye_noise"):
    P(f"| {a['level']:g} | {v(a,'basic.k')} | {v(a,'basic.var_vis_body',2)} | {vs(a,'R0.crossing_net',2)} / {v(a,'R0.crossing_R0',2)} / {v(a,'R1.crossing_R1',2)} | {vs(a,'R0.accuracy_net')} / {v(a,'R0.accuracy_R0')} / {v(a,'R1.accuracy_R1')} | {v(a,'R1.fused_visual_weight_R0',2)} / {v(a,'R1.fused_visual_weight_R1',2)} | {v(a,'R0.fusion_net')} / {v(a,'R0.fusion_R0')} | {v(a,'R0.hump_net',2)} / {v(a,'R0.hump_R0',2)} | {vs(a,'R0.weight_vs_post_slope')} | {vs(a,'partials.eye_gain_net_median')} ({v(a,'partials.eye_gain_k_observer')}) | {v(a,'units.n_opposite',1)} ({v(a,'twin.n_opposite',1)}) | {frm('eye_noise', a['level'], 'bayes_frame_share')} ({frm('eye_noise', a['level'], 'twin_bayes_frame_share')}) |")

P("\n### T3. The eye-spread levels, saved (channels D, E, G; 3 seeds per level)\n")
P("| eye sd (deg) | k | window net / R0 / R1 (deg) | accuracy net / R0 / R1 | effective gain net (k) | (w − p) on x_eye | bias of μ_vis per deg of eye, net / R0 | window vs body / retinal disparity, net (deg) | R² from \\|body\\| / \\|retinal\\| disparity, net (R0) | ∂μ_hand/∂x_eye segregated | opposite net (twin) | opposite DI median (expected 1 − k) | Bayes-frame share net (twin) | RMSE μ_vis (deg) |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for a in lv("eye_sd"):
    P(f"| {a['level']:g} | {v(a,'basic.k')} | {vs(a,'R0.crossing_net',2)} / {v(a,'R0.crossing_R0',2)} / {v(a,'R1.crossing_R1',2)} | {vs(a,'R0.accuracy_net')} / {v(a,'R0.accuracy_R0')} / {v(a,'R1.accuracy_R1')} | {vs(a,'partials.eye_gain_net_median')} ({v(a,'partials.eye_gain_k_observer')}) | {vs(a,'terr.residual_on_xeye',4)} | {v(a,'beta.bias_muvis_on_eye_net_1')} / {v(a,'beta.bias_muvis_on_eye_R0beta_1')} | {v(a,'frame.crossing_body_net',2)} / {v(a,'frame.crossing_retinal_net',2)} | {v(a,'frame.lookup_r2_body_net',2)} / {v(a,'frame.lookup_r2_retinal_net',2)} ({v(a,'frame.lookup_r2_body_R0',2)} / {v(a,'frame.lookup_r2_retinal_R0',2)}) | {vs(a,'partials.leak_d_muprop_dxeye_segregated_median')} | {v(a,'units.n_opposite',1)} ({v(a,'twin.n_opposite',1)}) | {frm('eye_sd', a['level'], 'di_median_opposite')} ({1 - a['basic.k']:.2f}) | {frm('eye_sd', a['level'], 'bayes_frame_share')} ({frm('eye_sd', a['level'], 'twin_bayes_frame_share')}) | {v(a,'R0.rmse_mu_vis')} |")

w = lv("eye_wide")[0]
P("\n### T4. The wide-reliability network (channel B; sigma2_eye_range [1, 16] deg², 3 seeds)\n")
P("| statistic | network | observer R0 |\n|---|---|---|")
P(f"| causal accuracy (R1 {v(w,'R1.accuracy_R1')}) | {vs(w,'R0.accuracy_net')} | {v(w,'R0.accuracy_R0')} |")
P(f"| fusion window, all trials (deg) | {vs(w,'R0.crossing_net',2)} | {v(w,'R0.crossing_R0',2)} |")
P(f"| fusion window by eye-noise tercile, sd 1.0–2.4 / 2.4–3.3 / 3.3–4.0° (deg) | {v(w,'eyerel.crossing_net_1_6',2)} / {v(w,'eyerel.crossing_net_6_11',2)} / {v(w,'eyerel.crossing_net_11_16',2)} | {v(w,'eyerel.crossing_R0_1_6',2)} / {v(w,'eyerel.crossing_R0_6_11',2)} / {v(w,'eyerel.crossing_R0_11_16',2)} |")
P(f"| reliability-within-disparity slope (all reliabilities; Bayes 1) | {vs(w,'eyerel.reliability_within_disparity_slope')} | 1 |")
P(f"| ∂w/∂σ²_eye at \\|d\\| < 3° (per deg²) | {v(w,'eyerel.slope_near_net',4)} | {v(w,'eyerel.slope_near_R0',4)} |")
P(f"| ∂w/∂σ²_eye at \\|d\\| 4–12° (per deg²) | {v(w,'eyerel.slope_flank_net',4)} | {v(w,'eyerel.slope_flank_R0',4)} |")
P(f"| effective eye gain (k) | {vs(w,'partials.eye_gain_net_median')} | {v(w,'partials.eye_gain_k_observer')} |")
P(f"| opposite units (twin) | {v(w,'units.n_opposite',1)} ({v(w,'twin.n_opposite',1)}) | |")

P("\n### T5. Eccentricity-dependent eye noise (channel A, second lever; sig2_eye + c·eye²; 3 seeds per level; the c = 0 row is the eye-spread sd 9 level)\n")
P("| c | accuracy net / R0 / R1 | window, all trials, net / R0 (deg) | window at \\|eye\\| 0–4° net / R0 | 4–8° | 8–12° | 12–18° | fusion (\\|d\\| ≤ 2°) at 0–4° net / R0 | 12–18° | accuracy at 12–18° net / R0 | 18–30° |\n|---|---|---|---|---|---|---|---|---|---|---|")
rows_ecc = [(0.0, [x for x in lv("eye_sd") if x["level"] == 9.0][0])] + [(x["level"], x) for x in lv("eye_ecc")]
for c, a in rows_ecc:
    P(f"| {c:g} | {vs(a,'R0.accuracy_net')} / {v(a,'R0.accuracy_R0')} / {v(a,'R1.accuracy_R1')} | {vs(a,'R0.crossing_net',2)} / {v(a,'R0.crossing_R0',2)} | {v(a,'ecc.crossing_net_eye_0_4',2)} / {v(a,'ecc.crossing_R0_eye_0_4',2)} | {v(a,'ecc.crossing_net_eye_4_8',2)} / {v(a,'ecc.crossing_R0_eye_4_8',2)} | {v(a,'ecc.crossing_net_eye_8_12',2)} / {v(a,'ecc.crossing_R0_eye_8_12',2)} | {v(a,'ecc.crossing_net_eye_12_18',2)} / {v(a,'ecc.crossing_R0_eye_12_18',2)} | {v(a,'ecc.fusion_net_eye_0_4')} / {v(a,'ecc.fusion_R0_eye_0_4')} | {v(a,'ecc.fusion_net_eye_12_18')} / {v(a,'ecc.fusion_R0_eye_12_18')} | {v(a,'ecc.accuracy_net_eye_12_18')} / {v(a,'ecc.accuracy_R0_eye_12_18')} | {v(a,'ecc.accuracy_net_eye_18_60')} / {v(a,'ecc.accuracy_R0_eye_18_60')} |")

(OUT / "tables.md").write_text("\n".join(T) + "\n")

# --------------------------------------------------------------------------- #
# derived numbers quoted in the prose
D = {}
rows = [r for r in R if r["param"] in ("eye_noise", "eye_sd", "eye_wide", "eye_ecc")]
x = np.array([r["partials.eye_gain_k_observer"] for r in rows]); y = np.array([r["partials.eye_gain_net_median"] for r in rows])
b = np.polyfit(x, y, 1)
D["gain_vs_k"] = {"n": len(rows), "slope": float(b[0]), "intercept": float(b[1]), "r2": float(1 - np.var(y - np.polyval(b, x)) / np.var(y)),
                  "mean_abs_diff": float(np.abs(y - x).mean()), "max_abs_diff": float(np.abs(y - x).max())}
x = np.array([r["R0.crossing_R0"] for r in rows]); y = np.array([r["R0.crossing_net"] for r in rows]); ok = np.isfinite(x) & np.isfinite(y)
b = np.polyfit(x[ok], y[ok], 1)
D["window_vs_R0"] = {"n": int(ok.sum()), "slope": float(b[0]), "intercept": float(b[1]), "r2": float(1 - np.var(y[ok] - np.polyval(b, x[ok])) / np.var(y[ok])),
                     "mean_net_minus_R0": float(np.mean(y[ok] - x[ok])), "max_abs_net_minus_R0": float(np.abs(y[ok] - x[ok]).max())}
ln = lv("eye_noise"); ls_ = lv("eye_sd"); lw = lv("eye_wide")[0]; le = lv("eye_ecc")
D["eye_noise"] = {f"{a['level']:g}": {
    "k": a["basic.k"], "var_vis_body": a["basic.var_vis_body"],
    "window_net": a["R0.crossing_net"], "window_R0": a["R0.crossing_R0"], "window_R1": a["R1.crossing_R1"],
    "window_net_minus_R0": a["R0.crossing_net"] - a["R0.crossing_R0"],
    "acc_net": a["R0.accuracy_net"], "acc_R0": a["R0.accuracy_R0"], "acc_R1": a["R1.accuracy_R1"],
    "loss_bayes_R1_minus_R0": a["R1.accuracy_R1"] - a["R0.accuracy_R0"], "loss_net_R0_minus_net": a["R0.accuracy_R0"] - a["R0.accuracy_net"],
    "fused_w_R0": a["R1.fused_visual_weight_R0"], "fusion_net": a["R0.fusion_net"], "fusion_R0": a["R0.fusion_R0"],
    "fusion_net_minus_R0": a["R0.fusion_net"] - a["R0.fusion_R0"],
    "hump_net": a["R0.hump_net"], "hump_R0": a["R0.hump_R0"], "hump_net_minus_R0": a["R0.hump_net"] - a["R0.hump_R0"],
    "w_slope": a["R0.weight_vs_post_slope"], "rmse_vis": a["R0.rmse_mu_vis"], "best_epoch": a["best_epoch"], "best_epoch_sd": a.get("best_epoch_sd"),
    "gain_net": a["partials.eye_gain_net_median"], "opp": a["units.n_opposite"], "twin_opp": a["twin.n_opposite"], "opp_excess": a["units.n_opposite"] - a["twin.n_opposite"],
    "congr": a["units.n_congruent"], "twin_congr": a["twin.n_congruent"],
    "di_opposite": frv("eye_noise", a["level"], "di_median_opposite"), "expected_di": 1 - a["basic.k"],
    "bayes_share": frv("eye_noise", a["level"], "bayes_frame_share"), "twin_bayes_share": frv("eye_noise", a["level"], "twin_bayes_frame_share"),
    "twin_di": frv("eye_noise", a["level"], "twin_di_median"), "di_causal": frv("eye_noise", a["level"], "di_median")} for a in ln}
D["eye_sd"] = {f"{a['level']:g}": {
    "k": a["basic.k"], "var_vis_body": a["basic.var_vis_body"],
    "window_net": a["R0.crossing_net"], "window_R0": a["R0.crossing_R0"], "window_R1": a["R1.crossing_R1"],
    "acc_net": a["R0.accuracy_net"], "acc_R0": a["R0.accuracy_R0"], "acc_R1": a["R1.accuracy_R1"],
    "loss_bayes_R1_minus_R0": a["R1.accuracy_R1"] - a["R0.accuracy_R0"], "loss_net_R0_minus_net": a["R0.accuracy_R0"] - a["R0.accuracy_net"],
    "gain_net": a["partials.eye_gain_net_median"], "gain_net_sd": a["partials.eye_gain_net_median_sd"], "resid_xeye": a["terr.residual_on_xeye"], "resid_xeye_sd": a["terr.residual_on_xeye_sd"],
    "bias_net": a["beta.bias_muvis_on_eye_net_1"], "bias_R0": a["beta.bias_muvis_on_eye_R0beta_1"],
    "cross_body": a["frame.crossing_body_net"], "cross_ret": a["frame.crossing_retinal_net"], "cross_ret_sd": a.get("frame.crossing_retinal_net_sd"),
    "r2_body": a["frame.lookup_r2_body_net"], "r2_ret": a["frame.lookup_r2_retinal_net"], "r2_body_R0": a["frame.lookup_r2_body_R0"], "r2_ret_R0": a["frame.lookup_r2_retinal_R0"],
    "leak": a["partials.leak_d_muprop_dxeye_segregated_median"], "shared_vis": a["partials.shared_error_muvis_on_terr_segregated"], "shared_hand": a["partials.shared_error_muprop_on_terr_segregated"],
    "opp": a["units.n_opposite"], "twin_opp": a["twin.n_opposite"], "rmse_vis": a["R0.rmse_mu_vis"], "w_slope": a["R0.weight_vs_post_slope"],
    "di_opposite": frv("eye_sd", a["level"], "di_median_opposite"), "expected_di": 1 - a["basic.k"],
    "bayes_share": frv("eye_sd", a["level"], "bayes_frame_share"), "twin_bayes_share": frv("eye_sd", a["level"], "twin_bayes_frame_share"),
    "twin_di": frv("eye_sd", a["level"], "twin_di_median")} for a in ls_}
D["eye_wide"] = {"k": lw["basic.k"], "acc_net": lw["R0.accuracy_net"], "acc_R0": lw["R0.accuracy_R0"], "acc_R1": lw["R1.accuracy_R1"],
                 "window_net": lw["R0.crossing_net"], "window_R0": lw["R0.crossing_R0"],
                 "terciles_net": [lw["eyerel.crossing_net_1_6"], lw["eyerel.crossing_net_6_11"], lw["eyerel.crossing_net_11_16"]],
                 "terciles_R0": [lw["eyerel.crossing_R0_1_6"], lw["eyerel.crossing_R0_6_11"], lw["eyerel.crossing_R0_11_16"]],
                 "rel_slope": lw["eyerel.reliability_within_disparity_slope"], "rel_slope_sd": lw["eyerel.reliability_within_disparity_slope_sd"],
                 "near_net": lw["eyerel.slope_near_net"], "near_R0": lw["eyerel.slope_near_R0"], "flank_net": lw["eyerel.slope_flank_net"], "flank_R0": lw["eyerel.slope_flank_R0"],
                 "gain_net": lw["partials.eye_gain_net_median"], "opp": lw["units.n_opposite"], "twin_opp": lw["twin.n_opposite"]}
D["eye_wide"]["widening_share"] = (D["eye_wide"]["terciles_net"][2] - D["eye_wide"]["terciles_net"][0]) / (D["eye_wide"]["terciles_R0"][2] - D["eye_wide"]["terciles_R0"][0])
D["eye_wide"]["flank_ratio"] = D["eye_wide"]["flank_net"] / D["eye_wide"]["flank_R0"]
D["eye_wide"]["near_ratio"] = D["eye_wide"]["near_net"] / D["eye_wide"]["near_R0"]
# fusion of near trials by eye-noise tercile (low vs high), from the per-network numbers
tf = []
for r in [r for r in R if r["param"] == "eye_wide"]:
    nb = json.loads((OUT / "levels" / "numbers" / f"{r['name']}.json").read_text())["eyerel"]["by_sig2_eye_tercile"]
    tf.append([(t["fusion_net"], t["fusion_R0"]) for t in nb])
tf = np.array(tf)
D["eye_wide"]["fusion_near_by_tercile_net"] = tf[:, :, 0].mean(0).tolist()
D["eye_wide"]["fusion_near_by_tercile_R0"] = tf[:, :, 1].mean(0).tolist()
D["eye_ecc"] = {}
for c, a in rows_ecc:
    D["eye_ecc"][f"{c:g}"] = {"acc_net": a["R0.accuracy_net"], "acc_R0": a["R0.accuracy_R0"], "window_net": a["R0.crossing_net"], "window_R0": a["R0.crossing_R0"],
                            "win_0_4_net": a.get("ecc.crossing_net_eye_0_4"), "win_0_4_R0": a.get("ecc.crossing_R0_eye_0_4"),
                            "win_12_18_net": a.get("ecc.crossing_net_eye_12_18"), "win_12_18_R0": a.get("ecc.crossing_R0_eye_12_18"),
                            "fus_0_4_net": a.get("ecc.fusion_net_eye_0_4"), "fus_0_4_R0": a.get("ecc.fusion_R0_eye_0_4"),
                            "fus_12_18_net": a.get("ecc.fusion_net_eye_12_18"), "fus_12_18_R0": a.get("ecc.fusion_R0_eye_12_18"),
                            "acc_12_18_net": a.get("ecc.accuracy_net_eye_12_18"), "acc_12_18_R0": a.get("ecc.accuracy_R0_eye_12_18"),
                            "acc_18_60_net": a.get("ecc.accuracy_net_eye_18_60"), "acc_18_60_R0": a.get("ecc.accuracy_R0_eye_18_60"),
                            "opp": a["units.n_opposite"], "twin_opp": a["twin.n_opposite"]}
    e = D["eye_ecc"][f"{c:g}"]
    if e["win_12_18_net"] is not None and e["win_12_18_R0"] is not None:
        e["widen_net"] = e["win_12_18_net"] - e["win_0_4_net"]; e["widen_R0"] = e["win_12_18_R0"] - e["win_0_4_R0"]
        e["widen_share"] = e["widen_net"] / e["widen_R0"] if e["widen_R0"] else None
# stored (100 seeds) derived
D["stored"] = {k: ag[k]["mean"] for k in ag}
D["stored_sd"] = {k: ag[k]["sd"] for k in ag}
D["stored_derived"] = {"loss_bayes": ag["R1.accuracy_R1"]["mean"] - ag["R0.accuracy_R0"]["mean"],
                       "loss_net_extra": ag["R0.accuracy_R0"]["mean"] - ag["R0.accuracy_net"]["mean"],
                       "fusion_drop_net": ag["terr.fusion_true_net_0_1"]["mean"] - ag["terr.fusion_true_net_3_6"]["mean"],
                       "fusion_drop_R0": ag["terr.fusion_true_R0_0_1"]["mean"] - ag["terr.fusion_true_R0_3_6"]["mean"],
                       "k_flagship": F["eps"]["k"],
                       "peak_pred_10": -F["eps"]["k"] * 10, "peak_pred_15": -F["eps"]["k"] * 15,
                       "peak_gap_net_10": abs(ag["eps.peak_w_vs_true_disparity_net_+10"]["mean"] + F["eps"]["k"] * 10),
                       "peak_gap_net_15": abs(ag["eps.peak_w_vs_true_disparity_net_+15"]["mean"] + F["eps"]["k"] * 15),
                       "peak_gap_net_-10": abs(ag["eps.peak_w_vs_true_disparity_net_-10"]["mean"] - F["eps"]["k"] * 10)}
# flagship-only blocks used in prose
D["flagship"] = {"beta": {str(r["beta"]): {"acc_net": r["accuracy_net"], "acc_R0": r["accuracy_R0beta"], "bias_net": r["bias_muvis_on_eye_net"], "bias_R0": r["bias_muvis_on_eye_R0beta"],
                                           "ecc": [(q["lo"], q["hi"], q["accuracy_net"], q["accuracy_R0beta"]) for q in r["by_eye_eccentricity"]]} for r in F["beta"]["levels"]},
                 "terr_sd": F["terr"]["t_err_sd"], "terr_sd_analytical": F["terr"]["t_err_sd_analytical"], "sensory_sd": F["terr"]["sensory_noise_diff_sd"],
                 "eps": {str(r["eps"]): {"fusion_net": r["fusion_true_net"], "fusion_R0": r["fusion_true_R0eps"], "acc_net": r["accuracy_net"], "acc_R0": r["accuracy_R0eps"],
                                         "peak_net": r["peak_w_vs_true_disparity_net"], "peak_R0": r["peak_w_vs_true_disparity_R0eps"], "pred": r["predicted_shift"]} for r in F["eps"]["levels"]}}
save_json(D, OUT / "derived.json")
print(f"wrote {OUT / 'tables.md'} and {OUT / 'derived.json'}")
