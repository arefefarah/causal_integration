"""Across-seed replication of the lesion analysis for one configuration.

The lesion figure (fig10, scripts/neuronal/run.py r5) is one network per
prior. This script asks what changes when the same configuration is trained
many times with different seeds: are the class-lesion effects consistent
across networks, or are they one network's accident?

    python scripts/neuronal/seeds.py train   --run flagship --n 100     # PyTorch; hours (02_train.py per seed)
    python scripts/neuronal/seeds.py analyse --run flagship             # torch-free; ~10-15 s per network
    python scripts/neuronal/seeds.py analyse --run flagship --extras damage pairs clamp --budget 150
                                                                        # the optional designs, resumable
    python scripts/neuronal/seeds.py figure  --run flagship             # redraw from numbers.json
    python scripts/neuronal/seeds.py all     --run flagship --n 100     # analyse + figure (no training)
    python scripts/neuronal/seeds.py patterns --run flagship --k 5      # the intermediate units' tuning patterns
    python scripts/neuronal/seeds.py si      --run flagship             # Figs. S1, S2 from numbers.json

`train` trains seeds 1 .. n-1 of the run's configuration on the run's own
dataset, as results/<run>_s<k>/ (02_train.py --seed k, which changes the
initialisation and the train/val/test split, nothing else); seed 0 is the
run itself. It skips seeds whose model.pt exists, so it can be resumed.
`analyse` takes every network found (the run plus its seeds, or --runs) and,
on each, repeats the three lesion designs that need no curve fit, with the
same mean-clamping and the same random nulls as run.py but without the slow
per-draw statistics: the whole-class clamp with the read-out frozen (weight
curve, fusion, segregation, RMSE against 100 size-matched random lesions),
the cumulative lesion in class order (segregation and fusion after each
step against 30 random orders), and the whole-class clamp with the read-out
refitted (against 20 random refits). Per-network results are cached under
per_seed/ so an interrupted analysis resumes. `figure` draws the
across-seed version of the lesion panels and prints the across-seed
summary: for every class and statistic the mean and sd over seeds of the
within-network z-score, the fraction of seeds beyond |z| = 2, and a
one-sample t statistic of the mean z against zero.

Output: results/<run>/figures/model/neuronal_level_analysis/lesion_seeds/
(the composed figure `lesion_seeds`, panels, per_seed/*.json, numbers.json,
unit_counts_by_seed.csv). The manuscript folder gets the supplementary
figure figS_lesion_seeds (Fig. S1), which keeps only the panels that the
paper's lesion figure (fig11_lesion_seeds) and unit figure (fig10_units_seeds)
do not show: the read-out error, the damage-matched null, the equal-dose
reading and the refitted read-out as z-scores (si_lesion_figure; `si` redraws
it, and the pair figure, from the stored numbers). The unit classes of the same seeds get their own
figure, `unit_counts_seeds` (A the congruency index of every unit of every
network, B the counts per class over networks with mean ± sd, C the intermediate
class's sub-structure), copied as figS_unit_counts_seeds; the printed
summary carries the counts table (mean, sd, median, range, 2.5-97.5 %).

`--extras` adds designs to every record (and `--budget` seconds per call
makes any of this resumable from per_seed/): `damage`, the whole-class clamp
against random lesions matched in size AND in RMSE; `pairs`, every single
unit and every pair of units clamped (lesion_pairs_seeds: the change per
class pair, the additivity of the effects, the worst pairs; its manuscript
copy figS_lesion_pairs_seeds, Fig. S2, leaves out the single-unit and
class-additivity panels, which are Fig. 10E and F); `clamp`, the
whole-class clamp with the units set to 0 and to 1 instead of their mean
(lesion_clamp_seeds). `patterns` collects the tuning curves of every unit
of every network and partitions the intermediate units' curves by k-means
(mixed_patterns, figS_mixed_patterns). `units` draws the manuscript's unit
figure from the stored numbers (units_seeds, copied as fig10_units_seeds:
the patterns row over the unit-class row; see units_paper_figure).
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLASS_COLORS, CLASS_LABEL, CLASSES, COLORS, PANEL_FONT, PANEL_LW,  # noqa: E402
                    ROOT, Bundle, apply_style, binned_weight, finish_panel,
                    fusion_segregation, hybrid_reads, manuscript_dir, manuscript_grid,
                    out_dir, panel_axes, position_regression, save_figures, save_json)
from cmsi.viz.style import FORMATS  # noqa: E402

K_SEEDS = "lesion_seeds"
PAPER = "figS_lesion_seeds"
K_UNITS = "unit_counts_seeds"          # the unit-class figure of the same seeds, beside it
PAPER_UNITS = "figS_unit_counts_seeds"
K_PAIRS = "lesion_pairs_seeds"         # the pair-lesion figure of the same seeds (--extras pairs)
PAPER_PAIRS = "figS_lesion_pairs_seeds"
K_PATTERNS = "mixed_patterns"           # the tuning-curve patterns of the intermediate units over the seeds
PAPER_PATTERNS = "figS_mixed_patterns"
K_CLAMP = "lesion_clamp_seeds"          # the whole-class clamp with the units set to 0 / 1 (--extras clamp)
PAPER_CLAMP = "figS_lesion_clamp_seeds"
K_UNITS_PAPER = "units_seeds"           # the paper's unit figure: patterns row over the unit-class row
PAPER_UNITS_FIG = "fig10_units_seeds"   # replaces fig10_units (one network) as the manuscript's unit figure
K_LESION_PAPER = "lesion_paper"         # the paper's lesion figure: six panels from the 100-seed designs
PAPER_LESION_FIG = "fig11_lesion_seeds" # replaces fig11_lesion_behaviour (two networks) as the manuscript's lesion figure
STATS = ("fusion", "segregation", "rmse")
MIN_NETWORKS = 5                       # a mean cumulative curve is drawn where this many networks reach that k
WEAK = 0.1                             # a sweep that moves a unit by less than this: one-cue unit


# --------------------------------------------------------------------------- #
# train: the seeds, through stage 2 only
# --------------------------------------------------------------------------- #
def found_runs(run, n=None):
    """The run and its trained seeds, in seed order (results/<run>_s<k>/model.pt)."""
    found = sorted(p.parent.name for p in (ROOT / "results").glob(f"{run}_s*/model.pt"))
    found.sort(key=lambda s_: int(s_.rsplit("_s", 1)[1]))
    runs = [run] + found
    return runs if n is None else runs[:n]


def collect_tuning(run, runs=None, control="pcommon1", budget=None):
    """The two tuning curves of every multisensory unit of every network of
    the seed study, cached per network (mixed_patterns/per_seed/<name>.npz)
    and merged: resp_vis and resp_prop (networks x 64 units x 41 positions;
    the visual sweep with the hand at 0 and the hand sweep with vision at 0,
    noiseless, mid-range reliability, the curves behind the congruency
    index), the index and class of every unit. Returns None while networks
    are still missing under a time budget."""
    import time
    out = out_dir(run) / K_PATTERNS / "per_seed"
    out.mkdir(parents=True, exist_ok=True)
    runs = runs or found_runs(run)
    t0, pending = time.time(), 0
    for name in runs:
        cache = out / f"{name}.npz"
        if cache.exists():
            continue
        if budget is not None and time.time() - t0 > budget:
            pending += 1
            continue
        c = Bundle(name, control=control).cong
        np.savez(cache, positions=np.asarray(c["positions"], np.float32),
                 resp_vis=np.asarray(c["resp_vis"], np.float32).T, resp_prop=np.asarray(c["resp_prop"], np.float32).T,
                 index=np.asarray(c["index"], np.float32), classes=np.asarray(c["classes"]).astype("U9"))
    if pending:
        print(f"budget reached: {pending} network(s) still to compute; run again")
        return None
    parts = [np.load(out / f"{name}.npz") for name in runs]
    return {"networks": np.array(runs), "positions": parts[0]["positions"],
            "resp_vis": np.stack([z["resp_vis"] for z in parts]), "resp_prop": np.stack([z["resp_prop"] for z in parts]),
            "index": np.stack([z["index"] for z in parts]), "classes": np.stack([z["classes"] for z in parts])}


def seed_runs(run, n):
    """[run, run_s1, ..., run_s<n-1>]: seed 0 is the run itself."""
    return [run] + [f"{run}_s{k}" for k in range(1, n)]


def train(run, n, python=sys.executable):
    base = ROOT / "results" / run
    dataset = (base / "dataset.txt").read_text().strip() if (base / "dataset.txt").exists() else run
    todo = [(k, f"{run}_s{k}") for k in range(1, n)
            if not (ROOT / "results" / f"{run}_s{k}" / "model.pt").exists()]
    print(f"seed study of {run}: {n} networks in all, {len(todo)} still to train on data/{dataset}")
    for k, name in todo:
        cmd = [python, str(ROOT / "scripts" / "02_train.py"), "--data", dataset, "--run", name,
               "--seed", str(k), "--no-figures"]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)


# --------------------------------------------------------------------------- #
# analyse: the three fit-free lesion designs on one network
# --------------------------------------------------------------------------- #
def _summaries(b, pred):
    w, _ = hybrid_reads(b, pred)
    f, s = fusion_segregation(b.d["disparity"], w)
    target = np.stack([b.d[k] for k in b.names], axis=1)
    return w, {"fusion": f, "segregation": s, "rmse": float(np.sqrt(((pred - target) ** 2).mean()))}


def _z(value, null):
    null = np.asarray(null, float)
    sd = null.std()
    return {"value": float(value), "null_mean": float(null.mean()), "null_sd": float(sd),
            "z": float((value - null.mean()) / sd) if sd > 0 else float("nan")}


CLAMPS = {"mean": "clamped to their mean activation", "zero": "set to 0 (silenced)",
          "one": "set to 1 (saturated)"}            # the units are sigmoid, so 0 and 1 are the two extremes
CLAMP_KEY = {"mean": "whole_class", "zero": "whole_class_zero", "one": "whole_class_one"}


def clamp_values(b, clamp):
    """The per-unit value a lesioned unit is held at: its mean activation over
    the test trials ("mean", the manuscript's lesion), 0 or 1."""
    if clamp == "mean":
        return b.msl.mean(0)
    n_units = b.msl.shape[1]
    return np.zeros(n_units) if clamp == "zero" else np.ones(n_units)


def whole_class(b, grid, n_random=100, seed=0, clamp="mean"):
    """Whole-class clamp, read-out frozen: the weight curve on `grid`, fusion,
    segregation and RMSE per class, each against n_random random lesions of
    the same size (one shared cache per size). `clamp` is what the lesioned
    units are held at (CLAMPS): the random lesions use the same value, so
    each z compares like with like."""
    X, ref = b.d["X"], clamp_values(b, clamp)
    ad = np.abs(b.d["disparity"])
    rng = np.random.default_rng(seed)

    def curve(w):
        c, m, _, _ = binned_weight(ad, w, grid, min_count=25)
        return c.tolist(), m.tolist()

    w0, s0 = _summaries(b, b.pred)
    out = {"intact": {**s0, "curve": curve(w0)}, "classes": {}, "random_curves": {}}
    cache = {}
    for c in CLASSES:
        idx = np.flatnonzero(b.classes == c)
        k = len(idx)
        if k == 0:
            continue
        keep = np.ones(64, bool)
        keep[idx] = False
        p, _ = b.net.lesion_predict(X, keep, ref)
        w, st = _summaries(b, p)
        if k not in cache:
            draws, curves = [], []
            for _ in range(n_random):
                rkeep = np.ones(64, bool)
                rkeep[rng.choice(64, size=k, replace=False)] = False
                pr, _ = b.net.lesion_predict(X, rkeep, ref)
                wr, sr = _summaries(b, pr)
                draws.append(sr)
                curves.append(curve(wr)[1])
            cache[k] = (draws, np.array(curves))
        draws, curves = cache[k]
        rec = {"k": int(k), "curve": curve(w)}
        for st_name in STATS:
            rec[st_name] = _z(st[st_name], [d_[st_name] for d_ in draws])
        out["classes"][c] = rec
        out["random_curves"][str(k)] = {"mean": curves.mean(0).tolist(), "sd": curves.std(0).tolist()}
    return out


def cumulative(b, n_random=30, seed=0):
    """Units of one class clamped one at a time, most typical first; fusion and
    segregation after each step, against n_random random orders over all 64."""
    X, ref = b.d["X"], b.msl.mean(0)
    idx_all = np.abs(b.cong["index"])
    sets = {c: np.flatnonzero(b.classes == c) for c in CLASSES}
    kmax = max(len(v) for v in sets.values())
    out = {"k": list(range(1, kmax + 1)), "classes": {}, "random": {}}
    for c, ix in sets.items():
        if len(ix) == 0:
            continue
        # most typical first: largest |index| for congruent and opposite units,
        # index closest to zero for intermediate units
        order = ix[np.argsort(idx_all[ix] if c == "mixed" else -idx_all[ix])]
        keep = np.ones(64, bool)
        F, S = [], []
        for u in order:
            keep[u] = False
            p, _ = b.net.lesion_predict(X, keep, ref)
            _, st = _summaries(b, p)
            F.append(st["fusion"])
            S.append(st["segregation"])
        out["classes"][c] = {"fusion": F, "segregation": S}
    rng = np.random.default_rng(seed)
    RF, RS = [], []
    for _ in range(n_random):
        order = rng.permutation(64)[:kmax]
        keep = np.ones(64, bool)
        F, S = [], []
        for u in order:
            keep[u] = False
            p, _ = b.net.lesion_predict(X, keep, ref)
            _, st = _summaries(b, p)
            F.append(st["fusion"])
            S.append(st["segregation"])
        RF.append(F)
        RS.append(S)
    RF, RS = np.array(RF), np.array(RS)
    out["random"] = {"fusion_mean": RF.mean(0).tolist(), "fusion_sd": RF.std(0).tolist(),
                     "segregation_mean": RS.mean(0).tolist(), "segregation_sd": RS.std(0).tolist()}
    # the whole-class end point of each curve against the random orders at the same k
    for c, rec in out["classes"].items():
        k = len(rec["fusion"])
        rec["final"] = {"fusion": _z(rec["fusion"][-1], RF[:, k - 1]),
                        "segregation": _z(rec["segregation"][-1], RS[:, k - 1])}
    return out


def refit(b, n_random=20, seed=0, alpha=1e-3):
    """Whole-class clamp with a ridge read-out refitted on the training split
    from the remaining units; fusion, segregation, the position slopes and
    the RMSE against n_random random lesions, each refitted."""
    from sklearn.linear_model import Ridge
    tr = np.asarray(b.ckpt["splits"]["train"])
    X_tr = b.d_full["X"][tr]
    Y_tr = np.stack([b.d_full[k][tr] for k in b.names], axis=1)
    h_tr_full = b.net.hidden(X_tr)[-1]
    h_te_full = b.msl
    ref = h_te_full.mean(0)
    d, n = b.d, b.names
    target = np.stack([d[k] for k in n], axis=1)

    def predict(keep):
        h_tr, h_te = h_tr_full.copy(), h_te_full.copy()
        h_tr[:, ~keep] = ref[~keep]
        h_te[:, ~keep] = ref[~keep]
        pred = Ridge(alpha=alpha).fit(h_tr, Y_tr).predict(h_te)
        for c in b.net.var_cols:
            pred[:, c] = np.clip(pred[:, c], 1e-3, None)
        return pred

    def stats(pred):
        w, _ = hybrid_reads(b, pred)
        f, s = fusion_segregation(d["disparity"], w)
        pv = position_regression(pred[:, n.index("mu_vis")], d["seg_vis_mu"], d["fused_mu"], d["post_c1"])["slope"]
        pp = position_regression(pred[:, n.index("mu_prop")], d["seg_prop_mu"], d["fused_mu"], d["post_c1"])["slope"]
        return {"fusion": f, "segregation": s, "posreg_vis": float(pv), "posreg_prop": float(pp),
                "rmse": float(np.sqrt(((pred - target) ** 2).mean()))}

    rng = np.random.default_rng(seed)
    out = {"intact": stats(predict(np.ones(64, bool))), "classes": {}}
    cache = {}
    for c in CLASSES:
        ix = np.flatnonzero(b.classes == c)
        k = len(ix)
        if k == 0:
            continue
        keep = np.ones(64, bool)
        keep[ix] = False
        st = stats(predict(keep))
        if k not in cache:
            draws = []
            for _ in range(n_random):
                rkeep = np.ones(64, bool)
                rkeep[rng.choice(64, k, replace=False)] = False
                draws.append(stats(predict(rkeep)))
            cache[k] = draws
        rec = {"k": int(k)}
        for kk in st:
            rec[kk] = _z(st[kk], [d_[kk] for d_ in cache[k]])
        out["classes"][c] = rec
    return out


def unit_classes(b):
    """The congruency index and class of every multisensory unit of one
    network, the class counts, and the intermediate class's sub-structure (units a
    sweep of one cue moves by less than WEAK are unimodal within the sweep)."""
    c = b.cong
    idx = np.asarray(c["index"], float)
    cl = np.asarray(b.classes)
    rng_v = c["resp_vis"].max(0) - c["resp_vis"].min(0)
    rng_p = c["resp_prop"].max(0) - c["resp_prop"].min(0)
    weak = np.minimum(rng_v, rng_p) < WEAK
    mixed = cl == "mixed"
    return {"index": idx.tolist(), "classes": cl.tolist(),
            "sweep_range_vis": rng_v.tolist(), "sweep_range_prop": rng_p.tolist(),
            "counts": b.counts,
            "mixed_substructure": {"one_cue_only": int((mixed & weak).sum()),
                                   "hand_only": int((mixed & (rng_v < WEAK)).sum()),
                                   "vision_only": int((mixed & (rng_p < WEAK)).sum()),
                                   "both_cues": int((mixed & ~weak).sum())}}


RECORD_VERSION = 2          # 2: intermediate units removed nearest-zero first in the cumulative design
PAIR_GROUPS = ("congruent+congruent", "congruent+opposite", "congruent+mixed",
               "opposite+opposite", "opposite+mixed", "mixed+mixed")


def damage_matched(b, n_matched=60, tol=0.05, max_draws=1200, seed=1):
    """The whole-class clamp against a DAMAGE-matched null: random lesions of
    the class's size whose read-out RMSE lies within `tol` (relative) of the
    class lesion's own RMSE, so that fusion and segregation are compared at
    equal read-out damage rather than at equal unit count. Rejection
    sampling up to max_draws per class; the record says how many matched."""
    X, ref = b.d["X"], b.msl.mean(0)
    rng = np.random.default_rng(seed)
    out = {"tol": tol, "classes": {}}
    for c in CLASSES:
        idx = np.flatnonzero(b.classes == c)
        k = len(idx)
        if k == 0:
            continue
        keep = np.ones(64, bool)
        keep[idx] = False
        p, _ = b.net.lesion_predict(X, keep, ref)
        _, st = _summaries(b, p)
        target = st["rmse"]
        matched, tried = [], 0
        while len(matched) < n_matched and tried < max_draws:
            tried += 1
            rkeep = np.ones(64, bool)
            rkeep[rng.choice(64, size=k, replace=False)] = False
            pr, _ = b.net.lesion_predict(X, rkeep, ref)
            _, sr = _summaries(b, pr)
            if abs(sr["rmse"] - target) <= tol * target:
                matched.append(sr)
        rec = {"k": int(k), "rmse": float(target), "n_matched": len(matched), "n_tried": tried}
        if len(matched) >= 10:
            for s_ in ("fusion", "segregation"):
                rec[s_] = _z(st[s_], [m[s_] for m in matched])
            rec["rmse_matched_mean"] = float(np.mean([m["rmse"] for m in matched]))
        out["classes"][c] = rec
    return out


def pair_group(c1, c2):
    """The class pair in canonical order; pairs with an untuned unit (rare:
    0.0 ± 0.1 per network) fall outside the six PAIR_GROUPS and only count in
    all_pairs."""
    a, b_ = sorted((c1, c2), key=(CLASSES + ("untuned",)).index)
    return f"{a}+{b_}"


def pairs(b):
    """Every unit alone and every pair of units clamped: the change in fusion
    and in segregation, grouped by the classes of the pair, and the synergy of
    each pair (its effect minus the sum of its two single-unit effects)."""
    from itertools import combinations
    X, ref = b.d["X"], b.msl.mean(0)
    _, s0 = _summaries(b, b.pred)
    f0, seg0 = s0["fusion"], s0["segregation"]
    dF1, dS1 = np.zeros(64), np.zeros(64)
    for u in range(64):
        keep = np.ones(64, bool)
        keep[u] = False
        p, _ = b.net.lesion_predict(X, keep, ref)
        _, st = _summaries(b, p)
        dF1[u], dS1[u] = st["fusion"] - f0, st["segregation"] - seg0
    cl = np.asarray(b.classes)
    U, V, dF, dS = [], [], [], []
    for u, v in combinations(range(64), 2):
        keep = np.ones(64, bool)
        keep[[u, v]] = False
        p, _ = b.net.lesion_predict(X, keep, ref)
        _, st = _summaries(b, p)
        U.append(u); V.append(v)
        dF.append(st["fusion"] - f0); dS.append(st["segregation"] - seg0)
    U, V, dF, dS = map(np.asarray, (U, V, dF, dS))
    synF = dF - (dF1[U] + dF1[V])
    synS = dS - (dS1[U] + dS1[V])
    groups = np.array([pair_group(cl[u], cl[v]) for u, v in zip(U, V)])
    out = {"intact": {"fusion": f0, "segregation": seg0},
           "single": {"dF": dF1.tolist(), "dS": dS1.tolist()},
           "pairs": {"u": U.tolist(), "v": V.tolist(), "dF": np.round(dF, 5).tolist(),
                     "dS": np.round(dS, 5).tolist()},
           "groups": {}}
    for g in PAIR_GROUPS:
        m = groups == g
        if not m.any():
            continue
        out["groups"][g] = {"n": int(m.sum()),
                            "dF_mean": float(dF[m].mean()), "dS_mean": float(dS[m].mean()),
                            "dF_abs_mean": float(np.abs(dF[m]).mean()), "dS_abs_mean": float(np.abs(dS[m]).mean()),
                            "synF_mean": float(synF[m].mean()), "synS_mean": float(synS[m].mean())}
    out["all_pairs"] = {"dF_mean": float(dF.mean()), "dS_mean": float(dS.mean()),
                        "dF_sd": float(dF.std()), "dS_sd": float(dS.std())}
    # the ten pairs with the largest loss of segregation and of fusion
    for key, arr in (("top_segregation_loss", dS), ("top_fusion_loss", dF)):
        order = np.argsort(arr)[:10]
        out[key] = [{"u": int(U[i]), "v": int(V[i]), "classes": [str(cl[U[i]]), str(cl[V[i]])],
                     "dF": float(dF[i]), "dS": float(dS[i])} for i in order]
    return out


def analyse_one(name, grid, control, extras=(), record=None):
    """The record of one network. With `record` (a cached one), only the parts
    that are missing or out of date are recomputed; `extras` names the
    optional designs to add ("damage", "pairs")."""
    b = None

    def bundle():
        nonlocal b
        if b is None:
            b = Bundle(name, control=control)
        return b

    rec = dict(record or {})
    fresh = not rec or rec.get("version", 1) < RECORD_VERSION
    if fresh:
        b_ = bundle()
        rec.update({"run": name, "p_common": b_.p_common, "seed": int(b_.cfg.get("seed", -1)),
                    "counts": b_.counts, "units": unit_classes(b_), "cumulative": cumulative(b_)})
        if "whole_class" not in rec:
            rec["whole_class"] = whole_class(b_, grid)
        if "refit" not in rec:
            rec["refit"] = refit(b_)
        rec["version"] = RECORD_VERSION
    if "damage" in extras and "damage_matched" not in rec:
        rec["damage_matched"] = damage_matched(bundle())
    if "pairs" in extras and "pairs" not in rec:
        rec["pairs"] = pairs(bundle())
    if "clamp" in extras:
        for mode in ("zero", "one"):
            if CLAMP_KEY[mode] not in rec:
                rec[CLAMP_KEY[mode]] = whole_class(bundle(), grid, clamp=mode)
    return rec, b is not None


EXTRA_KEYS = {"damage": ("damage_matched",), "pairs": ("pairs",),
              "clamp": (CLAMP_KEY["zero"], CLAMP_KEY["one"])}     # what each --extras adds to a record


def analyse(run, runs=None, n=None, control="pcommon1", extras=(), budget=None):
    """Analyse the run's seeds (or `runs`), reusing per_seed caches and
    completing them where a part is missing or out of date. `extras` adds
    the optional designs ("damage", "pairs"). `budget` (seconds) stops
    computing new records after that long and aggregates what exists, so a
    long job can be done in several calls."""
    import time
    out = out_dir(run) / K_SEEDS
    (out / "per_seed").mkdir(parents=True, exist_ok=True)
    if runs is None:
        found = sorted(p.parent.name for p in (ROOT / "results").glob(f"{run}_s*/model.pt"))
        found.sort(key=lambda s_: int(s_.rsplit("_s", 1)[1]))
        runs = [run] + found
        if n is not None:
            runs = runs[:n]
    base = Bundle(run, control=control)
    grid = np.unique(np.abs(np.asarray(base.cfg["analysis"]["disparity_grid"], float))).tolist()
    records, t0, pending = [], time.time(), 0
    for i, name in enumerate(runs):
        cache = out / "per_seed" / f"{name}.json"
        rec = json.loads(cache.read_text()) if cache.exists() else None
        complete = (rec is not None and rec.get("version", 1) >= RECORD_VERSION
                    and all(k in rec for e in extras for k in EXTRA_KEYS[e]))
        if complete:
            records.append(rec)
            continue
        if budget is not None and time.time() - t0 > budget:
            pending += 1
            if rec is not None:
                records.append(rec)
            continue
        print(f"[{i + 1}/{len(runs)}] {name}" + (" (completing)" if rec else ""), flush=True)
        rec, _ = analyse_one(name, grid, control, extras=extras, record=rec)
        save_json(rec, cache)
        records.append(rec)
    if pending:
        print(f"budget reached: {pending} network(s) still to compute or complete; run again")
    agg = aggregate(records, grid)
    agg["units"] = aggregate_units(records)
    agg["damage_matched"] = aggregate_damage(records)
    agg["pairs"] = aggregate_pairs(records)
    save_json({"run": run, "control": control, "grid": grid, "networks": [r["run"] for r in records],
               "n": len(records), "aggregate": agg}, out / "numbers.json")
    write_counts_csv(records, out / "unit_counts_by_seed.csv")
    print_summary(agg, len(records))
    return records, agg


def aggregate_units(records):
    """Across-seed distribution of the unit classes: counts per class (with
    percentiles), the share of networks with more opposite than congruent
    units, the intermediate sub-structure, and the congruency index pooled over
    every unit of every network, binned."""
    n = len(records)
    out = {"n": n, "counts": {}, "mixed_substructure": {}}
    for c in CLASSES + ("untuned",):
        v = np.array([r["units"]["counts"][f"n_{c}"] for r in records], float)
        out["counts"][c] = {**_dist(v), "min": float(v.min()), "max": float(v.max()),
                            "median": float(np.median(v)),
                            "p2.5": float(np.percentile(v, 2.5)), "p97.5": float(np.percentile(v, 97.5))}
    opp = np.array([r["units"]["counts"]["n_opposite"] for r in records])
    con = np.array([r["units"]["counts"]["n_congruent"] for r in records])
    out["share_opposite_gt_congruent"] = float(np.mean(opp > con))
    out["share_opposite_ge_congruent"] = float(np.mean(opp >= con))
    out["opposite_minus_congruent"] = _dist(opp - con)
    for k in ("one_cue_only", "hand_only", "vision_only", "both_cues"):
        out["mixed_substructure"][k] = _dist([r["units"]["mixed_substructure"][k] for r in records])
    idx = np.concatenate([r["units"]["index"] for r in records])
    edges = np.linspace(-1, 1, 41)
    hist, _ = np.histogram(idx, bins=edges)
    out["index_pooled"] = {"edges": edges.tolist(), "counts": hist.tolist(), "n_units": int(len(idx)),
                           "mean": float(idx.mean()), "sd": float(idx.std()),
                           "share_congruent": float(np.mean(idx > 0.5)), "share_opposite": float(np.mean(idx < -0.5)),
                           "share_mixed": float(np.mean(np.abs(idx) <= 0.5))}
    return out


def aggregate_damage(records):
    """Across-seed summary of the damage-matched null (networks that have it)."""
    recs = [r["damage_matched"] for r in records if "damage_matched" in r]
    if not recs:
        return None
    out = {"n": len(recs), "classes": {}}
    for c in CLASSES:
        have = [r["classes"][c] for r in recs if c in r["classes"] and "fusion" in r["classes"][c]]
        if not have:
            continue
        out["classes"][c] = {"n": len(have),
                             "n_matched": _dist([h["n_matched"] for h in have]),
                             "rmse": _dist([h["rmse"] for h in have]),
                             "rmse_matched_mean": _dist([h["rmse_matched_mean"] for h in have]),
                             **{s_: {"value": _dist([h[s_]["value"] for h in have]),
                                     "null_mean": _dist([h[s_]["null_mean"] for h in have]),
                                     "z": _dist([h[s_]["z"] for h in have])} for s_ in ("fusion", "segregation")}}
    return out


def aggregate_pairs(records):
    """Across-seed summary of the pair lesions: per class pair, the mean change
    in fusion and in segregation and the mean synergy, as distributions over
    networks; plus the single-unit effects per class."""
    recs = [r for r in records if "pairs" in r]
    if not recs:
        return None
    out = {"n": len(recs), "groups": {}, "single": {}}
    for g in PAIR_GROUPS:
        have = [r["pairs"]["groups"][g] for r in recs if g in r["pairs"]["groups"]]
        if not have:
            continue
        out["groups"][g] = {"n_networks": len(have), "n_pairs": _dist([h["n"] for h in have]),
                            **{k: _dist([h[k] for h in have])
                               for k in ("dF_mean", "dS_mean", "dF_abs_mean", "dS_abs_mean", "synF_mean", "synS_mean")}}
    for c in CLASSES:
        dF = np.concatenate([np.asarray(r["pairs"]["single"]["dF"])[np.asarray(r["units"]["classes"]) == c] for r in recs])
        dS = np.concatenate([np.asarray(r["pairs"]["single"]["dS"])[np.asarray(r["units"]["classes"]) == c] for r in recs])
        out["single"][c] = {"n_units": int(len(dF)), "dF": _dist(dF), "dS": _dist(dS)}
        # per network: the class mean of the single-unit effects (a size-free
        # comparison of the classes), and the sum over the class's units
        # against the whole-class clamp of the same network (additivity up to
        # the class; the read-out is linear in the layer, so the position
        # outputs add exactly and the weight statistics nearly so)
        mean_f, mean_s, sum_f, sum_s, whole_f, whole_s = [], [], [], [], [], []
        for r in recs:
            m = np.asarray(r["units"]["classes"]) == c
            wc = r["whole_class"]["classes"].get(c)
            if not m.any() or wc is None:
                continue
            f1 = np.asarray(r["pairs"]["single"]["dF"])[m]
            s1 = np.asarray(r["pairs"]["single"]["dS"])[m]
            mean_f.append(f1.mean()); mean_s.append(s1.mean())
            sum_f.append(f1.sum()); sum_s.append(s1.sum())
            whole_f.append(wc["fusion"]["value"] - r["pairs"]["intact"]["fusion"])
            whole_s.append(wc["segregation"]["value"] - r["pairs"]["intact"]["segregation"])
        out["single"][c]["per_network_mean"] = {"dF": _dist(mean_f), "dS": _dist(mean_s)}
        out["single"][c]["sum_vs_whole"] = {"sum_dF": [float(v) for v in sum_f], "sum_dS": [float(v) for v in sum_s],
                                            "whole_dF": [float(v) for v in whole_f],
                                            "whole_dS": [float(v) for v in whole_s]}
    out["all_pairs"] = {k: _dist([r["pairs"]["all_pairs"][k] for r in recs]) for k in ("dF_mean", "dS_mean")}
    # the class composition of the ten worst pairs for segregation, pooled over networks
    comp = {}
    for r in recs:
        for item in r["pairs"]["top_segregation_loss"]:
            g = pair_group(*item["classes"])
            comp[g] = comp.get(g, 0) + 1
    out["top10_segregation_loss_composition"] = comp
    comp = {}
    for r in recs:
        for item in r["pairs"]["top_fusion_loss"]:
            g = pair_group(*item["classes"])
            comp[g] = comp.get(g, 0) + 1
    out["top10_fusion_loss_composition"] = comp
    return out


def write_counts_csv(records, path):
    cols = ("run", "seed", "p_common", "congruent", "opposite", "mixed", "untuned",
            "mixed_one_cue_only", "mixed_hand_only", "mixed_vision_only", "mixed_both_cues")
    lines = [",".join(cols)]
    for r in records:
        u = r["units"]
        lines.append(",".join(str(x) for x in (
            r["run"], r["seed"], r["p_common"], u["counts"]["n_congruent"], u["counts"]["n_opposite"],
            u["counts"]["n_mixed"], u["counts"]["n_untuned"], u["mixed_substructure"]["one_cue_only"],
            u["mixed_substructure"]["hand_only"], u["mixed_substructure"]["vision_only"],
            u["mixed_substructure"]["both_cues"])))
    Path(path).write_text("\n".join(lines) + "\n")


def aggregate(records, grid):
    """Across-seed mean / sd of every per-network quantity the figure draws."""
    n = len(records)
    agg = {"n": n, "counts": {}, "whole_class": {"intact": {}, "classes": {}}, "cumulative": {}, "refit": {}}
    for c in CLASSES:
        v = np.array([r["counts"][f"n_{c}"] for r in records], float)
        agg["counts"][c] = {"values": v.tolist(), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if n > 1 else 0.0}
    # whole-class clamp: curves (on the common grid) and z-scores
    def stack_curve(getter, recs_all=None):
        rows = []
        for r in (records if recs_all is None else recs_all):
            c, m = getter(r)
            row = np.full(len(grid), np.nan)
            for ci, mi in zip(c, m):
                row[grid.index(ci)] = mi
            rows.append(row)
        rows = np.array(rows)
        return {"mean": np.nanmean(rows, 0).tolist(),
                "sd": (np.nanstd(rows, 0, ddof=1) if len(rows) > 1 else np.zeros(len(grid))).tolist()}
    def whole_class_block(key, recs_all):
        wc = {"intact": {"curve": stack_curve(lambda r: r[key]["intact"]["curve"], recs_all),
                         **{s_: _dist([r[key]["intact"][s_] for r in recs_all]) for s_ in STATS}},
              "classes": {}, "random_curve": {}, "n": len(recs_all)}
        for c in CLASSES:
            recs = [r[key]["classes"][c] for r in recs_all if c in r[key]["classes"]]
            if not recs:
                continue
            entry = {"n": len(recs), "k": _dist([r_["k"] for r_ in recs]),
                     "curve": stack_curve(lambda r, c=c: r[key]["classes"][c]["curve"], recs_all)}
            for s_ in STATS:
                entry[s_] = {"value": _dist([r_[s_]["value"] for r_ in recs]),
                             "z": _dist([r_[s_]["z"] for r_ in recs]),
                             "null_mean": _dist([r_[s_]["null_mean"] for r_ in recs])}
            wc["classes"][c] = entry
            # the random curve of the class's size, averaged across seeds
            rc = np.array([r[key]["random_curves"][str(r[key]["classes"][c]["k"])]["mean"]
                           for r in recs_all if c in r[key]["classes"]])
            wc["random_curve"][c] = {"mean": rc.mean(0).tolist()}
        return wc

    agg["whole_class"] = whole_class_block("whole_class", records)
    for mode in ("zero", "one"):                      # the 0 / 1 clamps, where stored (--extras clamp)
        have = [r for r in records if CLAMP_KEY[mode] in r]
        if have:
            agg[CLAMP_KEY[mode]] = whole_class_block(CLAMP_KEY[mode], have)
    # cumulative: end-point z-scores and the mean curves (padded to the longest class)
    for c in CLASSES:
        recs = [r["cumulative"]["classes"][c] for r in records if c in r["cumulative"]["classes"]]
        if not recs:
            continue
        kmax = max(len(r_["fusion"]) for r_ in recs)
        curves = {}
        for s_ in ("fusion", "segregation"):
            M = np.full((len(recs), kmax), np.nan)
            for i, r_ in enumerate(recs):
                M[i, :len(r_[s_])] = r_[s_]
            curves[s_] = {"mean": np.nanmean(M, 0).tolist(),
                          "sd": (np.nanstd(M, 0, ddof=1) if len(recs) > 1 else np.zeros(kmax)).tolist(),
                          "n_at_k": np.sum(~np.isnan(M), 0).tolist()}
        agg["cumulative"][c] = {"curves": curves,
                                **{s_: {"value": _dist([r_["final"][s_]["value"] for r_ in recs]),
                                        "z": _dist([r_["final"][s_]["z"] for r_ in recs])}
                                   for s_ in ("fusion", "segregation")}}
    kmax = max(len(r["cumulative"]["k"]) for r in records)
    agg["cumulative"]["random"] = {}
    for s_ in ("fusion", "segregation"):
        P = np.full((n, kmax), np.nan)       # the random-order mean curve of each network, padded
        for i, r in enumerate(records):
            row = r["cumulative"]["random"][f"{s_}_mean"]
            P[i, :len(row)] = row
        agg["cumulative"]["random"][s_] = {"mean": np.nanmean(P, 0).tolist(),
                                           "n_at_k": np.sum(~np.isnan(P), 0).tolist()}
    # the cumulative design read at a COMMON number of removed units (equal dose
    # across classes): k = 10, and k = the smallest class of each network
    agg["cumulative"]["equal_k"] = {}
    for label, pick in (("k10", lambda r: 10), ("k_min_class", lambda r: min(len(v["fusion"]) for v in r["cumulative"]["classes"].values()))):
        entry = {"k": _dist([pick(r) for r in records]), "classes": {}}
        for c in CLASSES:
            zs = {"fusion": [], "segregation": []}
            vals = {"fusion": [], "segregation": []}
            for r in records:
                cu = r["cumulative"]
                if c not in cu["classes"]:
                    continue
                k = min(pick(r), len(cu["classes"][c]["fusion"]))
                for s_ in ("fusion", "segregation"):
                    v = cu["classes"][c][s_][k - 1]
                    rm, rs = cu["random"][f"{s_}_mean"][k - 1], cu["random"][f"{s_}_sd"][k - 1]
                    vals[s_].append(v)
                    zs[s_].append((v - rm) / rs if rs > 0 else np.nan)
            entry["classes"][c] = {s_: {"value": _dist(vals[s_]), "z": _dist(zs[s_])} for s_ in ("fusion", "segregation")}
        agg["cumulative"]["equal_k"][label] = entry
    # refit
    agg["refit"]["intact"] = {s_: _dist([r["refit"]["intact"][s_] for r in records])
                              for s_ in ("fusion", "segregation", "posreg_vis", "posreg_prop", "rmse")}
    for c in CLASSES:
        recs = [r["refit"]["classes"][c] for r in records if c in r["refit"]["classes"]]
        if not recs:
            continue
        agg["refit"][c] = {s_: {"value": _dist([r_[s_]["value"] for r_ in recs]),
                                "z": _dist([r_[s_]["z"] for r_ in recs]),
                                "null_mean": _dist([r_[s_]["null_mean"] for r_ in recs])}
                           for s_ in ("fusion", "segregation", "posreg_vis", "posreg_prop", "rmse")}
    return agg


def _dist(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    n = len(v)
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    return {"n": int(n), "values": v.tolist(), "mean": float(v.mean()) if n else float("nan"), "sd": sd,
            "se": float(sd / np.sqrt(n)) if n > 1 else float("nan"),
            "t": float(v.mean() / (sd / np.sqrt(n))) if n > 1 and sd > 0 else float("nan"),
            "frac_beyond_2": float(np.mean(np.abs(v) > 2)) if n else float("nan")}


def print_summary(agg, n):
    print(f"\nacross {n} network(s): within-network z of each class lesion, mean ± sd over seeds "
          f"(t = mean / se against 0; share of seeds with |z| > 2)")
    for design, key in (("whole-class clamp, frozen", "whole_class"), ("cumulative, end point", "cumulative"),
                        ("whole-class clamp, refitted", "refit")):
        print(f"  {design}:")
        for c in CLASSES:
            src = agg[key]["classes"][c] if key == "whole_class" else agg[key].get(c)
            if not src:
                continue
            parts = []
            for s_ in (STATS if key == "whole_class" else ("fusion", "segregation") if key == "cumulative"
                       else ("fusion", "segregation", "posreg_vis", "posreg_prop", "rmse")):
                z = src[s_]["z"]
                parts.append(f"{s_} {z['mean']:+.2f} ± {z['sd']:.2f} (t {z['t']:+.1f}, {100 * z['frac_beyond_2']:.0f}%)")
            print(f"    -{c:9s} " + "; ".join(parts))
    for mode in ("zero", "one"):
        wc = agg.get(CLAMP_KEY[mode])
        if not wc:
            continue
        print(f"  whole-class clamp, frozen, units {CLAMPS[mode]} ({wc['n']} networks; random lesions held at the same value):")
        for c in CLASSES:
            src = wc["classes"].get(c)
            if src:
                print(f"    -{c:9s} " + "; ".join(
                    f"{s_} {src[s_]['z']['mean']:+.2f} ± {src[s_]['z']['sd']:.2f} (t {src[s_]['z']['t']:+.1f}, "
                    f"{100 * src[s_]['z']['frac_beyond_2']:.0f}%; value {src[s_]['value']['mean']:.3f} vs {src[s_]['null_mean']['mean']:.3f})"
                    for s_ in STATS))
    eq = agg["cumulative"].get("equal_k")
    if eq:
        print("  cumulative at EQUAL k (same number removed from every class):")
        for label, entry in eq.items():
            print(f"    {label} (k = {entry['k']['mean']:.1f} ± {entry['k']['sd']:.1f}):")
            for c in CLASSES:
                e = entry["classes"].get(c)
                if e:
                    print(f"      -{c:9s} " + "; ".join(f"{s_} {e[s_]['z']['mean']:+.2f} ± {e[s_]['z']['sd']:.2f} (t {e[s_]['z']['t']:+.1f})"
                                                       for s_ in ("fusion", "segregation")))
    dm = agg.get("damage_matched")
    if dm:
        print(f"  whole-class clamp against a DAMAGE-matched null (same k AND same RMSE within tolerance; {dm['n']} networks):")
        for c in CLASSES:
            e = dm["classes"].get(c)
            if e:
                print(f"    -{c:9s} matched {e['n_matched']['mean']:.0f} draws; " + "; ".join(
                    f"{s_} {e[s_]['z']['mean']:+.2f} ± {e[s_]['z']['sd']:.2f} (t {e[s_]['z']['t']:+.1f})"
                    for s_ in ("fusion", "segregation")))
    pr = agg.get("pairs")
    if pr:
        print(f"  pairs of units clamped ({pr['n']} networks), change from intact, mean over pairs then over networks:")
        for g in PAIR_GROUPS:
            e = pr["groups"].get(g)
            if e:
                print(f"    {g:21s} n {e['n_pairs']['mean']:5.0f}  dF {e['dF_mean']['mean']:+.4f} ± {e['dF_mean']['sd']:.4f}  "
                      f"dS {e['dS_mean']['mean']:+.4f} ± {e['dS_mean']['sd']:.4f}  synergy F {e['synF_mean']['mean']:+.4f}  "
                      f"S {e['synS_mean']['mean']:+.4f}")
        print("    single units, mean dF / dS per class: " + "; ".join(
            f"{CLASS_LABEL[c]} {pr['single'][c]['dF']['mean']:+.4f} / {pr['single'][c]['dS']['mean']:+.4f}" for c in CLASSES if c in pr["single"]))
        print("    class composition of the 10 worst pairs per network: segregation", pr["top10_segregation_loss_composition"],
              "fusion", pr["top10_fusion_loss_composition"])
    u = agg.get("units")
    if u:
        print(f"\nunit classes of the multisensory layer over {n} network(s) (64 units each):")
        print("  class      mean    sd   median   min  max   [2.5, 97.5] %")
        for c in CLASSES + ("untuned",):
            d = u["counts"][c]
            print(f"  {c:9s} {d['mean']:6.1f} {d['sd']:5.1f} {d['median']:7.1f} {d['min']:5.0f} {d['max']:4.0f}   "
                  f"[{d['p2.5']:.0f}, {d['p97.5']:.0f}]")
        print(f"  opposite > congruent in {100 * u['share_opposite_gt_congruent']:.0f}% of networks "
              f"(opposite − congruent {u['opposite_minus_congruent']['mean']:+.1f} ± {u['opposite_minus_congruent']['sd']:.1f})")
        ms = u["mixed_substructure"]
        print(f"  intermediate sub-structure: one-cue-only {ms['one_cue_only']['mean']:.1f} ± {ms['one_cue_only']['sd']:.1f} "
              f"(hand-only {ms['hand_only']['mean']:.1f}, vision-only {ms['vision_only']['mean']:.1f}), "
              f"both cues {ms['both_cues']['mean']:.1f} ± {ms['both_cues']['sd']:.1f}")
        ip = u["index_pooled"]
        print(f"  congruency index pooled over {ip['n_units']} units: mean {ip['mean']:+.2f}, sd {ip['sd']:.2f}; "
              f"shares congruent {100 * ip['share_congruent']:.0f}%, opposite {100 * ip['share_opposite']:.0f}%, "
              f"intermediate {100 * ip['share_mixed']:.0f}%")
    else:
        print("  unit counts, mean ± sd: " + ", ".join(f"{CLASS_LABEL[c]} {agg['counts'][c]['mean']:.1f} ± {agg['counts'][c]['sd']:.1f}"
                                                     for c in CLASSES))


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def figure(run, agg=None, numbers=None):
    out = out_dir(run) / K_SEEDS
    if agg is None:
        numbers = numbers or json.loads((out / "numbers.json").read_text())
        agg = numbers["aggregate"]
    n = agg["n"]
    grid = np.asarray(numbers["grid"] if numbers else json.loads((out / "numbers.json").read_text())["grid"], float)
    label = f"{n} networks" if n > 1 else "1 network"

    def band(ax, x, m, s, color, lw=PANEL_LW, label_=None, alpha=0.18, ls="-"):
        m, s = np.asarray(m, float), np.asarray(s, float)
        ax.fill_between(x, m - s, m + s, color=color, alpha=alpha, lw=0)
        ax.plot(x, m, ls, lw=lw, color=color, label=label_)

    def pA(ax=None):
        ax = panel_axes(ax)
        wc = agg["whole_class"]
        band(ax, grid, wc["intact"]["curve"]["mean"], wc["intact"]["curve"]["sd"], "k", lw=PANEL_LW + 0.4,
             label_="intact")
        for c in CLASSES:
            if c in wc["classes"]:
                band(ax, grid, wc["classes"][c]["curve"]["mean"], wc["classes"][c]["curve"]["sd"], CLASS_COLORS[c],
                     label_=f"−{CLASS_LABEL[c]}")
        # the size-matched random mean of each class (its own k), dotted in
        # the class colour, so no class is held to another class's null
        for c in CLASSES:
            if c in wc["random_curve"]:
                ax.plot(grid, wc["random_curve"][c]["mean"], ":", lw=1.0, color=CLASS_COLORS[c])
        ax.plot([], [], ":", lw=1.0, color="gray", label="random, same k")
        ax.set_ylim(-0.1, 1.8)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        fig = finish_panel(ax, "|disparity| (deg)", "weight on fused estimate",
                           f"whole-class clamp ({label})", None)
        ax.legend(fontsize=7, loc="upper center", ncol=3, handlelength=1.2, columnspacing=0.6,
                  borderaxespad=0.2, handletextpad=0.4)
        return fig

    def zbox(ax, source, stats_, labels, title, headroom=0.3):
        x0 = 0
        ticks, ticklabels = [], []
        allv = []
        for s_, lab in zip(stats_, labels):
            for i, c in enumerate(CLASSES):
                src = source(c)
                if not src:
                    continue
                v = np.asarray(src[s_]["z"]["values"], float)
                allv.extend(v.tolist())
                pos = x0 + i
                if len(v) > 1:
                    ax.boxplot([v], positions=[pos], widths=0.6, showfliers=False,
                               medianprops=dict(color="k", lw=1.0),
                               boxprops=dict(color=CLASS_COLORS[c], lw=1.0),
                               whiskerprops=dict(color=CLASS_COLORS[c], lw=0.8),
                               capprops=dict(color=CLASS_COLORS[c], lw=0.8))
                jitter = (np.random.default_rng(1).uniform(-0.18, 0.18, len(v)) if len(v) > 1 else [0.0])
                ax.plot(pos + jitter, v, "o", ms=2.2, color=CLASS_COLORS[c], alpha=0.6)
            ticks.append(x0 + 1)
            ticklabels.append(lab)
            x0 += 4
        for y in (-2, 2):
            ax.axhline(y, ls=":", lw=0.7, color="gray")
        ax.axhline(0, lw=0.5, color="gray")
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels)
        ax.set_xlim(-1, x0 - 1)
        lo = min(-2.6, (min(allv) if allv else -2.6) - 0.3)
        hi = max(2.6, (max(allv) if allv else 2.6)) + headroom
        ax.set_ylim(lo, hi)
        return finish_panel(ax, "", "z vs random lesions", title, None)

    def pB(ax=None):
        ax = panel_axes(ax)
        fig = zbox(ax, lambda c: agg["whole_class"]["classes"].get(c), ("fusion", "segregation", "rmse"),
                   ("fusion", "segregation", "RMSE"), "frozen: whole-class clamp", headroom=1.4)
        for i, c in enumerate(CLASSES):
            ax.plot([], [], "o", ms=3, color=CLASS_COLORS[c], label=f"−{CLASS_LABEL[c]}")
        ax.legend(fontsize=7, loc="upper left", ncol=3, handlelength=1.0, columnspacing=0.6, borderaxespad=0.2)
        return fig

    def pC(ax=None):
        # the same whole-class clamp as B, scored against random lesions of the
        # same size whose RMSE also matches the class lesion's (within 5 %)
        ax = panel_axes(ax)
        dm = agg.get("damage_matched") or {"classes": {}}
        fig = zbox(ax, lambda c: dm["classes"].get(c), ("fusion", "segregation"),
                   ("fusion", "segregation"), "frozen: damage-matched null")
        ax.set_ylabel("z vs RMSE-matched lesions", fontsize=PANEL_FONT["label"])
        return fig

    def pD(ax=None):
        # the cumulative curves read at one k for every class: the smallest
        # class of each network (an equal dose, against random orders at that k)
        ax = panel_axes(ax)
        eq = (agg["cumulative"].get("equal_k") or {}).get("k_min_class") or {"classes": {}, "k": {}}
        k_ = eq.get("k") or {}
        title = ("cumulative at equal k" + (f" (k = {k_['mean']:.0f} ± {k_['sd']:.0f})" if k_ else ""))
        fig = zbox(ax, lambda c: eq["classes"].get(c), ("fusion", "segregation"),
                   ("fusion", "segregation"), title)
        ax.set_ylabel("z vs random orders, same k", fontsize=PANEL_FONT["label"])
        return fig

    def pE(ax=None):
        ax = panel_axes(ax)
        return zbox(ax, lambda c: agg["cumulative"].get(c), ("fusion", "segregation"),
                    ("fusion", "segregation"), "frozen: cumulative, end point")

    def pF(ax=None):
        ax = panel_axes(ax)
        return zbox(ax, lambda c: agg["refit"].get(c), ("fusion", "segregation", "posreg_vis", "posreg_prop"),
                    ("fusion", "segre-\ngation", "slope\nvis", "slope\nhand"), "refitted read-out")

    def pG(ax=None):
        ax = panel_axes(ax)
        # mean ± sd over the networks that still have a unit of the class at
        # that k; drawn where at least MIN_NETWORKS of them do, so the tail is
        # not one or two networks' curve
        cu = agg["cumulative"]
        for c in CLASSES:
            if c in cu:
                cs = cu[c]["curves"]["segregation"]
                m, s, cnt = (np.asarray(cs[key], float) for key in ("mean", "sd", "n_at_k"))
                keep = cnt >= min(MIN_NETWORKS, cnt.max())
                k = np.arange(1, len(m) + 1)
                band(ax, k[keep], m[keep], s[keep], CLASS_COLORS[c], label_=f"{CLASS_LABEL[c]} first")
        rnd = cu["random"]["segregation"]
        rm = np.asarray(rnd["mean"], float)
        cnt = np.asarray(rnd.get("n_at_k", np.full(len(rm), n)), float)
        keep = cnt >= min(MIN_NETWORKS, cnt.max())
        ax.plot(np.arange(1, len(rm) + 1)[keep], rm[keep], ":", lw=1.0, color="gray", label="random order")
        ax.set_ylim(-0.05, 1.05)
        return finish_panel(ax, "units removed (most typical first)", "segregation at |d| ≥ 15°",
                            f"cumulative lesion ({label})", "lower left")

    def pH(ax=None):
        ax = panel_axes(ax)
        x = np.arange(len(CLASSES))
        for i, c in enumerate(CLASSES):
            v = np.asarray(agg["counts"][c]["values"], float)
            if len(v) > 1:
                ax.boxplot([v], positions=[i], widths=0.6, showfliers=False,
                           medianprops=dict(color="k", lw=1.0),
                           boxprops=dict(color=CLASS_COLORS[c], lw=1.0),
                           whiskerprops=dict(color=CLASS_COLORS[c], lw=0.8),
                           capprops=dict(color=CLASS_COLORS[c], lw=0.8))
            jitter = np.random.default_rng(2).uniform(-0.18, 0.18, len(v)) if len(v) > 1 else [0.0]
            ax.plot(i + jitter, v, "o", ms=2.2, color=CLASS_COLORS[c], alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([CLASS_LABEL[c] for c in CLASSES])
        ax.set_ylim(0, 64)
        return finish_panel(ax, "", "units of 64", f"unit classes ({label})", None)

    draws = [pA, pB, pC, pD, pE, pF, pG, pH]
    figs = {f"{K_SEEDS}/{K_SEEDS}": manuscript_grid(draws, ncols=4)}
    for key, fn in zip(("A_whole_class_curves", "B_whole_class_z", "C_damage_matched_z", "D_equal_k_z",
                        "E_cumulative_z", "F_refit_z", "G_cumulative_curves", "H_unit_counts"), draws):
        figs[f"{K_SEEDS}/{key}"] = fn()
    # the eight-panel diagnostic stays beside the analysis; the manuscript gets
    # the supplementary figure without the panels of Figs. 9 and 10
    save_figures(figs, out_dir(run), formats=FORMATS)
    print(f"   -> {out}/")
    si_lesion_figure(run, agg, label)
    if agg.get("units"):
        unit_figure(run, agg["units"], label)
    if agg.get("pairs"):
        pairs_figure(run, agg["pairs"], label)
    if agg.get(CLAMP_KEY["zero"]) and agg.get(CLAMP_KEY["one"]):
        clamp_figure(run, agg, label, grid)


def _strip(ax, pos, v, color, seed_):
    v = np.asarray(v, float)
    if len(v) > 1:
        ax.boxplot([v], positions=[pos], widths=0.6, showfliers=False,
                   medianprops=dict(color="k", lw=1.0), boxprops=dict(color=color, lw=1.0),
                   whiskerprops=dict(color=color, lw=0.8), capprops=dict(color=color, lw=0.8))
    jitter = np.random.default_rng(seed_).uniform(-0.18, 0.18, len(v)) if len(v) > 1 else [0.0]
    ax.plot(pos + jitter, v, "o", ms=2.2, color=color, alpha=0.6)


def unit_class_panels(u, label, narrow=False):
    """The three unit-class panels of the seeds, as draw functions `fn(ax=None)`
    (shared by `unit_figure` and `units_paper_figure`): A the congruency index
    pooled over every unit of every network, coloured by class; B the counts
    per class, one point per network, box over seeds, mean ± sd written above;
    C the intermediate class's sub-structure per network. `narrow`: the panels sit in
    1.875-in cells (the paper figure), so titles are short and A's per-class
    labels, which B repeats, are left off."""
    n = u["n"]

    def pA(ax=None):
        ax = panel_axes(ax)
        ip = u["index_pooled"]
        edges, counts = np.asarray(ip["edges"]), np.asarray(ip["counts"], float)
        centres = 0.5 * (edges[:-1] + edges[1:])
        per_net = counts / max(n, 1)                      # units per network per bin
        colors = [CLASS_COLORS["congruent"] if c > 0.5 else CLASS_COLORS["opposite"] if c < -0.5
                  else CLASS_COLORS["mixed"] for c in centres]
        ax.bar(centres, per_net, width=edges[1] - edges[0], color=colors, alpha=0.85, lw=0)
        for x in (-0.5, 0.5):
            ax.axvline(x, ls=":", lw=0.7, color="gray")
        ax.set_xlim(-1.02, 1.02)
        top = per_net.max() if per_net.size else 1
        ax.set_ylim(0, 1.35 * top)
        if not narrow:
            for c, x in (("opposite", -0.75), ("mixed", 0.0), ("congruent", 0.75)):
                ax.text(x, 1.22 * top, f"{u['counts'][c]['mean']:.1f} ± {u['counts'][c]['sd']:.1f}",
                        ha="center", va="center", fontsize=7, color=CLASS_COLORS[c])
        return finish_panel(ax, "congruency index", "units per network per bin",
                            "all units" if narrow else f"all units, {label}", None)

    def pB(ax=None):
        ax = panel_axes(ax)
        for i, c in enumerate(CLASSES):
            d = u["counts"][c]
            _strip(ax, i, d["values"], CLASS_COLORS[c], 3 + i)
            ax.text(i, 61, f"{d['mean']:.1f}\n± {d['sd']:.1f}", ha="center", va="top", fontsize=7,
                    color=CLASS_COLORS[c])
        ax.set_xticks(range(len(CLASSES)))
        ax.set_xticklabels([CLASS_LABEL[c] for c in CLASSES])
        ax.set_xlim(-0.7, len(CLASSES) - 0.3)
        ax.set_ylim(0, 64)
        share = u["share_opposite_gt_congruent"]
        return finish_panel(ax, "", "units of 64",
                            "classes per network" if narrow else
                            f"classes per network (opp > con: {100 * share:.0f}%)", None)

    def pC(ax=None):
        ax = panel_axes(ax)
        ms = u["mixed_substructure"]
        keys = (("both_cues", "both cues"), ("hand_only", "hand only"), ("vision_only", "vision only"))
        for i, (k, lab) in enumerate(keys):
            _strip(ax, i, ms[k]["values"], CLASS_COLORS["mixed"], 7 + i)
            ax.text(i, 61, f"{ms[k]['mean']:.1f}\n± {ms[k]['sd']:.1f}", ha="center", va="top", fontsize=7,
                    color=CLASS_COLORS["mixed"])
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([lab for _, lab in keys])
        ax.set_xlim(-0.7, len(keys) - 0.3)
        ax.set_ylim(0, 64)
        return finish_panel(ax, "", "intermediate units of 64",
                            "intermediate sub-structure" if narrow else "intermediate class sub-structure", None)

    return [pA, pB, pC]


def unit_figure(run, u, label):
    """The unit classes of the multisensory layer over the seeds (the three
    panels of `unit_class_panels`), as its own supplementary figure."""
    draws = unit_class_panels(u, label)
    figs = {f"{K_UNITS}/{K_UNITS}": manuscript_grid(draws, ncols=3)}
    for key, fn in zip(("A_index_pooled", "B_counts_by_class", "C_mixed_substructure"), draws):
        figs[f"{K_UNITS}/{key}"] = fn()
    save_figures({f"{PAPER_UNITS}/{PAPER_UNITS}": figs[f"{K_UNITS}/{K_UNITS}"]}, manuscript_dir(run),
                 formats=FORMATS, close=False)
    save_figures(figs, out_dir(run), formats=FORMATS)
    print(f"   -> {out_dir(run) / K_UNITS}/ and {manuscript_dir(run) / PAPER_UNITS}/ (+ flat copies)")
    units_paper_figure(run, u, label)


PAIR_LABELS = {"congruent+congruent": "con\n+con", "congruent+opposite": "con\n+opp",
               "congruent+mixed": "con\n+int", "opposite+opposite": "opp\n+opp",
               "opposite+mixed": "opp\n+int", "mixed+mixed": "int\n+int"}
STAT_STYLE = {"fusion": ("o", "#111111"), "segregation": ("^", "#8c8c8c")}


def _pair_color(g):
    a, b_ = g.split("+")
    return CLASS_COLORS[a] if a == b_ else "#555555"


def pairs_figure(run, pr, label):
    """Pairs of units clamped, over the seeds. A single-unit lesions: the
    class mean of the change in fusion and in segregation, one point per
    network (a size-free comparison of the classes). B, C every pair of units
    clamped, the mean change per class pair (fusion, segregation), one point
    per network. D additivity in the run's own network: each pair's effect
    against the sum of its two units' single effects (the read-out is linear
    in the layer, so the pair effect is the sum of the single effects up to
    the small nonlinearity of the weight statistics). E the same up to the
    class: the whole-class clamp against the sum of the class's single-unit
    effects, one point per network and class. F the class composition of the
    ten pairs that cost the most segregation, and the ten that cost the most
    fusion, pooled over networks."""
    n = pr["n"]
    rec = json.loads((out_dir(run) / K_SEEDS / "per_seed" / f"{run}.json").read_text())

    def strip(ax, pos, v, color, seed_):
        v = np.asarray(v, float)
        if len(v) > 1:
            ax.boxplot([v], positions=[pos], widths=0.6, showfliers=False,
                       medianprops=dict(color="k", lw=1.0), boxprops=dict(color=color, lw=1.0),
                       whiskerprops=dict(color=color, lw=0.8), capprops=dict(color=color, lw=0.8))
        jitter = np.random.default_rng(seed_).uniform(-0.18, 0.18, len(v)) if len(v) > 1 else [0.0]
        ax.plot(pos + jitter, v, "o", ms=2.2, color=color, alpha=0.6)

    def class_legend(ax, loc="lower left"):
        for c in CLASSES:
            ax.plot([], [], "o", ms=3, color=CLASS_COLORS[c], label=CLASS_LABEL[c])
        ax.legend(fontsize=7, loc=loc, ncol=3, handlelength=1.0, columnspacing=0.6, borderaxespad=0.2)

    def pA(ax=None):
        ax = panel_axes(ax)
        x0, ticks = 0, []
        for key in ("dF", "dS"):
            for i, c in enumerate(CLASSES):
                d = pr["single"][c].get("per_network_mean")
                if d:
                    strip(ax, x0 + i, d[key]["values"], CLASS_COLORS[c], 11 + i)
            ticks.append(x0 + 1)
            x0 += 4
        ax.axhline(0, lw=0.5, color="gray")
        ax.set_xticks(ticks)
        ax.set_xticklabels(["fusion", "segregation"])
        ax.set_xlim(-1, x0 - 1)
        lo = min(np.min(pr["single"][c]["per_network_mean"][k]["values"])
                 for c in CLASSES for k in ("dF", "dS") if pr["single"][c].get("per_network_mean"))
        ax.set_ylim(min(lo, -0.03) * 1.15, 0.012)
        fig = finish_panel(ax, "", "Δ per unit clamped (class mean)",
                           f"single-unit lesions ({label})", None)
        class_legend(ax)
        return fig

    def pair_panel(ax, key, ylabel, title):
        allv = []
        for i, g in enumerate(PAIR_GROUPS):
            d = pr["groups"].get(g)
            if d:
                strip(ax, i, d[key]["values"], _pair_color(g), 21 + i)
                allv.extend(d[key]["values"])
        ax.axhline(0, lw=0.5, color="gray")
        ax.set_xticks(range(len(PAIR_GROUPS)))
        ax.set_xticklabels([PAIR_LABELS[g] for g in PAIR_GROUPS], fontsize=7)
        ax.set_xlim(-0.7, len(PAIR_GROUPS) - 0.3)
        lo = min(allv) if allv else -0.06
        ax.set_ylim(min(lo, -0.06) * 1.1, 0.012)
        return finish_panel(ax, "", ylabel, title, None)

    def pB(ax=None):
        return pair_panel(panel_axes(ax), "dF_mean", "Δ fusion (mean over pairs)", f"pairs: fusion ({label})")

    def pC(ax=None):
        return pair_panel(panel_axes(ax), "dS_mean", "Δ segregation (mean over pairs)",
                          f"pairs: segregation ({label})")

    def pD(ax=None):
        ax = panel_axes(ax)
        p_ = rec["pairs"]
        U, V = np.asarray(p_["pairs"]["u"]), np.asarray(p_["pairs"]["v"])
        lo = 0.0
        for key, one, lab in (("dF", "dF", "fusion"), ("dS", "dS", "segregation")):
            single = np.asarray(p_["single"][one], float)
            pair = np.asarray(p_["pairs"][key], float)
            x = single[U] + single[V]
            mk, col = STAT_STYLE[lab]
            ax.plot(x, pair, mk, ms=1.8, alpha=0.35, color=col, mew=0, label=lab, rasterized=True)
            lo = min(lo, x.min(), pair.min())
        lo = 1.1 * lo
        ax.plot([lo, 0.02], [lo, 0.02], "--", lw=0.8, color="k", label="sum of singles")
        ax.set_xlim(lo, 0.02)
        ax.set_ylim(lo, 0.02)
        ax.set_aspect("equal")
        fig = finish_panel(ax, "sum of the two single-unit effects", "effect of the pair",
                           f"additivity ({'main network' if run == 'flagship' else run}, {len(U)} pairs)", None)
        ax.legend(fontsize=7, loc="upper left", markerscale=2.5, handlelength=1.4, borderaxespad=0.2)
        return fig

    def pE(ax=None):
        ax = panel_axes(ax)
        lo = 0.0
        for c in CLASSES:
            sv = pr["single"][c].get("sum_vs_whole")
            if not sv:
                continue
            for lab, xs, ys in (("fusion", sv["sum_dF"], sv["whole_dF"]), ("segregation", sv["sum_dS"], sv["whole_dS"])):
                mk, _ = STAT_STYLE[lab]
                ax.plot(xs, ys, mk, ms=2.6, alpha=0.6, color=CLASS_COLORS[c], mew=0)
                lo = min(lo, float(np.min(xs)), float(np.min(ys)))
        lo = 1.1 * lo
        ax.plot([lo, 0.05], [lo, 0.05], "--", lw=0.8, color="k")
        ax.set_xlim(lo, 0.05)
        ax.set_ylim(lo, 0.05)
        ax.set_aspect("equal")
        for lab in ("fusion", "segregation"):
            mk, _ = STAT_STYLE[lab]
            ax.plot([], [], mk, ms=3, color="#555555", mew=0, label=lab)
        fig = finish_panel(ax, "sum of the class's single-unit effects", "effect of the whole-class clamp",
                           f"additivity up to the class ({label})", "upper left")
        return fig

    def pF(ax=None):
        ax = panel_axes(ax)
        comp_s = pr.get("top10_segregation_loss_composition", {})
        comp_f = pr.get("top10_fusion_loss_composition", {})
        tot_s, tot_f = max(sum(comp_s.values()), 1), max(sum(comp_f.values()), 1)
        x = np.arange(len(PAIR_GROUPS))
        for off, comp, tot, lab in ((-0.2, comp_f, tot_f, "fusion"), (0.2, comp_s, tot_s, "segregation")):
            _, col = STAT_STYLE[lab]
            ax.bar(x + off, [100.0 * comp.get(g, 0) / tot for g in PAIR_GROUPS], width=0.38, color=col, lw=0, label=lab)
        for i, g in enumerate(PAIR_GROUPS):          # the class pair, as a coloured tick under each bar
            ax.plot([i - 0.38, i + 0.38], [-2.5, -2.5], "-", lw=2.5, color=_pair_color(g), solid_capstyle="butt",
                    clip_on=False)
        ax.set_xticks(x)
        ax.set_xticklabels([PAIR_LABELS[g] for g in PAIR_GROUPS], fontsize=7)
        ax.set_xlim(-0.7, len(PAIR_GROUPS) - 0.3)
        ax.set_ylim(0, 100)
        return finish_panel(ax, "", "share of the 10 worst pairs (%)",
                            f"most damaging pairs ({label})", "upper left")

    draws = [pA, pB, pC, pD, pE, pF]
    figs = {f"{K_PAIRS}/{K_PAIRS}": manuscript_grid(draws, ncols=3)}
    for key, fn in zip(("A_single_unit_by_class", "B_pairs_fusion", "C_pairs_segregation", "D_additivity_pairs",
                        "E_additivity_class", "F_worst_pairs_composition"), draws):
        figs[f"{K_PAIRS}/{key}"] = fn()
    # Fig. S2: the pair panels only; A and E are Fig. 10E and F of the paper
    paper = manuscript_grid([pB, pC, pD, pF], ncols=2)
    save_figures({f"{PAPER_PAIRS}/{PAPER_PAIRS}": paper}, manuscript_dir(run), formats=FORMATS)
    save_figures(figs, out_dir(run), formats=FORMATS)
    print(f"   -> {out_dir(run) / K_PAIRS}/ and {manuscript_dir(run) / PAPER_PAIRS}/ (+ flat copies)")


# --------------------------------------------------------------------------- #
# the whole-class clamp with the units set to 0 and to 1 instead of their mean
def _box_strip(ax, pos, v, color, seed_):
    """A box over networks with every network as a jittered point."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) > 1:
        ax.boxplot([v], positions=[pos], widths=0.6, showfliers=False,
                   medianprops=dict(color="k", lw=1.0), boxprops=dict(color=color, lw=1.0),
                   whiskerprops=dict(color=color, lw=0.8), capprops=dict(color=color, lw=0.8))
    jitter = np.random.default_rng(seed_).uniform(-0.18, 0.18, len(v)) if len(v) > 1 else [0.0]
    ax.plot(pos + jitter, v, "o", ms=2.2, color=color, alpha=0.6)
    return v


def _zbox_groups(ax, groups, title, ylabel="z vs random lesions", headroom=0.3):
    """Groups of three class boxes: groups = [(label, {class: z values})]."""
    x0, ticks, labels, allv = 0, [], [], []
    for lab, by_class in groups:
        for i, c in enumerate(CLASSES):
            if c in by_class:
                allv.extend(_box_strip(ax, x0 + i, by_class[c], CLASS_COLORS[c], 41 + i).tolist())
        ticks.append(x0 + 1)
        labels.append(lab)
        x0 += 4
    for y in (-2, 2):
        ax.axhline(y, ls=":", lw=0.7, color="gray")
    ax.axhline(0, lw=0.5, color="gray")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(-1, x0 - 1)
    lo = min(-2.6, (min(allv) if allv else -2.6) - 0.3)
    hi = max(2.6, (max(allv) if allv else 2.6)) + headroom
    ax.set_ylim(lo, hi)
    return finish_panel(ax, "", ylabel, title, None)


def si_lesion_figure(run, agg=None, label=None):
    """figS_lesion_seeds (Fig. S1): the lesion statistics of the 100 seeds that
    the paper's lesion figure (fig11_lesion_seeds) does not show, four z-score
    panels, two by two. A the read-out error (RMSE) after each whole-class
    lesion, with the read-out as trained and refitted, each against random
    lesions of the class's size; B fusion and segregation against the
    damage-matched null (random lesions of the same size whose RMSE is within
    5 % of the class lesion's); C the unit-by-unit removal read at an equal
    dose, the smallest class of each network, against random orders at that
    number of units; D the refitted read-out as z-scores against 20 random
    refits, for fusion, segregation and the two position slopes. One point per
    network, box over networks, dotted lines at +-2. The panels left out are
    Fig. 10A to C (the whole-class curves, the fusion and segregation z, the
    unit-by-unit curves) and Fig. 9H (the unit counts)."""
    if agg is None:
        agg = json.loads((out_dir(run) / K_SEEDS / "numbers.json").read_text())["aggregate"]
    n = agg["n"]
    label = label or (f"{n} networks" if n > 1 else "1 network")
    wc, rf = agg["whole_class"]["classes"], agg["refit"]

    def legend(ax):
        for c in CLASSES:
            ax.plot([], [], "o", ms=3, color=CLASS_COLORS[c], label=f"−{CLASS_LABEL[c]}")
        ax.legend(fontsize=7, loc="upper left", ncol=3, handlelength=1.0, columnspacing=0.5,
                  borderaxespad=0.2, handletextpad=0.2)

    def pA(ax=None):
        ax = panel_axes(ax)
        groups = [("as trained", {c: wc[c]["rmse"]["z"]["values"] for c in CLASSES if c in wc}),
                  ("refitted", {c: rf[c]["rmse"]["z"]["values"] for c in CLASSES if c in rf})]
        fig = _zbox_groups(ax, groups, f"read-out error ({label})", ylabel="z of RMSE vs random lesions",
                           headroom=1.6)
        legend(ax)
        return fig

    def pB(ax=None):
        ax = panel_axes(ax)
        dm = (agg.get("damage_matched") or {}).get("classes", {})
        groups = [(lab, {c: dm[c][s_]["z"]["values"] for c in CLASSES if c in dm})
                  for s_, lab in (("fusion", "fusion"), ("segregation", "segregation"))]
        return _zbox_groups(ax, groups, "damage-matched null", ylabel="z vs RMSE-matched lesions")

    def pC(ax=None):
        ax = panel_axes(ax)
        eq = (agg["cumulative"].get("equal_k") or {}).get("k_min_class") or {"classes": {}, "k": {}}
        k_ = eq.get("k") or {}
        groups = [(lab, {c: eq["classes"][c][s_]["z"]["values"] for c in CLASSES if c in eq["classes"]})
                  for s_, lab in (("fusion", "fusion"), ("segregation", "segregation"))]
        title = "equal dose" + (f" ({k_['mean']:.0f} ± {k_['sd']:.0f} units)" if k_ else "")
        return _zbox_groups(ax, groups, title, ylabel="z vs random orders")

    def pD(ax=None):
        ax = panel_axes(ax)
        groups = [(lab, {c: rf[c][s_]["z"]["values"] for c in CLASSES if c in rf})
                  for s_, lab in (("fusion", "fusion"), ("segregation", "segre-\ngation"),
                                  ("posreg_vis", "slope\nvis"), ("posreg_prop", "slope\nhand"))]
        return _zbox_groups(ax, groups, "read-out refitted, z", ylabel="z vs random refits")

    draws = [pA, pB, pC, pD]
    fig = manuscript_grid(draws, ncols=2)
    save_figures({f"{PAPER}/{PAPER}": fig}, manuscript_dir(run), formats=FORMATS)
    print(f"   -> {manuscript_dir(run) / PAPER}/ (+ flat copy)")
    return fig


def clamp_figure(run, agg, label, grid):
    """The whole-class clamp of lesion_seeds B repeated with the lesioned
    units set to 0 (A, B) and to 1 (D, E) instead of their mean activation,
    each against 100 random lesions of the class's size held at the same
    value; C and F put the three clamps side by side for fusion and for
    segregation."""
    keys = [("mean", "whole_class"), ("zero", CLAMP_KEY["zero"]), ("one", CLAMP_KEY["one"])]
    short = {"mean": "mean", "zero": "0", "one": "1"}

    def curves(ax, key, mode):
        wc = agg[key]
        x = np.asarray(grid, float)
        ax.fill_between(x, np.asarray(wc["intact"]["curve"]["mean"]) - np.asarray(wc["intact"]["curve"]["sd"]),
                        np.asarray(wc["intact"]["curve"]["mean"]) + np.asarray(wc["intact"]["curve"]["sd"]),
                        color="k", alpha=0.18, lw=0)
        ax.plot(x, wc["intact"]["curve"]["mean"], "-", lw=PANEL_LW + 0.4, color="k", label="intact")
        for c in CLASSES:
            if c in wc["classes"]:
                m, sd = (np.asarray(wc["classes"][c]["curve"][k_], float) for k_ in ("mean", "sd"))
                ax.fill_between(x, m - sd, m + sd, color=CLASS_COLORS[c], alpha=0.18, lw=0)
                ax.plot(x, m, "-", lw=PANEL_LW, color=CLASS_COLORS[c], label=f"−{CLASS_LABEL[c]}")
            if c in wc["random_curve"]:
                ax.plot(x, wc["random_curve"][c]["mean"], ":", lw=1.0, color=CLASS_COLORS[c])
        ax.plot([], [], ":", lw=1.0, color="gray", label="random, same k")
        lo = min(-0.1, np.nanmin([np.nanmin(wc["classes"][c]["curve"]["mean"]) for c in wc["classes"]]) - 0.1)
        hi = max(1.8, np.nanmax([np.nanmax(wc["classes"][c]["curve"]["mean"]) for c in wc["classes"]]) + 0.6)
        ax.set_ylim(lo, hi)
        fig = finish_panel(ax, "|disparity| (deg)", "weight on fused estimate",
                           f"units {CLAMPS[mode].replace(' (', ', ').rstrip(')')} ({label})", None)
        ax.legend(fontsize=7, loc="upper center", ncol=3, handlelength=1.2, columnspacing=0.6,
                  borderaxespad=0.2, handletextpad=0.4)
        return fig

    def zbox_one(ax, key, mode):
        wc = agg[key]
        groups = [(lab, {c: wc["classes"][c][s_]["z"]["values"] for c in CLASSES if c in wc["classes"]})
                  for s_, lab in zip(STATS, ("fusion", "segregation", "RMSE"))]
        fig = _zbox_groups(ax, groups, f"frozen read-out, units {CLAMPS[mode].split(' (')[0]}",
                           ylabel="z vs random lesions, same clamp", headroom=1.4)
        for c in CLASSES:
            ax.plot([], [], "o", ms=3, color=CLASS_COLORS[c], label=f"−{CLASS_LABEL[c]}")
        ax.legend(fontsize=7, loc="upper left", ncol=3, handlelength=1.0, columnspacing=0.6, borderaxespad=0.2)
        return fig

    def compare(ax, stat):
        groups = [(f"clamp: {short[mode]}", {c: agg[key]["classes"][c][stat]["z"]["values"]
                                             for c in CLASSES if c in agg[key]["classes"]})
                  for mode, key in keys if key in agg]
        return _zbox_groups(ax, groups, f"{stat} z by clamp value", ylabel="z vs random lesions, same clamp")

    def pA(ax=None):
        return curves(panel_axes(ax), CLAMP_KEY["zero"], "zero")

    def pB(ax=None):
        return zbox_one(panel_axes(ax), CLAMP_KEY["zero"], "zero")

    def pC(ax=None):
        return compare(panel_axes(ax), "fusion")

    def pD(ax=None):
        return curves(panel_axes(ax), CLAMP_KEY["one"], "one")

    def pE(ax=None):
        return zbox_one(panel_axes(ax), CLAMP_KEY["one"], "one")

    def pF(ax=None):
        return compare(panel_axes(ax), "segregation")

    draws = [pA, pB, pC, pD, pE, pF]
    figs = {f"{K_CLAMP}/{K_CLAMP}": manuscript_grid(draws, ncols=3)}
    for key, fn in zip(("A_curves_zero", "B_z_zero", "C_fusion_by_clamp", "D_curves_one", "E_z_one",
                        "F_segregation_by_clamp"), draws):
        figs[f"{K_CLAMP}/{key}"] = fn()
    save_figures({f"{PAPER_CLAMP}/{PAPER_CLAMP}": figs[f"{K_CLAMP}/{K_CLAMP}"]}, manuscript_dir(run),
                 formats=FORMATS, close=False)
    save_figures(figs, out_dir(run), formats=FORMATS)
    print(f"   -> {out_dir(run) / K_CLAMP}/ and {manuscript_dir(run) / PAPER_CLAMP}/ (+ flat copies)")


# --------------------------------------------------------------------------- #
# the tuning-curve patterns of the intermediate units over the seeds
SWEEP_COLORS = {"vis": COLORS["visual"], "prop": COLORS["prop"]}


def canonical_curves(rv, rp, positions, orient="dominant"):
    """Each unit's two curves on one footing: shifted by the unit's mean over
    both sweeps and scaled by its range over both sweeps (so the larger sweep
    spans about 1 and the relative amplitude of the two sweeps is kept), and
    the position axis of BOTH sweeps reversed when the orienting sweep falls
    with position, so that left- and right-preferring units of one shape are
    one pattern. orient="dominant": the sweep with the larger range (used for
    the clustering, where the weaker sweep may have no slope to orient by);
    orient="vis": the visual sweep (used to draw the congruent and opposite
    classes, where both sweeps have a slope and the opposite class is only
    legible with one sweep fixed to rise). rv, rp: units x positions."""
    rv, rp = np.asarray(rv, float), np.asarray(rp, float)
    x = np.concatenate([rv, rp], 1)
    x = x - x.mean(1, keepdims=True)
    x = x / np.maximum(x.max(1) - x.min(1), 1e-6)[:, None]
    n_pos = rv.shape[1]
    v, p_ = x[:, :n_pos].copy(), x[:, n_pos:].copy()
    pc = (positions - positions.mean()) / positions.std()
    if orient == "vis":
        slope = (v * pc).mean(1)
    else:
        dom_vis = (rv.max(1) - rv.min(1)) >= (rp.max(1) - rp.min(1))
        slope = np.where(dom_vis, (v * pc).mean(1), (p_ * pc).mean(1))
    flip = slope < 0
    v[flip], p_[flip] = v[flip, ::-1], p_[flip, ::-1]
    return v, p_, flip


def patterns(run, k=5, control="pcommon1", budget=None, k_range=range(2, 11)):
    """The most common tuning patterns of the intermediate units over the seeds: the
    canonical curves of every intermediate unit of every network (82 values: the two
    sweeps, see canonical_curves) are partitioned by k-means into k patterns
    (10 restarts, seed 0), which are reported by size: share of the intermediate
    units, mean congruency index, mean sweep ranges, the share of networks
    with at least one unit of the pattern and the units per network, and the
    pattern's mean ± sd curves. The silhouette of the partition for every k in
    k_range (on 2000 units, seed 0) is recorded so the choice of k can be
    judged: the variation is continuous, so the partition is descriptive.
    Writes mixed_patterns/numbers.json, labels.npz (the pattern of every unit)
    and the figure (figS_mixed_patterns in the manuscript folder)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    T = collect_tuning(run, control=control, budget=budget)
    if T is None:
        return None
    out = out_dir(run) / K_PATTERNS
    n_net, n_units, n_pos = T["resp_vis"].shape
    pos = np.asarray(T["positions"], float)
    rv, rp = T["resp_vis"].reshape(-1, n_pos), T["resp_prop"].reshape(-1, n_pos)
    v, p_, flip = canonical_curves(rv, rp, pos)
    X = np.concatenate([v, p_], 1)
    cl, idx = T["classes"].reshape(-1), np.asarray(T["index"], float).reshape(-1)
    net = np.repeat(np.arange(n_net), n_units)
    rng_v, rng_p = rv.max(1) - rv.min(1), rp.max(1) - rp.min(1)
    m = cl == "mixed"
    Xm = X[m]
    rs = np.random.default_rng(0)
    sub = rs.choice(len(Xm), min(2000, len(Xm)), replace=False)
    sil = {}
    for kk in k_range:
        km_ = KMeans(n_clusters=kk, n_init=10, random_state=0).fit(Xm)
        sil[str(kk)] = float(silhouette_score(Xm[sub], km_.labels_[sub])) if kk > 1 else None
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xm)
    lab = km.labels_
    order = np.argsort(-np.bincount(lab, minlength=k))
    rank = np.empty(k, int)
    rank[order] = np.arange(k)
    pattern = np.full(len(cl), -1)
    pattern[m] = rank[lab]                   # 0 = the most populated pattern
    numbers = {"run": run, "n_networks": int(n_net), "n_units_per_network": int(n_units), "k": int(k),
               "n_mixed_units": int(m.sum()), "flipped_share_mixed": float(flip[m].mean()),
               "silhouette_by_k": sil, "positions": pos.tolist(), "reference": {}, "patterns": []}

    def curves(sel, vv=v, pp=p_):
        return {"vis_mean": vv[sel].mean(0).tolist(), "vis_sd": vv[sel].std(0).tolist(),
                "prop_mean": pp[sel].mean(0).tolist(), "prop_sd": pp[sel].std(0).tolist()}

    v_ref, p_ref, _ = canonical_curves(rv, rp, pos, orient="vis")     # the reference classes, visual sweep rising
    for c in ("congruent", "opposite"):
        sel = cl == c
        numbers["reference"][c] = {"n": int(sel.sum()), "index_mean": float(idx[sel].mean()),
                                   "index_sd": float(idx[sel].std()), "curves": curves(sel, v_ref, p_ref)}
    for r in range(k):
        sel = pattern == r
        per_net = np.bincount(net[sel], minlength=n_net)
        numbers["patterns"].append({
            "rank": r + 1, "n": int(sel.sum()), "share_of_mixed": float(sel.sum() / m.sum()),
            "index_mean": float(idx[sel].mean()), "index_sd": float(idx[sel].std()),
            "range_vis_mean": float(rng_v[sel].mean()), "range_prop_mean": float(rng_p[sel].mean()),
            "vision_dominant_share": float((rng_v[sel] >= rng_p[sel]).mean()),
            "one_cue_only_share": float((np.minimum(rng_v[sel], rng_p[sel]) < WEAK).mean()),
            "networks_with_pattern": int((per_net > 0).sum()), "units_per_network": per_net.tolist(),
            "units_per_network_mean": float(per_net.mean()), "units_per_network_sd": float(per_net.std(ddof=1)),
            "curves": curves(sel)})
    out.mkdir(parents=True, exist_ok=True)
    save_json(numbers, out / "numbers.json")
    np.savez(out / "labels.npz", network=T["networks"][net], unit=np.tile(np.arange(n_units), n_net),
             pattern=pattern, index=idx, classes=cl, flipped=flip)
    patterns_figure(run, numbers)
    units_paper_figure(run, numbers=numbers)    # the paper's unit figure uses these patterns
    return numbers


def patterns_figure(run, numbers=None):
    """A, B the congruent and opposite classes (mean ± sd canonical curves,
    for reference); C.. the intermediate patterns by size; the last panel the units
    of each pattern per network."""
    out = out_dir(run) / K_PATTERNS
    numbers = numbers or json.loads((out / "numbers.json").read_text())
    pos = np.asarray(numbers["positions"], float)
    pats = numbers["patterns"]
    n_net, label = numbers["n_networks"], f"{numbers['n_networks']} seeds"

    def curve_panel(ax, cv, title, note, legend=False):
        for key, lab in (("vis", "vision sweep"), ("prop", "hand sweep")):
            mu, sd = np.asarray(cv[f"{key}_mean"]), np.asarray(cv[f"{key}_sd"])
            ax.fill_between(pos, mu - sd, mu + sd, color=SWEEP_COLORS[key], alpha=0.18, lw=0)
            ax.plot(pos, mu, "-", lw=PANEL_LW, color=SWEEP_COLORS[key], label=lab)
        ax.axhline(0, lw=0.5, color="gray")
        ax.axvline(0, lw=0.5, color="gray")
        ax.set_xlim(pos[0], pos[-1])
        ax.set_ylim(-0.85, 1.15)
        ax.set_yticks([-0.5, 0, 0.5])
        ax.text(0.03, 0.97, note, transform=ax.transAxes, ha="left", va="top", fontsize=7)
        fig = finish_panel(ax, "position (deg)", "response (range-normalised)", title, None)
        if legend:
            ax.legend(fontsize=7, loc="lower right", handlelength=1.2, borderaxespad=0.2)
        return fig

    draws, keys = [], []
    for c in ("congruent", "opposite"):
        ref = numbers["reference"][c]

        def p_ref(ax=None, c=c, ref=ref):
            ax = panel_axes(ax)
            return curve_panel(ax, ref["curves"], f"{CLASS_LABEL[c]} units ({label})",
                               f"n = {ref['n']}, index {ref['index_mean']:+.2f} ± {ref['index_sd']:.2f}", legend=(c == "congruent"))
        draws.append(p_ref)
        keys.append(f"{c}_reference")
    for pt in pats:

        def p_pat(ax=None, pt=pt):
            ax = panel_axes(ax)
            return curve_panel(ax, pt["curves"], f"intermediate pattern {pt['rank']} ({100 * pt['share_of_mixed']:.0f}%)",
                               f"n = {pt['n']}, index {pt['index_mean']:+.2f} ± {pt['index_sd']:.2f}\n"
                               f"in {pt['networks_with_pattern']}/{n_net} networks, "
                               f"{pt['units_per_network_mean']:.1f} ± {pt['units_per_network_sd']:.1f} each")
        draws.append(p_pat)
        keys.append(f"mixed_pattern_{pt['rank']}")

    def p_counts(ax=None):
        ax = panel_axes(ax)
        for i, pt in enumerate(pats):
            vals = np.asarray(pt["units_per_network"], float)
            if len(vals) > 1:
                ax.boxplot([vals], positions=[i], widths=0.6, showfliers=False,
                           medianprops=dict(color="k", lw=1.0), boxprops=dict(color=CLASS_COLORS["mixed"], lw=1.0),
                           whiskerprops=dict(color=CLASS_COLORS["mixed"], lw=0.8),
                           capprops=dict(color=CLASS_COLORS["mixed"], lw=0.8))
            jitter = np.random.default_rng(31 + i).uniform(-0.18, 0.18, len(vals)) if len(vals) > 1 else [0.0]
            ax.plot(i + jitter, vals, "o", ms=2.2, color=CLASS_COLORS["mixed"], alpha=0.6)
        ax.set_xticks(range(len(pats)))
        ax.set_xticklabels([str(pt["rank"]) for pt in pats])
        ax.set_xlim(-0.7, len(pats) - 0.3)
        ax.set_ylim(0, max(1.15 * max(max(pt["units_per_network"]) for pt in pats), 5))
        return finish_panel(ax, "intermediate pattern", "units per network", f"patterns per network ({label})", None)
    draws.append(p_counts)
    keys.append("patterns_per_network")
    letters = "ABCDEFGHIJKLMNOP"[:len(draws)]
    figs = {f"{K_PATTERNS}/{K_PATTERNS}": manuscript_grid(draws, ncols=4, letters=letters)}
    for letter, key, fn in zip(letters, keys, draws):
        figs[f"{K_PATTERNS}/{letter}_{key}"] = fn()
    save_figures({f"{PAPER_PATTERNS}/{PAPER_PATTERNS}": figs[f"{K_PATTERNS}/{K_PATTERNS}"]}, manuscript_dir(run),
                 formats=FORMATS, close=False)
    save_figures(figs, out_dir(run), formats=FORMATS)
    print(f"   -> {out}/ and {manuscript_dir(run) / PAPER_PATTERNS}/ (+ flat copies)")


# --------------------------------------------------------------------------- #
# The manuscript's unit figure (18 Oct 2026): the tuning patterns of the 100
# seeds in one row of small panels (A, B the congruent and opposite classes,
# C-F the four largest intermediate patterns) over the unit classes of the same seeds
# (G-I, the three panels of `unit_class_panels`; J, opposite units of the
# causal networks against their always-fuse twins). It replaces fig10_units, the
# one-network figure (three example units; counts per network, per prior and
# against the twin), whose files are in results/_to_delete/.
# --------------------------------------------------------------------------- #
TOP_ROW = {"height": 1.9, "gap": 0.2, "xlabel": 0.42, "title": 0.33}   # inches
TOP_FONT = {"tick": 7, "label": 7, "title": 7.5, "note": 6.2, "legend": 6.2}
N_MIXED_SHOWN = 4                      # intermediate patterns in the top row, by size


def _twin_panel(run):
    """Opposite units of each causal network against its always-fuse twin,
    from <run>/.../units/numbers.json (written by scripts/neuronal/run.py
    units: `runs` and `twins`); dotted, the always-fuse control. None when
    that file is missing."""
    f = out_dir(run) / "units" / "numbers.json"
    if not f.exists():
        print(f"   (no {f}: twin panel skipped)")
        return None
    d = json.loads(f.read_text())
    rows, twins = d.get("runs", []), d.get("twins", [])
    pairs = [(r, tw) for r in rows for tw in twins if np.isclose(r["p_common"], tw["p_common"])]
    if not pairs:
        return None
    ctrl1 = [r for r in rows if r["p_common"] == 1.0]

    def pJ(ax=None):
        ax = panel_axes(ax)
        x = np.arange(len(pairs))
        wdt = 0.36
        ax.bar(x - wdt / 2, [r["n_opposite"] for r, _ in pairs], wdt, color=CLASS_COLORS["opposite"],
               label="causal network")
        ax.bar(x + wdt / 2, [tw["n_opposite"] for _, tw in pairs], wdt, color=CLASS_COLORS["opposite"],
               alpha=0.35, label="always-fuse twin")
        if ctrl1:
            ax.axhline(ctrl1[0]["n_opposite"], ls=":", lw=0.8, color="gray")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r['p_common']:g}" for r, _ in pairs])
        ax.set_ylim(0, 40)
        return finish_panel(ax, "common-cause prior", "opposite units of 64",
                            "causal vs twin", "upper right")
    return pJ


def units_paper_figure(run, u=None, label=None, numbers=None):
    """fig10_units_seeds: six small tuning-pattern panels (7.5 in across) over
    the three unit-class cells and, when units/numbers.json exists, the
    causal-vs-twin opposite-unit counts (four cells of 1.875 x 2.5 in),
    7.5 x 4.4 in in all.
    `u` is the `units` aggregate of lesion_seeds/numbers.json; `numbers` the
    mixed_patterns/numbers.json of the same run."""
    from cmsi.viz.manuscript import CELL_SQUARE, PANEL_MARGIN, cell_axes
    from cmsi.viz.style import exact_frame, label_panels
    from matplotlib import pyplot as plt
    if u is None:
        u = json.loads((out_dir(run) / K_SEEDS / "numbers.json").read_text())["aggregate"]["units"]
    label = label or f"{u['n']} seeds"
    pat_file = out_dir(run) / K_PATTERNS / "numbers.json"
    if numbers is None:
        if not pat_file.exists():
            print(f"   (no {pat_file}: run `seeds.py patterns` first; fig10_units_seeds not drawn)")
            return None
        numbers = json.loads(pat_file.read_text())
    pos = np.asarray(numbers["positions"], float)
    pats = numbers["patterns"][:N_MIXED_SHOWN]

    W, H = CELL_SQUARE[0] * 3, CELL_SQUARE[1] + TOP_ROW["height"]
    fig = plt.figure(figsize=(W, H))

    # ---- top row: six small curve panels, same left/right margins as the cells
    m = PANEL_MARGIN
    n_top = 2 + len(pats)
    avail = W - m["left"] - m["right"]
    pw = (avail - (n_top - 1) * TOP_ROW["gap"]) / n_top
    bottom = CELL_SQUARE[1] + TOP_ROW["xlabel"]
    ph = TOP_ROW["height"] - TOP_ROW["xlabel"] - TOP_ROW["title"]
    tops = []
    for i in range(n_top):
        left = m["left"] + i * (pw + TOP_ROW["gap"])
        tops.append(fig.add_axes([left / W, bottom / H, pw / W, ph / H]))

    def small_curves(ax, cv, title, note, legend=False, ylabel=False):
        for key, lab in (("vis", "vision"), ("prop", "hand")):
            mu, sd = np.asarray(cv[f"{key}_mean"]), np.asarray(cv[f"{key}_sd"])
            ax.fill_between(pos, mu - sd, mu + sd, color=SWEEP_COLORS[key], alpha=0.18, lw=0)
            ax.plot(pos, mu, "-", lw=1.0, color=SWEEP_COLORS[key], label=lab)
        ax.axhline(0, lw=0.5, color="gray")
        ax.axvline(0, lw=0.5, color="gray")
        ax.set_xlim(pos[0], pos[-1])
        ax.set_ylim(-0.85, 1.2)
        ax.set_xticks([-20, 0, 20])
        ax.set_yticks([-0.5, 0, 0.5])
        ax.tick_params(labelsize=TOP_FONT["tick"], length=2, pad=1.5, labelleft=ylabel)
        ax.text(0.04, 0.97, note, transform=ax.transAxes, ha="left", va="top", fontsize=TOP_FONT["note"],
                linespacing=1.15)
        ax.set_title(title, fontsize=TOP_FONT["title"], pad=3)
        ax.set_xlabel("position (deg)", fontsize=TOP_FONT["label"], labelpad=1.5)
        if ylabel:
            ax.set_ylabel("response (normalised)", fontsize=TOP_FONT["label"], labelpad=2)
        if legend:
            ax.legend(fontsize=TOP_FONT["legend"], loc="lower left", handlelength=1.0,
                      borderaxespad=0.2, labelspacing=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    for i, c in enumerate(("congruent", "opposite")):
        ref = numbers["reference"][c]
        small_curves(tops[i], ref["curves"], c,
                     f"n = {ref['n']}\nindex {ref['index_mean']:+.2f} ± {ref['index_sd']:.2f}",
                     legend=(c == "congruent"), ylabel=(i == 0))
    for i, pt in enumerate(pats):
        small_curves(tops[2 + i], pt["curves"],
                     f"intermediate {pt['rank']} ({100 * pt['share_of_mixed']:.0f}%)",
                     f"n = {pt['n']}\nindex {pt['index_mean']:+.2f} ± {pt['index_sd']:.2f}")

    # ---- bottom row: the three unit-class cells of the same seeds, and the
    # opposite-unit counts of the causal networks against their always-fuse
    # twins (from the one-network analysis, units/numbers.json of the run)
    twin_fn = _twin_panel(run)
    draws = unit_class_panels(u, label, narrow=twin_fn is not None)
    if twin_fn is not None:
        draws.append(twin_fn)
    cell = (W / len(draws), CELL_SQUARE[1])             # four cells across when the twin panel exists
    bottoms = []
    for i, fn in enumerate(draws):
        ax = cell_axes(fig, cell, origin=(i * cell[0], 0))
        fn(ax)
        ax.spines[["top", "right"]].set_visible(False)
        if len(draws) > 3:                              # narrow cells: slant the category labels
            for lab in ax.get_xticklabels():
                if not lab.get_text().replace(".", "").isdigit():
                    lab.set_rotation(25)
                    lab.set_ha("right")
        bottoms.append(ax)

    label_panels(tops, letters="ABCDEF", dx=-14, dy=9)
    label_panels(bottoms, letters="GHIJ", dx=-34, dy=2)
    exact_frame(fig)
    save_figures({f"{PAPER_UNITS_FIG}/{PAPER_UNITS_FIG}": fig}, manuscript_dir(run), formats=FORMATS,
                 close=False)
    save_figures({f"{K_UNITS_PAPER}/{K_UNITS_PAPER}": fig}, out_dir(run), formats=FORMATS)
    print(f"   -> {out_dir(run) / K_UNITS_PAPER}/ and {manuscript_dir(run) / PAPER_UNITS_FIG}/ (+ flat copies)")
    return fig


def lesion_paper_figure(run, agg=None, numbers=None, band="sd"):
    """fig11_lesion_seeds: the lesion result of the 100 seeds in seven panels,
    7.5 x 5 in: three standard cells on top (A-C) and four cells of
    1.875 x 2.5 in below (D-G, as the bottom row of Fig. 9). A the whole-class
    clamp with the read-out as trained, implied weight against |d|, mean
    over networks (band: +-1 sd across networks), each class against random lesions of its
    own size (dotted); B the within-network z of fusion and segregation
    against those size-matched random lesions, one point per network; C the
    cumulative removal of one class, most typical first, against random
    orders; D the read-out refitted after each whole-class lesion: fusion and
    segregation per network, against the intact network refitted (dashed) and
    random lesions of the class's size refitted (grey ticks); E the decision
    read-out (decision_weights.py): per network, the mean weight of each class
    towards 'segregated' in a logistic read-out of the network's own
    fused/segregated label; F the single-unit lesions, the class mean change
    per unit removed; G additivity: the whole-class effect against the sum of
    the class's single-unit effects. (18 Oct: A-F with D the refit; 19 Oct
    morning: D the decision read-out, refit in the SI; 19 Oct: both, A-G.) Needs numbers.json with the pairs design
    (`analyse --extras pairs`). `band` sets the band of A and C: "sd" (the
    paper's choice, 2026-10-18), the spread across networks, or "sem", the
    standard error of the mean over networks (ten times narrower at n = 100)."""
    out = out_dir(run) / K_SEEDS
    if agg is None:
        numbers = numbers or json.loads((out / "numbers.json").read_text())
        agg = numbers["aggregate"]
    grid = np.asarray(numbers["grid"] if numbers else json.loads((out / "numbers.json").read_text())["grid"], float)
    n = agg["n"]
    label = f"{n} networks"
    pr = agg.get("pairs")
    if not pr:
        print("   (no pairs design in numbers.json: run `seeds.py analyse --extras pairs`; fig11_lesion_seeds not drawn)")
        return None

    def sem_band(ax, x, m, sd, count, color, lw=PANEL_LW, label_=None, ls="-"):
        m, sd, count = np.asarray(m, float), np.asarray(sd, float), np.asarray(count, float)
        half = sd if band == "sd" else sd / np.sqrt(np.maximum(count, 1))
        ax.fill_between(x, m - half, m + half, color=color, alpha=0.13 if band == "sd" else 0.22, lw=0)
        ax.plot(x, m, ls, lw=lw, color=color, label=label_)

    def pA(ax=None):
        ax = panel_axes(ax)
        wc = agg["whole_class"]
        sem_band(ax, grid, wc["intact"]["curve"]["mean"], wc["intact"]["curve"]["sd"], n, "k", lw=PANEL_LW + 0.4,
                 label_="intact")
        for c in CLASSES:
            if c in wc["classes"]:
                e = wc["classes"][c]
                sem_band(ax, grid, e["curve"]["mean"], e["curve"]["sd"], e["n"], CLASS_COLORS[c],
                         label_=f"−{CLASS_LABEL[c]} ({e['k']['mean']:.0f})")
                ax.plot(grid, wc["random_curve"][c]["mean"], ":", lw=1.0, color=CLASS_COLORS[c])
        ax.plot([], [], ":", lw=1.0, color="gray", label="random, same size")
        ax.set_ylim(-0.1, 2.0)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        fig = finish_panel(ax, "|disparity| (deg)", "weight on fused estimate",
                           f"whole class clamped ({label})", None)
        ax.legend(fontsize=7, loc="upper center", ncol=2, handlelength=1.2, columnspacing=0.8,
                  borderaxespad=0.2, handletextpad=0.4, labelspacing=0.3)
        return fig

    def pB(ax=None):
        ax = panel_axes(ax)
        wc = agg["whole_class"]["classes"]
        groups = [(lab, {c: wc[c][s_]["z"]["values"] for c in CLASSES if c in wc})
                  for s_, lab in (("fusion", "fusion"), ("segregation", "segregation"))]
        fig = _zbox_groups(ax, groups, f"class vs random lesions ({label})",
                           ylabel="z (class vs size-matched random)", headroom=1.3)
        for c in CLASSES:
            ax.plot([], [], "o", ms=3, color=CLASS_COLORS[c], label=f"−{CLASS_LABEL[c]}")
        ax.legend(fontsize=6.5, loc="upper left", ncol=3, handlelength=0.6, columnspacing=0.4,
                  handletextpad=0.15, borderaxespad=0.2)
        return fig

    def pC(ax=None):
        ax = panel_axes(ax)
        cu = agg["cumulative"]
        for c in CLASSES:
            if c in cu:
                cs = cu[c]["curves"]["segregation"]
                m, sd, cnt = (np.asarray(cs[key], float) for key in ("mean", "sd", "n_at_k"))
                keep = cnt >= min(MIN_NETWORKS, cnt.max())
                k = np.arange(1, len(m) + 1)
                sem_band(ax, k[keep], m[keep], sd[keep], cnt[keep], CLASS_COLORS[c], label_=f"{CLASS_LABEL[c]} first")
        rnd = cu["random"]["segregation"]
        rm = np.asarray(rnd["mean"], float)
        cnt = np.asarray(rnd.get("n_at_k", np.full(len(rm), n)), float)
        keep = cnt >= min(MIN_NETWORKS, cnt.max())
        ax.plot(np.arange(1, len(rm) + 1)[keep], rm[keep], ":", lw=1.0, color="gray", label="random order")
        ax.set_ylim(-0.05, 1.05)
        return finish_panel(ax, "units removed (most typical first)", "segregation at |d| ≥ 15°",
                            f"removed unit by unit ({label})", "lower left")

    def pD(ax=None, narrow=False):
        # the read-out refitted after the class lesion (agg["refit"]): does the
        # information survive the loss of a class?
        ax = panel_axes(ax)
        rf = agg["refit"]
        x0, ticks, allv = 0, [], []
        for s_ in ("fusion", "segregation"):
            intact = rf["intact"][s_]["mean"]
            ax.plot([x0 - 0.7, x0 + 2.7], [intact, intact], "--", lw=0.9, color="k", zorder=1)
            for i, c in enumerate(CLASSES):
                if c not in rf:
                    continue
                allv.extend(_box_strip(ax, x0 + i, rf[c][s_]["value"]["values"], CLASS_COLORS[c], 51 + i).tolist())
                nm = rf[c][s_]["null_mean"]["mean"]
                ax.plot([x0 + i - 0.3, x0 + i + 0.3], [nm, nm], "-", lw=1.4, color="#8c8c8c", zorder=4)
            ticks.append(x0 + 1)
            x0 += 4
        ax.plot([], [], "--", lw=0.9, color="k", label="intact, refitted")
        ax.plot([], [], "-", lw=1.4, color="#8c8c8c", label="random, refitted")
        ax.set_xticks(ticks)
        ax.set_xticklabels(["fusion", "segregation"])
        ax.set_xlim(-1, x0 - 1)
        lo = min(allv) if allv else 0.7
        ax.set_ylim(min(0.7, lo - 0.02), 1.06)
        fig = finish_panel(ax, "", "value after the class lesion",
                           "read-out refitted" if narrow else f"read-out refitted ({label})", None)
        ax.legend(fontsize=6.5 if narrow else 7, loc="lower left", ncol=1, handlelength=1.4,
                  borderaxespad=0.2, labelspacing=0.3)
        return fig

    def pE(ax=None, narrow=False):
        # the decision read-out (scripts/neuronal/decision_weights.py): a
        # logistic read-out of the network's own fused/segregated label from
        # the z-scored MSL; per network, the mean weight of each class
        # towards 'segregated', one point per network, box over networks
        ax = panel_axes(ax)
        files = sorted((out_dir(run) / "decision_weights" / "per_network").glob("*.json"))
        recs = [json.loads(f.read_text()) for f in files]
        if not recs:
            ax.text(0.5, 0.5, "run decision_weights.py --runs ...", ha="center", transform=ax.transAxes)
            return finish_panel(ax, "", "", "decision read-out", None)
        allv = []
        for i, c in enumerate(CLASSES):
            v = np.array([-r["summary"]["w_fused_all"][c]["mean"] for r in recs], float)
            allv.extend(_box_strip(ax, i, v, CLASS_COLORS[c], 61 + i).tolist())
        ax.axhline(0, lw=0.5, color="gray")
        ax.set_xticks(range(len(CLASSES)))
        ax.set_xticklabels([CLASS_LABEL[c] for c in CLASSES], fontsize=7.5)
        ax.set_xlim(-0.7, len(CLASSES) - 0.3)
        lim = 1.15 * max(abs(min(allv)), abs(max(allv)))
        ax.set_ylim(-lim, lim)
        acc = np.mean([r["acc_all"] for r in recs])
        ax.text(0.03, 0.04, f"decoding accuracy {acc:.2f}", transform=ax.transAxes, fontsize=7, va="bottom")
        return finish_panel(ax, "", "weight towards 'segregated'",
                            "decision read-out" if narrow else f"decision read-out ({len(recs)} networks)", None)

    def pF(ax=None, narrow=False):
        ax = panel_axes(ax)
        x0, ticks, allv = 0, [], []
        for key in ("dF", "dS"):
            for i, c in enumerate(CLASSES):
                d = pr["single"][c].get("per_network_mean")
                if d:
                    allv.extend(_box_strip(ax, x0 + i, d[key]["values"], CLASS_COLORS[c], 11 + i).tolist())
            ticks.append(x0 + 1)
            x0 += 4
        ax.axhline(0, lw=0.5, color="gray")
        ax.set_xticks(ticks)
        ax.set_xticklabels(["fusion", "segregation"])
        ax.set_xlim(-1, x0 - 1)
        lo = min(allv) if allv else -0.03
        ax.set_ylim(min(lo, -0.03) * 1.15, 0.012)
        return finish_panel(ax, "", "change per unit removed",
                            "single units clamped" if narrow else f"single units clamped ({label})", None)

    def pG(ax=None, narrow=False):
        ax = panel_axes(ax)
        lo = 0.0
        for c in CLASSES:
            sv = pr["single"][c].get("sum_vs_whole")
            if not sv:
                continue
            for lab, xs, ys in (("fusion", sv["sum_dF"], sv["whole_dF"]), ("segregation", sv["sum_dS"], sv["whole_dS"])):
                mk, _ = STAT_STYLE[lab]
                ax.plot(xs, ys, mk, ms=2.6, alpha=0.6, color=CLASS_COLORS[c], mew=0)
                lo = min(lo, float(np.min(xs)), float(np.min(ys)))
        lo = 1.1 * lo
        ax.plot([lo, 0.05], [lo, 0.05], "--", lw=0.8, color="k")
        ax.set_xlim(lo, 0.05)
        ax.set_ylim(lo, 0.05)
        # narrow cell: keep the cell's frame and stretch the data limits instead
        ax.set_aspect("equal", adjustable="datalim" if narrow else "box")
        for lab in ("fusion", "segregation"):
            mk, _ = STAT_STYLE[lab]
            ax.plot([], [], mk, ms=3, color="#555555", mew=0, label=lab)
        return finish_panel(ax, "sum of single-unit effects", "whole-class effect",
                            "additivity" if narrow else f"additivity ({label})",
                            "lower right" if narrow else "upper left")

    # three standard cells on top, four narrow cells below (as Fig. 9)
    from cmsi.viz.manuscript import CELL_SQUARE, cell_axes, letter
    from cmsi.viz.style import exact_frame
    from matplotlib import pyplot as plt
    top, bottom = [pA, pB, pC], [pD, pE, pF, pG]
    W, ch = CELL_SQUARE[0] * 3, CELL_SQUARE[1]
    fig = plt.figure(figsize=(W, 2 * ch))
    axes = []
    for i, fn in enumerate(top):
        ax = cell_axes(fig, CELL_SQUARE, origin=(i * CELL_SQUARE[0], ch))
        fn(ax)
        axes.append(ax)
    narrow_cell = (W / len(bottom), ch)
    for i, fn in enumerate(bottom):
        ax = cell_axes(fig, narrow_cell, origin=(i * narrow_cell[0], 0))
        fn(ax, narrow=True)
        if fn is pE:                                  # class names: slant them in the narrow cell
            for lab in ax.get_xticklabels():
                lab.set_rotation(25)
                lab.set_ha("right")
        axes.append(ax)
    letter(axes, "ABCDEFG")
    exact_frame(fig)
    figs = {f"{K_LESION_PAPER}/{K_LESION_PAPER}": fig}
    for key, fn in zip(("A_whole_class_curves", "B_z_vs_random", "C_cumulative", "D_refit",
                        "E_decision_readout", "F_per_unit", "G_additivity"), top + bottom):
        figs[f"{K_LESION_PAPER}/{key}"] = fn()
    save_figures({f"{PAPER_LESION_FIG}/{PAPER_LESION_FIG}": fig}, manuscript_dir(run), formats=FORMATS,
                 close=False)
    save_figures(figs, out_dir(run), formats=FORMATS)
    print(f"   -> {out_dir(run) / K_LESION_PAPER}/ and {manuscript_dir(run) / PAPER_LESION_FIG}/ (+ flat copies)")
    return fig


# --------------------------------------------------------------------------- #
def main(argv=None):
    import argparse
    import warnings
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["train", "analyse", "figure", "all", "patterns", "units", "lesion",
                                            "si"])
    ap.add_argument("--run", default="flagship", help="the configuration's run (seed 0)")
    ap.add_argument("--n", type=int, default=100, help="networks in all, the run included (train: seeds 1..n-1)")
    ap.add_argument("--runs", nargs="*", default=None, help="analyse these runs instead of the run's seeds")
    ap.add_argument("--control", default="pcommon1", help="the run whose residuals are sigma_out")
    ap.add_argument("--extras", nargs="*", default=[], choices=["damage", "pairs", "clamp"],
                    help="optional designs: damage (RMSE-matched null), pairs (all unit pairs), "
                         "clamp (the whole-class clamp with the units set to 0 and to 1 instead of their mean)")
    ap.add_argument("--budget", type=float, default=None,
                    help="seconds of computation per call; the rest is left for the next call")
    ap.add_argument("--k", type=int, default=5, help="patterns: number of k-means patterns of the intermediate units")
    ap.add_argument("--band", choices=["sem", "sd"], default="sd",
                    help="lesion: the band of panels A and C, the SEM over networks or the sd across networks")
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if args.command == "train":
        train(args.run, args.n)
        return
    apply_style()
    if args.command == "patterns":
        patterns(args.run, k=args.k, control=args.control, budget=args.budget)
        return
    if args.command == "units":                 # the paper's unit figure from the stored numbers
        units_paper_figure(args.run)
        return
    if args.command == "lesion":                # the paper's lesion figure from the stored numbers
        lesion_paper_figure(args.run, band=args.band)
        return
    if args.command == "si":                    # Figs. S1 and S2 (lesions) from the stored numbers
        agg = json.loads((out_dir(args.run) / K_SEEDS / "numbers.json").read_text())["aggregate"]
        label = f"{agg['n']} networks"
        si_lesion_figure(args.run, agg, label)
        if agg.get("pairs"):
            pairs_figure(args.run, agg["pairs"], label)
        return
    if args.command in ("analyse", "all"):
        records, agg = analyse(args.run, runs=args.runs, n=None if args.runs else args.n, control=args.control,
                               extras=tuple(args.extras), budget=args.budget)
        figure(args.run, agg, json.loads((out_dir(args.run) / K_SEEDS / "numbers.json").read_text()))
    else:
        figure(args.run)


if __name__ == "__main__":
    main()
