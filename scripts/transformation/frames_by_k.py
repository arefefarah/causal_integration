"""Channel G: which frame the MSL units hold, against the frame the OBSERVER
transforms into.

A unit tuned to x_vis + k x_eye shifts its preferred spatial position with the
eye by 1 - k (displacement index DI: 0 = body frame, 1 = retinal), so the
expected DI of a unit in the observer's frame is 1 - k, not 0. For every causal
level network and its always-fuse twin this script takes the per-unit DI of
transformation.units_block (alignment over five eye positions; usable units =
modulation > 0.1 and alignment R2 > 0.8) and reports the share of usable units
within 0.2 of 1 - k ("Bayes-frame share"), the retinal share (DI >= 0.8) and the
medians by class.

    python scripts/transformation/frames_by_k.py

Reads  results/transformation_investigation/levels/{rows.json, numbers/<run>.json}
       results/transformation_investigation/runs/<run>_twin/
Writes results/transformation_investigation/levels/frames_by_k.json  (one record per network)
"""

import collections
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from inv_common import OUT, RUNS, load_run, save_json  # noqa: E402

from transformation import units_block  # noqa: E402

LEVELS = OUT / "levels"


def _arr(v):
    return np.array([np.nan if x is None else x for x in v], float)


def main():
    rows = json.loads((LEVELS / "rows.json").read_text())
    out = []
    for r in sorted(rows, key=lambda r: (r["param"], r["level"], r["seed"])):
        num = json.loads((LEVELS / "numbers" / f"{r['name']}.json").read_text())
        k = r["basic.k"]
        di, r2 = _arr(num["units"]["di"]), _arr(num["units"]["di_r2"])
        cls = np.array(num["units"]["classes"])
        use = np.isfinite(di) & (r2 > 0.8)
        exp = 1 - k
        rec = {"name": r["name"], "param": r["param"], "level": r["level"], "seed": r["seed"], "k": k,
               "expected_di": exp, "n_usable": int(use.sum()),
               "bayes_frame_share": float(np.mean(np.abs(di[use] - exp) < 0.2)) if use.sum() else np.nan,
               "retinal_share": float(np.mean(di[use] >= 0.8)) if use.sum() else np.nan,
               "di_median": float(np.median(di[use])) if use.sum() else np.nan}
        for c in ("congruent", "opposite", "mixed"):
            m = use & (cls == c)
            rec[f"bayes_frame_share_{c}"] = float(np.mean(np.abs(di[m] - exp) < 0.2)) if m.sum() else np.nan
            rec[f"di_median_{c}"] = float(np.median(di[m])) if m.sum() else np.nan
            rec[f"n_usable_{c}"] = int(m.sum())
        tw = RUNS / (r["name"] + "_twin")
        if tw.exists():
            u = units_block(load_run(tw), di=True)
            dit, r2t = _arr(u["di"]), _arr(u["di_r2"])
            uset = np.isfinite(dit) & (r2t > 0.8)
            rec.update({"twin_n_usable": int(uset.sum()),
                        "twin_bayes_frame_share": float(np.mean(np.abs(dit[uset] - exp) < 0.2)) if uset.sum() else np.nan,
                        "twin_retinal_share": float(np.mean(dit[uset] >= 0.8)) if uset.sum() else np.nan,
                        "twin_di_median": float(np.median(dit[uset])) if uset.sum() else np.nan})
        out.append(rec)
        print(f"  {r['name']}: k {k:.3f} expected DI {exp:.2f} | DI median {rec['di_median']:+.2f} Bayes share {rec['bayes_frame_share']:.2f} "
              f"| opposite {rec['di_median_opposite']:+.2f} | twin {rec.get('twin_di_median', float('nan')):+.2f}", flush=True)
    save_json(out, LEVELS / "frames_by_k.json")
    groups = collections.defaultdict(list)
    for rec in out:
        groups[(rec["param"], rec["level"])].append(rec)
    print("\nparam level | k | expected DI | causal: DI median, Bayes share, retinal share | opposite DI, Bayes share | congruent DI | twin: DI median, Bayes share, retinal share")
    for key in sorted(groups):
        v = groups[key]

        def m(k_):
            return np.nanmean([x.get(k_, np.nan) for x in v])

        print(f"{key[0]:9s} {key[1]:6g} | {m('k'):.3f} | {m('expected_di'):.2f} | {m('di_median'):+.2f}, {m('bayes_frame_share'):.2f}, {m('retinal_share'):.2f} "
              f"| {m('di_median_opposite'):+.2f}, {m('bayes_frame_share_opposite'):.2f} | {m('di_median_congruent'):+.2f} "
              f"| {m('twin_di_median'):+.2f}, {m('twin_bayes_frame_share'):.2f}, {m('twin_retinal_share'):.2f}")
    print(f"wrote {LEVELS / 'frames_by_k.json'}")


if __name__ == "__main__":
    main()
