"""The transformation -> causal inference statistics on the SAVED level networks
(results/transformation_investigation/runs, trained by train_levels.py).

    python scripts/transformation/run_levels.py                  # every causal run in the manifest
    python scripts/transformation/run_levels.py --param eye_noise

Writes results/transformation_investigation/levels/
    rows.json           one flat row per causal network (+ its twin's unit classes)
    aggregated.json     mean / sd / sem per (param, level) over seeds
    numbers/<name>.json every block of transformation.analyse per network
    curves.npz          per-trial arrays of seed 0 of every level (for the figures)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from inv_common import OUT, RUNS, load_run, save_json  # noqa: E402

from transformation import analyse, flatten, units_block  # noqa: E402

LEVELS = OUT / "levels"


def aggregate(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["param"], r["level"]), []).append(r)
    out = []
    for (param, level), g in sorted(groups.items()):
        keys = [k for k in g[0] if isinstance(g[0][k], (int, float)) and not isinstance(g[0][k], bool)]
        agg = {"param": param, "level": level, "n_seeds": len(g), "seeds": sorted(r["seed"] for r in g)}
        for k in keys:
            v = np.array([r.get(k) for r in g if r.get(k) is not None], float)
            v = v[np.isfinite(v)]
            agg[k] = float(v.mean()) if v.size else None
            agg[k + "_sd"] = float(v.std(ddof=1)) if v.size > 1 else 0.0
            agg[k + "_sem"] = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0
        out.append(agg)
    return out


def main(a):
    LEVELS.mkdir(parents=True, exist_ok=True)
    (LEVELS / "numbers").mkdir(exist_ok=True)
    manifest = json.loads((RUNS / "manifest.json").read_text())
    causal = [m for m in manifest if not m["twin"] and (a.param is None or m["param"] == a.param)]
    twins = {m["name"]: m for m in manifest if m["twin"]}
    rows_path = LEVELS / "rows.json"
    rows = json.loads(rows_path.read_text()) if rows_path.exists() and a.resume else []
    done = {r["name"] for r in rows}
    curves = dict(np.load(LEVELS / "curves.npz")) if (LEVELS / "curves.npz").exists() and a.resume else {}
    for m in sorted(causal, key=lambda r: (r["param"], r["level"], r["seed"])):
        if m["name"] in done:
            continue
        t0 = time.time()
        b = load_run(RUNS / m["name"])
        res = analyse(b, do_eps=True, do_beta=True, do_units=True, do_eyerel=True, do_di=True)
        row = flatten(res)
        row.update({"name": m["name"], "param": m["param"], "level": m["level"], "seed": m["seed"],
                    "best_epoch": m["best_epoch"], "rejection_rate": m["rejection_rate"], "ecc_c": m.get("ecc_c", 0.0)})
        tw = twins.get(m["name"] + "_twin")
        if tw is not None:
            bt = load_run(RUNS / tw["name"])
            u = units_block(bt, di=True)
            row.update({"twin.n_congruent": u["n_congruent"], "twin.n_opposite": u["n_opposite"], "twin.n_mixed": u["n_mixed"],
                        "twin.di_median_usable": u["di_median_usable"], "twin.di_share_body": u["di_share_body"],
                        "twin.di_share_retinal": u["di_share_retinal"], "twin.best_epoch": tw["best_epoch"]})
            row["units.opposite_excess_over_twin"] = row["units.n_opposite"] - u["n_opposite"]
            res["twin_units"] = {k: v for k, v in u.items() if k not in ("index", "classes", "di", "di_r2", "di_amplitude")}
        rows.append(row)
        save_json({k: v for k, v in res.items() if k != "_arrays"}, LEVELS / "numbers" / f"{m['name']}.json")
        if m["seed"] == 0:
            for k, v in res["_arrays"].items():
                curves[f"{m['param']}_{m['level']:g}_{k}"] = v
            np.savez_compressed(LEVELS / "curves.npz", **curves)
        save_json(rows, rows_path)
        save_json(aggregate(rows), LEVELS / "aggregated.json")
        print(f"  {m['name']}: k {row['basic.k']:.3f} var_vis_body {row['basic.var_vis_body']:.2f} | acc net {row['R0.accuracy_net']:.3f} "
              f"R0 {row['R0.accuracy_R0']:.3f} R1 {row['R1.accuracy_R1']:.3f} | cross net/R0/R1 {row['R0.crossing_net']:.2f}/{row['R0.crossing_R0']:.2f}/{row['R1.crossing_R1']:.2f} "
              f"| gain {row['partials.eye_gain_net_median']:.3f} (k {row['partials.eye_gain_k_observer']:.3f}) | frame {row['frame.decision_frame_index_net']:.2f} "
              f"| opp {row['units.n_opposite']} twin {row.get('twin.n_opposite')} | {time.time() - t0:.0f} s", flush=True)
    print(f"wrote {LEVELS}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--param", default=None)
    p.add_argument("--resume", action="store_true", help="keep rows already computed")
    main(p.parse_args())
