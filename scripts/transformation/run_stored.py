"""The transformation -> causal inference statistics on the STORED networks:
the flagship and its 99 seeds (results/flagship, results/flagship_s1..s99; same
dataset, same encoding).

    python scripts/transformation/run_stored.py                 # flagship + all seeds
    python scripts/transformation/run_stored.py --seeds 0 1 2   # a subset (0 = the flagship)

Writes results/transformation_investigation/stored/
    numbers_flagship.json        every block of transformation.analyse on the flagship
    seeds.json                   one flat row per network + mean / sd / sem over networks
    curves_flagship.npz          the per-trial arrays behind the flagship's figures
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from inv_common import OUT, ROOT, load_run, save_json  # noqa: E402

from transformation import analyse, flatten  # noqa: E402

STORED = OUT / "stored"


def aggregate(rows):
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float)) and not isinstance(rows[0][k], bool)]
    agg = {}
    for k in keys:
        v = np.array([r.get(k) for r in rows if r.get(k) is not None], float)
        v = v[np.isfinite(v)]
        agg[k] = {"mean": float(v.mean()) if v.size else None, "sd": float(v.std(ddof=1)) if v.size > 1 else None,
                  "sem": float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else None, "n": int(v.size)}
    return agg


def main(a):
    STORED.mkdir(parents=True, exist_ok=True)
    seeds = a.seeds if a.seeds is not None else list(range(100))
    rows = []
    for s in seeds:
        name = "flagship" if s == 0 else f"flagship_s{s}"
        t0 = time.time()
        b = load_run(ROOT / "results" / name)
        full = s == 0
        res = analyse(b, do_eps=full or a.eps_all, do_beta=full or a.eps_all, do_units=True, do_eyerel=full, do_di=full or a.di_all)
        row = flatten(res)
        row["seed"] = s
        row["name"] = name
        rows.append(row)
        if full:
            save_json({k: v for k, v in res.items() if k != "_arrays"}, STORED / "numbers_flagship.json")
            np.savez_compressed(STORED / "curves_flagship.npz", **res["_arrays"])
        print(f"  {name}: acc net {row['R0.accuracy_net']:.3f} R0 {row['R0.accuracy_R0']:.3f} R1 {row['R1.accuracy_R1']:.3f} | "
              f"frame index {row['frame.decision_frame_index_net']:.2f} | eye gain {row['partials.eye_gain_net_median']:.3f} | "
              f"leak {row['partials.leak_d_muprop_dxeye_segregated_median']:+.3f} | opp {row['units.n_opposite']} | {time.time() - t0:.1f} s", flush=True)
        save_json({"rows": rows, "aggregated": aggregate(rows)}, STORED / "seeds.json")
    print(f"wrote {STORED}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--eps-all", action="store_true", help="run the eps/beta manipulations on every seed (slower)")
    p.add_argument("--di-all", action="store_true", help="run the displacement index on every seed")
    main(p.parse_args())
