"""Stage 1 -- sample trials, run the analytical observer, encode the inputs.

    python scripts/01_generate_data.py --name main
    python scripts/01_generate_data.py --name small --n 8000
    python scripts/01_generate_data.py --name twin --head fused

Writes data/<name>.npz, which carries the config that produced it. Everything
downstream reads that config back, so a dataset and its parameters can never
drift apart.
"""

import argparse

import _bootstrap  # noqa: F401

from cmsi.data import make_dataset
from cmsi.utils import dataset_path, load_config, save_dataset, tweak


def main(args):
    cfg = load_config(args.config)
    overrides = {k: v for k, v in
                 {"n_trials": args.n, "seed": args.seed, "head": args.head}.items()
                 if v is not None}
    if overrides:
        cfg = tweak(cfg, **overrides)

    print(f"generating {cfg['training']['n_trials']} trials "
          f"(head={cfg['model']['head']}, seed={cfg['seed']})")
    d = make_dataset(cfg)

    path = save_dataset(d, cfg, dataset_path(args.name))
    print(f"X {d['X'].shape}   Y {d['Y'].shape}   targets {d['target_names']}")
    print(f"wrote {path}")
    return path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="main", help="dataset name -> data/<name>.npz")
    p.add_argument("--config", default=None, help="yaml config (default: configs/default.yaml)")
    p.add_argument("--n", type=int, default=None, help="number of trials")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--head", choices=["causal", "fused"], default=None,
                   help="which targets to store")
    main(p.parse_args())
