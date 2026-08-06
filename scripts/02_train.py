"""Stage 2 -- train the network on a generated dataset.

    python scripts/02_train.py --data main --run baseline
    python scripts/02_train.py --data twin --run twin --epochs 200

Writes results/<run>/model.pt (weights + config + train/val/test split) and
copies the config in as results/<run>/config.yaml, so a run always records what
produced it.
"""

import argparse

import _bootstrap  # noqa: F401
from cmsi.models import train
from cmsi.utils import (
    dataset_path,
    load_dataset,
    run_dir,
    save_checkpoint,
    save_config,
    tweak,
)


def main(args):
    d, cfg = load_dataset(dataset_path(args.data))
    overrides = {k: v for k, v in
                 {"epochs": args.epochs, "lr": args.lr, "seed": args.seed}.items()
                 if v is not None}
    if overrides:
        cfg = tweak(cfg, **overrides)

    out = run_dir(args.run)
    print(f"training on data/{args.data} -> results/{args.run}")
    model, history, splits = train(d, cfg)

    save_checkpoint(model, cfg, history, splits, out / "model.pt")
    save_config(cfg, out / "config.yaml")
    # remember which dataset this run used, so stage 3 can find it again
    (out / "dataset.txt").write_text(args.data)
    print(f"wrote {out / 'model.pt'}")

    if args.figures:
        from cmsi.viz import apply_style, save_figures
        from cmsi.viz import training as viz_training
        apply_style()
        save_figures(viz_training.all_figures(history), out / "figures" / "training")
        print(f"wrote {out / 'figures' / 'training'}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="main", help="dataset name in data/")
    p.add_argument("--run", default="baseline", help="run name -> results/<run>/")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-figures", dest="figures", action="store_false",
                   help="skip the training-diagnostic figures")
    main(p.parse_args())
