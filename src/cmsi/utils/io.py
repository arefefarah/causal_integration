"""Reading and writing the artefacts that move between pipeline stages.

Stage boundaries are files, so any script can be re-run on its own:

    01_generate_data  ->  data/<name>.npz
    02_train          ->  results/<run>/model.pt
    03_analyze        ->  results/<run>/metrics.json
    04_figures        ->  results/<run>/figures/**
"""

import json
from pathlib import Path

import numpy as np

_META_CONFIG = "__config__"
_META_TARGETS = "__target_names__"
_ENC_PREFIX = "enc__"


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #
def save_dataset(d, cfg, path):
    """Write a dataset dict (arrays + encoders + config) to a compressed .npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in d.items() if isinstance(v, np.ndarray)}
    arrays.update({_ENC_PREFIX + k: v for k, v in d["encoders"].items()})
    arrays[_META_CONFIG] = np.array(json.dumps(cfg))
    arrays[_META_TARGETS] = np.array(d["target_names"])
    np.savez_compressed(path, **arrays)
    return path


def load_dataset(path):
    """Read a dataset back. Returns (dataset dict, config dict)."""
    z = np.load(path, allow_pickle=False)
    cfg = json.loads(str(z[_META_CONFIG]))
    d = {"encoders": {}, "target_names": [str(t) for t in z[_META_TARGETS]]}
    for key in z.files:
        if key.startswith("__"):
            continue
        if key.startswith(_ENC_PREFIX):
            d["encoders"][key[len(_ENC_PREFIX):]] = z[key]
        else:
            d[key] = z[key]
    return d, cfg


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def save_checkpoint(model, cfg, history, splits, path):
    """Save weights together with the config and the exact train/val/test split.

    Storing the splits is what lets a later analysis script evaluate on the same
    held-out trials the model was actually validated against.
    """
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": cfg,
        "input_dim": int(model.x_mean.numel()),
        "history": history,
        "splits": {k: np.asarray(v) for k, v in splits.items()},
    }, path)
    return path


def load_checkpoint(path):
    """Rebuild a trained model. Returns (model, config, history, splits)."""
    import torch

    from cmsi.models.network import Net

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = Net(ckpt["input_dim"], ckpt["config"]["model"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["config"], ckpt["history"], ckpt["splits"]


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def save_json(obj, path):
    """Write metrics as json, converting numpy types on the way out."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_encode))
    return path


def load_json(path):
    return json.loads(Path(path).read_text())


def _encode(obj):
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"cannot serialise {type(obj)}")
