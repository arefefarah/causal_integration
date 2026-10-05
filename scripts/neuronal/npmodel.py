"""Torch-free access to a trained checkpoint, for analyses run on a machine
without torch.

`load_checkpoint_np(path)` reads the zip that `torch.save` wrote and rebuilds
every tensor as a numpy array; `NpNet` is the forward pass of `models.Net`
in numpy, with the same standardiser, sigmoid hidden layers, linear read-out
and softplus on the variance columns. `predict`, `hidden` and `lesion_predict`
mirror `models.predict`, `models.hidden_activations` and `analysis.lesion`.
Validated against results/<run>/analysis.npz: the numpy predictions on the
stored test split match the torch predictions to float32 rounding.

The `cmsi` package pulls torch in through its __init__; `stub_cmsi()` registers
empty package objects for the levels whose files we import directly, so
style, manuscript, encoding, generative, causal, decoding and behavior load
without it.
"""

import io
import pickle
import sys
import types
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

_DTYPES = {"FloatStorage": np.float32, "DoubleStorage": np.float64,
           "HalfStorage": np.float16, "LongStorage": np.int64,
           "IntStorage": np.int32, "ShortStorage": np.int16,
           "ByteStorage": np.uint8, "CharStorage": np.int8,
           "BoolStorage": np.bool_}


def stub_cmsi():
    """Make `cmsi.*` importable file by file, without cmsi/__init__ (torch)."""
    for name, rel in (("cmsi", "cmsi"), ("cmsi.viz", "cmsi/viz"),
                      ("cmsi.utils", "cmsi/utils"), ("cmsi.analysis", "cmsi/analysis"),
                      ("cmsi.data", "cmsi/data")):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [str(SRC / rel)]
            sys.modules[name] = m
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))


def _rebuild_tensor_v2(storage, offset, size, stride, *rest):
    arr, dtype = storage
    item = np.dtype(dtype).itemsize
    view = np.lib.stride_tricks.as_strided(
        arr[offset:], shape=tuple(size), strides=tuple(s * item for s in stride))
    return np.array(view, dtype=dtype)      # an owned copy, C-contiguous


class _Unpickler(pickle.Unpickler):
    def __init__(self, data, zf, root):
        super().__init__(io.BytesIO(data))
        self.zf, self.root = zf, root

    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if module == "torch" and name in _DTYPES:
            return name
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        return super().find_class(module, name)

    def persistent_load(self, pid):
        typename, storage_type, key, location, numel = pid
        assert typename == "storage", pid
        dtype = _DTYPES[storage_type]
        raw = self.zf.read(f"{self.root}/data/{key}")
        return (np.frombuffer(raw, dtype=dtype), dtype)


def load_checkpoint_np(path):
    """-> dict with 'state_dict' (name -> ndarray), 'config', 'input_dim',
    'history', 'splits' (name -> index array)."""
    zf = zipfile.ZipFile(path)
    root = zf.namelist()[0].split("/")[0]
    return _Unpickler(zf.read(f"{root}/data.pkl"), zf, root).load()


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softplus(x):
    return np.logaddexp(0.0, x)


class NpNet:
    """models.Net, forward only, in float32 numpy."""

    VAR_COLUMNS = {"causal": [1, 3], "fused": [1]}

    def __init__(self, ckpt):
        sd, cfg = ckpt["state_dict"], ckpt["config"]["model"]
        if cfg["activation"] != "sigmoid":
            raise NotImplementedError(cfg["activation"])
        self.head = cfg["head"]
        self.var_cols = self.VAR_COLUMNS[self.head]
        self.x_mean = sd["x_mean"].astype(np.float32)
        self.x_std = sd["x_std"].astype(np.float32)
        self.layers = []
        i = 0
        while f"layers.{i}.0.weight" in sd:
            self.layers.append((sd[f"layers.{i}.0.weight"].astype(np.float32),
                                sd[f"layers.{i}.0.bias"].astype(np.float32)))
            i += 1
        self.W_out = sd["readout.weight"].astype(np.float32)
        self.b_out = sd["readout.bias"].astype(np.float32)
        self.n_hidden = [w.shape[0] for w, _ in self.layers]

    def hidden(self, X):
        """-> list of hidden activations, one (n, units) array per layer."""
        h = (np.asarray(X, np.float32) - self.x_mean) / self.x_std
        hs = []
        for W, b in self.layers:
            h = _sigmoid(h @ W.T + b)
            hs.append(h)
        return hs

    def readout(self, h_last):
        out = h_last @ self.W_out.T + self.b_out
        for c in self.var_cols:
            out[:, c] = _softplus(out[:, c])
        return out

    def predict(self, X):
        return self.readout(self.hidden(X)[-1])

    def lesion_predict(self, X, keep, clamp_to):
        """Predictions with last-layer units where keep==False clamped to
        `clamp_to` (per-unit values): analysis.lesion with mode='mean'."""
        h = self.hidden(X)[-1].copy()
        keep = np.asarray(keep, bool)
        h[:, ~keep] = np.asarray(clamp_to, np.float32)[~keep]
        return self.readout(h), h


def load_run(run, dataset=None):
    """(net, ckpt, cfg, d_test, d_full) for results/<run>: the checkpoint, its
    config, the dataset it was trained on restricted to the stored test split,
    and the full dataset (for its encoders)."""
    stub_cmsi()
    from cmsi.utils.io import load_dataset
    out = ROOT / "results" / run
    ckpt = load_checkpoint_np(out / "model.pt")
    cfg = ckpt["config"]
    name = dataset or (out / "dataset.txt").read_text().strip()
    d_full, _ = load_dataset(ROOT / "data" / f"{name}.npz")
    idx = np.asarray(ckpt["splits"]["test"])
    d_test = {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(d_full["X"]) else v)
              for k, v in d_full.items()}
    return NpNet(ckpt), ckpt, cfg, d_test, d_full


if __name__ == "__main__":
    # self-test against the stored predictions of every run that has them
    import sys as _sys
    runs = _sys.argv[1:] or ["flagship"]
    for run in runs:
        net, ckpt, cfg, d, d_full = load_run(run)
        pred = net.predict(d["X"])
        saved = np.load(ROOT / "results" / run / "analysis.npz")
        err = np.abs(pred - saved["pred"]).max()
        same = np.allclose(d["disparity"], saved["disparity"])
        print(f"{run}: hidden {net.n_hidden}, n_test {len(pred)}, "
              f"max|pred - stored| = {err:.2e}, split matches: {same}")
