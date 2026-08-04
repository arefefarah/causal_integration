"""The feedforward network.

    input -> hidden 0 ("SIL", sensory integration)
          -> hidden 1 ("MSL", multisensory)
          -> linear readout

Heads:
    "causal"  4 outputs [mu_vis, var_vis, mu_prop, var_prop]  (Kording Eqs. 9/10)
    "fused"   2 outputs [mu_fused, var_fused] -- the always-fuse control twin,
              never trained on anything causal-inference-specific

Softplus keeps the variance columns positive; the means stay linear. Input
standardisation lives inside the model as buffers, so a loaded checkpoint is
self-contained: hand it raw X and it does the right thing.
"""

import numpy as np
import torch
from torch import nn

ACTIVATIONS = {"sigmoid": nn.Sigmoid, "relu": nn.ReLU, "tanh": nn.Tanh}
N_OUTPUTS = {"causal": 4, "fused": 2}
VAR_COLUMNS = {"causal": [1, 3], "fused": [1]}


class Net(nn.Module):
    """`model_cfg` is cfg["model"]: hidden sizes, activation, head type."""

    def __init__(self, input_dim, model_cfg):
        super().__init__()
        self.head = model_cfg["head"]
        self.var_cols = VAR_COLUMNS[self.head]
        activation = ACTIVATIONS[model_cfg["activation"]]

        self.layers = nn.ModuleList()
        prev = input_dim
        for size in model_cfg["hidden"]:
            self.layers.append(nn.Sequential(nn.Linear(prev, size), activation()))
            prev = size
        self.readout = nn.Linear(prev, N_OUTPUTS[self.head])

        self.register_buffer("x_mean", torch.zeros(input_dim))
        self.register_buffer("x_std", torch.ones(input_dim))

    def fit_standardizer(self, X_train):
        """Store the z-scoring statistics of the training inputs."""
        x = torch.as_tensor(np.asarray(X_train), dtype=torch.float32)
        self.x_mean = x.mean(0)
        self.x_std = x.std(0).clamp_min(1e-6)

    def forward(self, x, return_hidden=False):
        h = (x - self.x_mean) / self.x_std
        hiddens = []
        for layer in self.layers:
            h = layer(h)
            hiddens.append(h)
        out = self.readout(h)
        columns = list(out.unbind(dim=1))
        for c in self.var_cols:
            columns[c] = nn.functional.softplus(columns[c])
        out = torch.stack(columns, dim=1)
        return (out, hiddens) if return_hidden else out


@torch.no_grad()
def predict(model, X, batch_size=4096):
    """Network outputs for raw (unstandardised) X, as a numpy array."""
    model.eval()
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.as_tensor(np.asarray(X[i:i + batch_size]),
                             dtype=torch.float32, device=device)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def hidden_activations(model, X, batch_size=4096):
    """Activations as {"layer0": ..., "layer1": ..., "sil": ..., "msl": ...}.

    "sil" is the first hidden layer and "msl" the last -- the two the layer-wise
    decoding analysis compares.
    """
    model.eval()
    device = next(model.parameters()).device
    chunks = None
    for i in range(0, len(X), batch_size):
        xb = torch.as_tensor(np.asarray(X[i:i + batch_size]),
                             dtype=torch.float32, device=device)
        _, hs = model(xb, return_hidden=True)
        hs = [h.cpu().numpy() for h in hs]
        chunks = hs if chunks is None else [np.concatenate([a, b])
                                            for a, b in zip(chunks, hs)]
    acts = {f"layer{i}": h for i, h in enumerate(chunks)}
    acts["sil"], acts["msl"] = chunks[0], chunks[-1]
    return acts
