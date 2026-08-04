"""Training loop: mini-batch, early stopping on validation loss.

    model, history, splits = train(d, cfg)

The splits are returned (and saved in the checkpoint) so every later analysis
runs on the same held-out trials the model was validated against.
"""

import numpy as np
import torch

from cmsi.data.dataset import split_indices
from cmsi.models.losses import mse_loss, output_weights
from cmsi.models.network import Net


def make_optimizer(model, train_cfg):
    name = train_cfg["optimizer"]
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])
    if name == "rprop":
        return torch.optim.Rprop(model.parameters(), lr=train_cfg["lr"])
    raise ValueError(f"unknown optimizer {name!r}")


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(d, cfg, seed=None, splits=None, verbose=True):
    """Train one model. Returns (model, history, splits)."""
    train_cfg = cfg["training"]
    seed = cfg["seed"] if seed is None else seed
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = get_device()

    X = torch.as_tensor(d["X"], dtype=torch.float32, device=device)
    Y = torch.as_tensor(d["Y"], dtype=torch.float32, device=device)
    if splits is None:
        splits = split_indices(len(X), train_cfg["split"], rng)
    train_idx = torch.as_tensor(np.asarray(splits["train"]), device=device)
    val_idx = torch.as_tensor(np.asarray(splits["val"]), device=device)

    model = Net(X.shape[1], cfg["model"]).to(device)
    if train_cfg["standardize_inputs"]:
        model.fit_standardizer(d["X"][splits["train"]])
        model.to(device)

    weights = output_weights(Y[train_idx], train_cfg["balance_loss"])
    optimizer = make_optimizer(model, train_cfg)

    best = {"val": np.inf, "epoch": -1, "state": None}
    history = {"train": [], "val": [], "val_per_output": []}

    for epoch in range(train_cfg["epochs"]):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), device=device)]
        batch_losses = []
        for i in range(0, len(order), train_cfg["batch_size"]):
            batch = order[i:i + train_cfg["batch_size"]]
            optimizer.zero_grad()
            loss, _ = mse_loss(model(X[batch]), Y[batch], weights)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach()))

        model.eval()
        with torch.no_grad():
            val_total, val_per_output = mse_loss(model(X[val_idx]), Y[val_idx], weights)
        val_total = float(val_total)
        history["train"].append(float(np.mean(batch_losses)))
        history["val"].append(val_total)
        history["val_per_output"].append(val_per_output.cpu().numpy().tolist())

        if val_total < best["val"] - 1e-9:
            best = {"val": val_total, "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
        elif epoch - best["epoch"] >= train_cfg["patience"]:
            if verbose:
                print(f"early stop at epoch {epoch} (best was {best['epoch']})")
            break

        if verbose and epoch % 10 == 0:
            print(f"  epoch {epoch:4d}   train {history['train'][-1]:.4f}"
                  f"   val {val_total:.4f}")

    model.load_state_dict(best["state"])
    history.update(best_val=best["val"], best_epoch=best["epoch"],
                   target_names=d["target_names"])
    if verbose:
        print(f"best val {best['val']:.4f} at epoch {best['epoch']}")
    return model, history, splits
