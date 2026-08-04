"""The network's contract: shapes, positive variances, and a loss that trains."""

import numpy as np
import pytest
import torch

from cmsi.models import Net, hidden_activations, mse_loss, output_weights, predict, train


@pytest.fixture
def net(cfg):
    return Net(input_dim=130, model_cfg=cfg["model"])


def test_output_shape_matches_the_head(cfg):
    x = torch.randn(16, 130)
    assert Net(130, dict(cfg["model"], head="causal"))(x).shape == (16, 4)
    assert Net(130, dict(cfg["model"], head="fused"))(x).shape == (16, 2)


def test_variance_outputs_are_positive(net):
    """Softplus on the variance columns -- a negative variance is meaningless
    and would poison every downstream comparison."""
    out = net(torch.randn(256, 130) * 50)
    assert torch.all(out[:, net.var_cols] > 0)


def test_hidden_activations_expose_both_layers(net):
    acts = hidden_activations(net, np.random.randn(32, 130))
    assert acts["sil"].shape == (32, 64) and acts["msl"].shape == (32, 64)
    assert np.array_equal(acts["sil"], acts["layer0"])
    assert np.array_equal(acts["msl"], acts["layer1"])


def test_standardizer_normalises_the_training_inputs(net):
    X = np.random.randn(500, 130) * 30 + 100
    net.fit_standardizer(X)
    z = (torch.as_tensor(X, dtype=torch.float32) - net.x_mean) / net.x_std
    assert torch.allclose(z.mean(0), torch.zeros(130), atol=1e-4)
    assert torch.allclose(z.std(0), torch.ones(130), atol=1e-2)


def test_predict_matches_a_direct_forward_pass(net):
    X = np.random.randn(70, 130)
    with torch.no_grad():
        direct = net(torch.as_tensor(X, dtype=torch.float32)).numpy()
    assert np.allclose(predict(net, X, batch_size=16), direct, atol=1e-6)


def test_balancing_equalises_outputs_of_different_scale():
    """Without it the large-scale column dominates the gradient, which is how
    mu_prop ends up unlearned while var_prop trains fine."""
    Y = torch.stack([torch.randn(1000) * 100, torch.randn(1000) * 0.1], dim=1)
    pred = torch.zeros_like(Y)

    _, unbalanced = mse_loss(pred, Y, None)
    _, balanced = mse_loss(pred, Y, output_weights(Y))
    assert unbalanced[0] > 100 * unbalanced[1]
    assert balanced.max() / balanced.min() < 1.5


def test_training_reduces_the_loss_and_respects_the_split(dataset, cfg):
    model, history, splits = train(dataset, cfg, verbose=False)
    assert history["val"][-1] <= history["val"][0]
    assert history["best_epoch"] >= 0
    assert set(splits) == {"train", "val", "test"}
    assert len(np.intersect1d(splits["train"], splits["test"])) == 0
