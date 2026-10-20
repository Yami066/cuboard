"""Unit tests for DiffLogicNet implementation."""
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from models import DiffLogicNet, LogicGateLayer, apply_all_16_ops  # noqa: E402


def test_16_ops_bounds():
    a = torch.tensor([0.0, 0.0, 1.0, 1.0])
    b = torch.tensor([0.0, 1.0, 0.0, 1.0])
    ops = apply_all_16_ops(a, b)
    assert ops.shape == (4, 16)
    # Check AND (index 1)
    assert torch.equal(ops[:, 1], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    # Check OR (index 7)
    assert torch.equal(ops[:, 7], torch.tensor([0.0, 1.0, 1.0, 1.0]))
    # Check XOR (index 6)
    assert torch.equal(ops[:, 6], torch.tensor([0.0, 1.0, 1.0, 0.0]))


def test_forward_pass_and_discretization():
    model = DiffLogicNet(in_features=784, num_classes=10, num_layers=2, neurons_per_layer=100)
    x = torch.rand(4, 784)
    out_cont = model(x)
    assert out_cont.shape == (4, 10)

    model.discretize()
    out_disc = model(x)
    assert out_disc.shape == (4, 10)
