"""Differentiable and Discretized Logic Gate Network implementation (Petersen et al., 2022)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


# Table 1: Real-valued relaxation of the 16 binary logic operators
def apply_all_16_ops(a, b):
    """
    Computes all 16 relaxed logic operations for inputs a, b in [0, 1].
    Returns tensor of shape [batch, neurons, 16].
    """
    ab = a * b
    ops = [
        torch.zeros_like(a),               # 0: False
        ab,                                # 1: A and B
        a - ab,                            # 2: not(A implies B)
        a,                                 # 3: A
        b - ab,                            # 4: not(B implies A)
        b,                                 # 5: B
        a + b - 2 * ab,                    # 6: A xor B
        a + b - ab,                        # 7: A or B
        1.0 - (a + b - ab),                # 8: not(A or B)
        1.0 - (a + b - 2 * ab),            # 9: not(A xor B)
        1.0 - b,                           # 10: not B
        1.0 - b + ab,                      # 11: B implies A
        1.0 - a,                           # 12: not A
        1.0 - a + ab,                      # 13: A implies B
        1.0 - ab,                          # 14: not(A and B)
        torch.ones_like(a),                # 15: True
    ]
    return torch.stack(ops, dim=-1)


class LogicGateLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, seed: int = 42):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Fixed pseudo-random connections (2 inputs per neuron)
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("in_idx1", torch.randint(0, in_features, (out_features,), generator=g))
        self.register_buffer("in_idx2", torch.randint(0, in_features, (out_features,), generator=g))
        
        # 16 operator logits per neuron, initialized from standard normal (Section 4.1)
        self.weights = nn.Parameter(torch.randn(out_features, 16) * 0.1)
        self.discretized = False

    def discretize(self):
        self.discretized = True

    def undiscretize(self):
        self.discretized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, in_features]
        a = x[:, self.in_idx1]
        b = x[:, self.in_idx2]
        all_ops = apply_all_16_ops(a, b)  # [batch, out_features, 16]

        if self.discretized:
            best_op = self.weights.argmax(dim=-1)  # [out_features]
            # Gather mode operator
            out = all_ops.gather(-1, best_op.unsqueeze(0).unsqueeze(-1).expand(x.shape[0], -1, 1)).squeeze(-1)
            return out.round()
        else:
            probs = F.softmax(self.weights, dim=-1)  # [out_features, 16]
            out = (all_ops * probs.unsqueeze(0)).sum(dim=-1)
            return out


class DiffLogicNet(nn.Module):
    def __init__(self, in_features: int = 784, num_classes: int = 10,
                 num_layers: int = 6, neurons_per_layer: int = 4000,
                 tau: float = 10.0, seed: int = 42):
        super().__init__()
        self.num_classes = num_classes
        self.tau = tau
        self.neurons_per_layer = neurons_per_layer

        layers = []
        curr_in = in_features
        for i in range(num_layers):
            layers.append(LogicGateLayer(curr_in, neurons_per_layer, seed=seed + i * 17))
            curr_in = neurons_per_layer
        self.layers = nn.ModuleList(layers)

    def discretize(self):
        for layer in self.layers:
            layer.discretize()

    def undiscretize(self):
        for layer in self.layers:
            layer.undiscretize()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Binarize inputs: pixels > 0.5 -> 1.0, else 0.0
        h = (x > 0.5).float() if x.dtype != torch.bool else x.float()
        for layer in self.layers:
            h = layer(h)

        # Output aggregation (Equation 3): group output neurons into k classes and sum
        neurons_per_class = self.neurons_per_layer // self.num_classes
        h_grouped = h.view(x.shape[0], self.num_classes, neurons_per_class)
        class_scores = h_grouped.sum(dim=-1) / self.tau
        return class_scores


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
