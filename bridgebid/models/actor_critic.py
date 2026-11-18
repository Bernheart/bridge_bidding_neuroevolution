"""MLP policy (+ optional value head) used by PPO, by the ported brl baselines and for evaluation."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from bridgebid.constants import NUM_ACTIONS, OBS_DIM


class MLP(nn.Module):
    def __init__(self, in_dim: int, width: int, depth: int):
        super().__init__()
        dims = [in_dim] + [width] * depth
        self.layers = [nn.Linear(i, o) for i, o in zip(dims[:-1], dims[1:], strict=True)]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = nn.relu(layer(x))
        return x


class PolicyNet(nn.Module):
    """Torso -> 38 logits (+ value if `value_head`). Same shape as brl's 'DeepMind' model at width=1024."""

    def __init__(self, width: int = 512, depth: int = 4, value_head: bool = True, in_dim: int = OBS_DIM):
        super().__init__()
        self.config = {"width": width, "depth": depth, "value_head": value_head, "in_dim": in_dim}
        self.torso = MLP(in_dim, width, depth)
        self.pi = nn.Linear(width, NUM_ACTIONS)
        if value_head:
            self.v = nn.Linear(width, 1)
        # small output layer -> near-uniform initial policy
        self.pi.weight = self.pi.weight * 0.01
        self.pi.bias = mx.zeros_like(self.pi.bias)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array | None]:
        h = self.torso(x)
        v = self.v(h).squeeze(-1) if "v" in self else None
        return self.pi(h), v


class ValueNet(nn.Module):
    """Separate critic (optionally 'privileged': sees all hands and the DD table during training only)."""

    def __init__(self, in_dim: int, width: int = 512, depth: int = 3):
        super().__init__()
        self.config = {"width": width, "depth": depth, "in_dim": in_dim}
        self.torso = MLP(in_dim, width, depth)
        self.v = nn.Linear(width, 1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.v(self.torso(x)).squeeze(-1)
