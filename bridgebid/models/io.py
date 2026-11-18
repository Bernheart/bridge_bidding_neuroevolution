"""Save / load models as safetensors + a JSON sidecar with the architecture."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from bridgebid.models.actor_critic import PolicyNet


def save_policy(model: PolicyNet, path: str | Path, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path.with_suffix(".safetensors")), dict(tree_flatten(model.parameters())))
    path.with_suffix(".json").write_text(json.dumps({"config": model.config, **(meta or {})}, indent=2))


def load_policy(path: str | Path) -> PolicyNet:
    path = Path(path)
    cfg = json.loads(path.with_suffix(".json").read_text())["config"]
    model = PolicyNet(**cfg)
    weights = mx.load(str(path.with_suffix(".safetensors")))
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    return model
