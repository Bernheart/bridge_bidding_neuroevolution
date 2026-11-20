"""Policy interface: act(state, obs, mask, key) -> actions [N]. All policies work on whole batches."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from bridgebid.env.core import State
from bridgebid.policies.random_legal import uniform_legal


class Policy:
    name = "policy"

    def act(self, s: State, obs: mx.array, mask: mx.array, key: mx.array) -> mx.array:
        raise NotImplementedError


class AlwaysPass(Policy):
    name = "always-pass"

    def act(self, s, obs, mask, key):
        return mx.zeros(obs.shape[:1], mx.int32)


class RandomLegal(Policy):
    name = "random-legal"

    def act(self, s, obs, mask, key):
        return uniform_legal(mask, key)


class NeuralPolicy(Policy):
    """Wraps a PolicyNet. Greedy by default (evaluation); `temperature` > 0 samples."""

    def __init__(self, model, name: str = "neural", temperature: float = 0.0):
        self.model = model
        self.name = name
        self.temperature = temperature
        self._logits = mx.compile(lambda x: self.model(x)[0], inputs=[model.state])

    def logits(self, obs: mx.array, mask: mx.array) -> mx.array:
        return mx.where(mask, self._logits(obs), -1e9)

    def act(self, s, obs, mask, key):
        logits = self.logits(obs, mask)
        if self.temperature <= 0:
            return mx.argmax(logits, axis=-1).astype(mx.int32)
        return mx.random.categorical(logits / self.temperature, key=key).astype(mx.int32)


TUNED_WEIGHTS = Path(__file__).resolve().parents[2] / "results" / "weighted_tuned.json"


def load(spec: str) -> Policy:
    """'pass' | 'random' | 'rule' | 'weighted' | 'weighted-tuned' | 'weighted:<weights.json>' | 'brl:<name>' |
    path to a .safetensors policy checkpoint. 'weighted-tuned' = the committed ES-tuned weights."""
    from bridgebid.models.io import load_policy
    from bridgebid.policies.rule_bot import RuleBot
    from bridgebid.policies.weighted_random import WeightedRandom, load_weights

    if spec in ("pass", "always-pass"):
        return AlwaysPass()
    if spec in ("random", "random-legal"):
        return RandomLegal()
    if spec in ("rule", "rule-bot"):
        return RuleBot()
    if spec in ("weighted", "weighted-random"):
        return WeightedRandom()
    if spec == "weighted-tuned":
        return WeightedRandom(load_weights(TUNED_WEIGHTS), name="weighted-tuned")
    if spec.startswith("weighted:"):
        path = Path(spec[len("weighted:"):])
        return WeightedRandom(load_weights(path), name=spec)  # the path keeps different files apart
    if spec.startswith("brl:"):
        name = spec[4:]
        name = name if name.startswith("model-") else f"model-{name}"
        return NeuralPolicy(load_policy(Path("models/brl") / name), name=f"brl:{name[6:]}")
    path = Path(spec)
    return NeuralPolicy(load_policy(path), name=path.stem if path.stem != "policy" else path.parent.parent.name)
