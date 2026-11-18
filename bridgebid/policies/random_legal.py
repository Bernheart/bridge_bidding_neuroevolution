"""Random policies: uniform over legal calls (a ladder baseline) and a biased one for stress tests."""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def uniform_legal(mask: mx.array, key: mx.array) -> mx.array:
    """[N,38] bool mask -> [N] uniformly random legal action."""
    logits = mx.where(mask, 0.0, -mx.inf)
    return mx.random.categorical(logits, key=key)


def biased_random_action(rng: np.random.Generator, mask: np.ndarray) -> int:
    """Random legal call that exercises the rules: many passes (incl. pass-outs), X/XX, small bids."""
    r = rng.random()
    if r < 0.45:
        return 0
    if mask[1] and r < 0.62:
        return 1
    if mask[2] and r < 0.80:
        return 2
    bids = np.flatnonzero(mask[3:]) + 3
    if len(bids) == 0:
        return 0
    return int(bids[min(rng.geometric(0.35) - 1, len(bids) - 1)])
