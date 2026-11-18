"""Duplicate bridge scoring and the IMP scale.

Written against an array module `xp` (numpy or mlx.core) so the env (MLX) and the tests (numpy) share one
implementation. All inputs broadcast; scores are for the declaring side.
"""

from __future__ import annotations

import numpy as np

# Score difference d gives k IMPs when |d| >= IMP_THRESHOLDS[k-1] (scores are multiples of 10).
IMP_THRESHOLDS = (20, 50, 90, 130, 170, 220, 270, 320, 370, 430, 500, 600, 750, 900, 1100, 1300, 1500,
                  1750, 2000, 2250, 2500, 3000, 3500, 4000)


def _arr(xp, v, dtype: str):
    if xp is np:
        return np.asarray(v, dtype=bool if dtype == "bool_" else np.int32)
    return xp.array(v).astype(getattr(xp, dtype))


def contract_score(xp, level, strain, vul, doubled, redoubled, tricks):
    """Score for the declaring side. level 1..7, strain C,D,H,S,NT = 0..4, tricks 0..13 (DD result)."""
    level, strain, tricks = (_arr(xp, v, "int32") for v in (level, strain, tricks))
    vul, x, xx = (_arr(xp, v, "bool_") for v in (vul, doubled, redoubled))
    x = x & ~xx

    need = level + 6
    over = tricks - need  # >= 0 made, < 0 down

    # --- made ---
    per_trick = xp.where(strain < 2, 20, 30)
    base = per_trick * level + xp.where(strain == 4, 10, 0)
    trick_score = base * xp.where(xx, 4, xp.where(x, 2, 1))
    game = trick_score >= 100
    made = trick_score + xp.where(game, xp.where(vul, 500, 300), 50)
    made = made + xp.where(level >= 6, xp.where(vul, 750, 500), 0)
    made = made + xp.where(level == 7, xp.where(vul, 750, 500), 0)
    made = made + xp.where(xx, 100, xp.where(x, 50, 0))  # "insult"
    over_val = xp.where(xx, xp.where(vul, 400, 200), xp.where(x, xp.where(vul, 200, 100), per_trick))
    made = made + over_val * xp.maximum(over, 0)

    # --- down ---
    u = xp.maximum(-over, 0)
    down_plain = u * xp.where(vul, 100, 50)
    down_x_nv = 100 + 200 * xp.minimum(xp.maximum(u - 1, 0), 2) + 300 * xp.maximum(u - 3, 0)
    down_x_v = 200 + 300 * xp.maximum(u - 1, 0)
    down_x = xp.where(vul, down_x_v, down_x_nv)
    down = xp.where(xx, 2 * down_x, xp.where(x, down_x, down_plain))

    return xp.where(over >= 0, made, -down)


def imps(xp, diff):
    """Signed IMPs for a score difference (array or scalar)."""
    if xp is np:
        d = np.abs(np.asarray(diff))
        k = (d[..., None] >= np.asarray(IMP_THRESHOLDS)).sum(-1)
        return np.sign(diff) * k
    d = xp.abs(diff)
    k = (d[..., None] >= xp.array(IMP_THRESHOLDS)).astype(xp.int32).sum(-1)
    return xp.sign(diff).astype(xp.int32) * k


def score_np(level: int, strain: int, vul: bool, doubled: bool, redoubled: bool, tricks: int) -> int:
    return int(contract_score(np, level, strain, vul, doubled, redoubled, tricks))
