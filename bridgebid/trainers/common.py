"""Pieces shared by the trainers: privileged critic features, periodic evaluation, checkpoint helpers."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from bridgebid.data.sampler import DealBank
from bridgebid.env import core
from bridgebid.eval.diagnostics import diagnose
from bridgebid.eval.ladder import evaluate
from bridgebid.policies.base import Policy

PRIV_DIM = 3 * 52 + 20


def privileged_extra(s: core.State) -> mx.array:
    """[N, 176] uint8: the other three hands (LHO, partner, RHO) and the DD table, relative to the player to act.
    Only the critic sees this, and only during training."""
    cur = core.current_seat(s)
    rel = (cur[:, None] + mx.arange(1, 4)[None, :]) % 4
    n = cur.shape[0]
    others = mx.take_along_axis(s["hands"], mx.broadcast_to(rel[:, :, None], (n, 3, 52)), axis=1).reshape(n, -1)
    rel4 = (cur[:, None] + mx.arange(4)[None, :]) % 4
    idx = (5 * rel4[:, :, None] + mx.arange(5)[None, None, :]).reshape(n, 20)
    tricks = mx.take_along_axis(s["tricks"], idx, axis=1)
    return mx.concatenate([others.astype(mx.uint8), tricks.astype(mx.uint8)], axis=1)


def critic_input(obs: mx.array, extra: mx.array) -> mx.array:
    return mx.concatenate([obs.astype(mx.float32), extra[:, :156].astype(mx.float32),
                           extra[:, 156:].astype(mx.float32) / 13.0], axis=1)


def periodic_eval(policy: Policy, bank: DealBank, opponents: list[str], n_boards: int, cache: dict) -> dict:
    res = evaluate(policy, bank, opponents, n_boards=n_boards, cache=cache)
    out = {f"imps_vs_{k}": round(m, 4) for k, (m, se) in res.items()}
    out.update({f"se_vs_{k}": round(se, 4) for k, (m, se) in res.items()})
    d = diagnose(policy, bank, n_boards=min(n_boards, 2000))
    out.update({k: d[k] for k in ("pass_out_rate", "auction_length", "made_rate", "game_or_higher_rate",
                                  "slam_rate", "doubled_rate", "top_opening_share", "distinct_auctions")})
    out["alarms"] = ",".join(d["alarms"])
    out["top_openings"] = " ".join(f"{k}:{v}" for k, v in d["top_openings"])
    return out


def save_tree(tree, path: Path) -> None:
    flat = dict(tree_flatten(tree))
    mx.save_safetensors(str(path), {k: v for k, v in flat.items() if isinstance(v, mx.array)})


def load_tree(path: Path):
    return tree_unflatten(list(mx.load(str(path)).items()))


def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)))


def key_to_list(key: mx.array) -> list[int]:
    return np.array(key).astype(np.uint64).tolist()


def key_from_list(v: list[int]) -> mx.array:
    return mx.array(np.array(v, dtype=np.uint32))


def linear(start: float, end: float, frac: float) -> float:
    frac = min(max(frac, 0.0), 1.0)
    return start + (end - start) * frac
