"""A weighted random bot: a few hand-picked weights x a few features, then a softmax over the legal calls.

It is a ladder baseline between `random` (uniform legal calls) and `pass`, and a small ES experiment
(`scripts/tune_weighted.py` tunes the weights). All calls are natural; there are no conventions.

Features, computed in NumPy over the whole batch (from `rule_bot.auction_features`):

    points         HCP + 1 per card above 4 in a suit
    partner pts    unknown: (40 - HCP) / 3; opened 13, other first bid 9, +2 per further bid, X 11;
                   a pass-only partner at most 8; each partner pass -1 (passes after the first, if only passed)
    pair pts       own points + partner points
    partner len    4 in a suit partner bid; 2.5 in the others once partner bid a suit; else (13 - len) / 3
    target(s)      fit_len(s) - 6 + (pair pts - 20) / 3, fit_len = own + partner len (NT: 7)
                   (a rough law of total tricks: trumps + 1 trick per 3 points above 20)

Scores (sampled from softmax(score / temperature) over legal calls):

    bid L,s  length·(len(s) - 4) + fit·[partner bid s]·(len(s) - 2) - opp·[opponents bid s]
             - over·max(0, L - target(s)) - economy·(L - lowest legal level)
             (NT "length" is +1 balanced, -1 otherwise)
    Pass     pass_base + pass_level·(auction level - max_s target(s))
    X        x_hcp·(HCP - 12) + x_len·(len in their suit - 3) + x_over·max(0, their level - their est. level)
             their est. level = their fit (8 in a suit, 7 in NT) - 6 + (40 - our pair HCP - 20) / 3
    XX       xx (a constant)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import mlx.core as mx
import numpy as np

from bridgebid.policies.base import Policy
from bridgebid.policies.rule_bot import auction_features

_LEVEL = np.arange(35) // 5 + 1
_STRAIN = np.arange(35) % 5


@dataclass(frozen=True)
class Weights:
    length: float = 1.0
    fit: float = 0.5
    opp: float = 2.0
    over: float = 3.0
    economy: float = 1.5
    pass_base: float = 1.5
    pass_level: float = 1.5
    x_hcp: float = 0.2
    x_len: float = 0.6
    x_over: float = 1.0
    xx: float = -3.0
    temperature: float = 0.6

    @classmethod
    def names(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    def to_vector(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in self.names()], np.float64)

    @classmethod
    def from_vector(cls, v) -> Weights:
        return cls(**{n: float(x) for n, x in zip(cls.names(), v, strict=True)})


def save_weights(w: Weights, path: str | Path, **meta) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"weights": asdict(w), **meta}, indent=2) + "\n")


def load_weights(path: str | Path) -> Weights:
    data = json.loads(Path(path).read_text())
    data = data.get("weights", data)
    unknown = set(data) - set(Weights.names())
    if unknown:
        raise ValueError(f"{path}: unknown weights {sorted(unknown)}")
    return Weights(**{k: float(v) for k, v in data.items()})


def features(f: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-deal features of the player to act (see the module docstring)."""
    H, L, bal = f["H"].astype(np.float64), f["L"].astype(np.float64), f["bal"]
    points = H + np.maximum(L - 4, 0).sum(1)

    unknown = (40 - H) / 3
    pb, passes = f["partner_bids"], f["partner_passes"]
    bid_pts = np.where(f["opener_rel"] == 2, 13.0, 9.0) + 2.0 * np.maximum(pb - 1, 0)
    partner_pts = np.where(pb > 0, bid_pts - passes,
                           np.where(f["partner_x"], 11.0 - passes,
                                    np.where(passes > 0, np.minimum(unknown, 8.0) - (passes - 1), unknown)))
    partner_pts = np.maximum(partner_pts, 0)
    pair = points + partner_pts

    ps = f["partner_strains"]
    partner_len = np.where(ps[:, :4], 4.0, np.where(ps[:, :4].any(1, keepdims=True), 2.5, (13 - L) / 3))
    fit_len = np.concatenate([L + partner_len, np.full((len(H), 1), 7.0)], axis=1)
    target = fit_len - 6 + (pair[:, None] - 20) / 3

    length = np.concatenate([L - 4, np.where(bal, 1.0, -1.0)[:, None]], axis=1)
    fit = ps * np.concatenate([np.maximum(L - 2, 0), np.ones((len(H), 1))], axis=1)

    lb = f["last_bid"]
    auction_level = np.where(lb >= 0, lb // 5 + 1, 0)
    their_strain = np.where(lb >= 0, lb % 5, 4)
    their_len = np.where(their_strain < 4, L[f["ar"], np.clip(their_strain, 0, 3)] - 3, 0)
    opp_pts = np.maximum(40 - H - partner_pts, 0)
    their_est = np.where(their_strain < 4, 2.0, 1.0) + (opp_pts - 20) / 3
    return {
        "H": H, "target": target, "length": length, "fit": fit, "opp": f["opp_strains"].astype(np.float64),
        "auction_level": auction_level, "min_level": (lb + 1) // 5 + 1, "their_len": their_len,
        "their_over": np.maximum(auction_level - their_est, 0),
    }


def scores(w: Weights, x: dict[str, np.ndarray]) -> np.ndarray:
    """[N, 38] unnormalised scores (illegal calls are masked later)."""
    n = len(x["H"])
    bid = (w.length * x["length"][:, _STRAIN] + w.fit * x["fit"][:, _STRAIN] - w.opp * x["opp"][:, _STRAIN]
           - w.over * np.maximum(_LEVEL[None, :] - x["target"][:, _STRAIN], 0)
           - w.economy * np.maximum(_LEVEL[None, :] - x["min_level"][:, None], 0))
    pas = w.pass_base + w.pass_level * (x["auction_level"] - x["target"].max(1))
    dbl = w.x_hcp * (x["H"] - 12) + w.x_len * x["their_len"] + w.x_over * x["their_over"]
    rdbl = np.full(n, w.xx)
    return np.concatenate([pas[:, None], dbl[:, None], rdbl[:, None], bid], axis=1).astype(np.float32)


class WeightedRandom(Policy):
    def __init__(self, weights: Weights | None = None, name: str = "weighted-random"):
        self.weights = Weights() if weights is None else weights
        self.name = name

    def logits(self, s, mask: mx.array) -> mx.array:
        sc = scores(self.weights, features(auction_features(s)))
        return mx.where(mask, mx.array(sc) / max(self.weights.temperature, 1e-3), -mx.inf)

    def probs(self, s, mask: mx.array) -> mx.array:
        return mx.softmax(self.logits(s, mask), axis=-1)

    def act(self, s, obs, mask, key):
        return mx.random.categorical(self.logits(s, mask), key=key).astype(mx.int32)
