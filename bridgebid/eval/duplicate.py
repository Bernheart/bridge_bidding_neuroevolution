"""Duplicate evaluation: every board is played twice with the two sides swapped.

Table 1: A sits North-South, B East-West.  Table 2: B sits North-South, A East-West.
A's result on a board = IMPs(score_NS at table 1 - score_NS at table 2). Luck of the cards cancels out.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from bridgebid.data.sampler import DealBank
from bridgebid.env import core
from bridgebid.policies.base import Policy
from bridgebid.scoring import imps


@dataclass
class TableRecord:
    """Per-board final state of one table (numpy), for diagnostics and viewers."""

    score_ns: np.ndarray
    calls: np.ndarray  # [n, max_len] (-1 padded)
    n_calls: np.ndarray
    level: np.ndarray
    strain: np.ndarray
    declarer: np.ndarray
    doubled: np.ndarray
    redoubled: np.ndarray


@dataclass
class MatchResult:
    a: str
    b: str
    imps: np.ndarray  # [n] IMPs for A per board
    table1: TableRecord  # A = NS
    table2: TableRecord  # B = NS
    board_idx: np.ndarray

    @property
    def mean(self) -> float:
        return float(self.imps.mean())

    @property
    def se(self) -> float:
        return float(self.imps.std(ddof=1) / np.sqrt(len(self.imps)))

    def __str__(self) -> str:
        return f"{self.a} vs {self.b}: {self.mean:+.3f} ± {self.se:.3f} IMPs/board over {len(self.imps):,} boards"


def play(state: core.State, policy_for_ns: mx.array, a: Policy, b: Policy, key: mx.array,
         max_plies: int = 320, on_step=None) -> core.State:
    """Play every deal to the end. policy_for_ns [N] bool: True -> A plays NS (B plays EW), else reversed.

    `on_step(state, obs, mask, a_turn)`, if given, is called before every call (viewers use it to record A's
    probabilities); it must not change the state.
    """
    step = mx.compile(core.step)
    for ply in range(max_plies):
        key, ka, kb = mx.random.split(key, 3)
        obs, mask = core.observe(state), core.legal_mask(state)
        a_turn = ((core.current_seat(state) % 2) == 0) == policy_for_ns
        act = mx.where(a_turn, a.act(state, obs, mask, ka), b.act(state, obs, mask, kb))
        if on_step is not None:
            on_step(state, obs, mask, a_turn)
        state = step(state, act)
        if ply % 8 == 7 and mx.all(state["done"]).item():
            break
    assert mx.all(state["done"]).item(), "auction did not finish"
    return state


def _record(state: core.State) -> TableRecord:
    info = core.contract_info(state)
    n_calls = np.array(state["turn"])
    calls = np.array(state["calls"][:, : max(int(n_calls.max()), 1)])
    return TableRecord(
        score_ns=np.array(state["score_ns"]), calls=calls, n_calls=n_calls,
        level=np.array(info["level"]), strain=np.array(info["strain"]), declarer=np.array(info["declarer"]),
        doubled=np.array(info["doubled"]), redoubled=np.array(info["redoubled"]),
    )


def _concat(records: list[TableRecord]) -> TableRecord:
    width = max(r.calls.shape[1] for r in records)
    calls = np.concatenate([np.pad(r.calls, ((0, 0), (0, width - r.calls.shape[1])), constant_values=-1)
                            for r in records])
    fields = {k: np.concatenate([getattr(r, k) for r in records]) for k in TableRecord.__dataclass_fields__
              if k != "calls"}
    return TableRecord(calls=calls, **fields)


def match(a: Policy, b: Policy, bank: DealBank, n_boards: int = 10_000, start: int = 0, batch: int = 8192,
          seed: int = 0) -> MatchResult:
    key = mx.random.key(seed)
    imp_chunks, t1, t2 = [], [], []
    for s0 in range(start, start + n_boards, batch):
        n = min(batch, start + n_boards - s0)
        st = bank.fixed_state(s0, n)
        both = {k: mx.concatenate([v, v]) for k, v in st.items()}
        a_ns = mx.concatenate([mx.ones((n,), mx.bool_), mx.zeros((n,), mx.bool_)])
        key, k = mx.random.split(key)
        final = play(both, a_ns, a, b, k)
        half1 = {kk: v[:n] for kk, v in final.items()}
        half2 = {kk: v[n:] for kk, v in final.items()}
        imp_chunks.append(np.array(imps(mx, half1["score_ns"] - half2["score_ns"])))
        t1.append(_record(half1))
        t2.append(_record(half2))
    return MatchResult(a.name, b.name, np.concatenate(imp_chunks).astype(np.float64), _concat(t1), _concat(t2),
                       np.arange(start, start + n_boards))
