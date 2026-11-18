"""Board data for the HTML viewer: hands, DD table, and the model's duplicate auctions against each opponent.

Every opponent is played exactly as `eval/duplicate.match` plays it (same boards, same seed, same key splits),
so the IMPs in the page are the evaluation IMPs. The model's top-k calls are recorded on the GPU during play
and converted to numpy once at the end.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import mlx.core as mx
import numpy as np

from bridgebid.cards import hcp
from bridgebid.constants import RANKS
from bridgebid.data.sampler import DealBank
from bridgebid.eval.duplicate import MatchResult, TableRecord, _concat, _record, play
from bridgebid.policies.base import NeuralPolicy, Policy
from bridgebid.scoring import imps

DATA_MARKER = "/*__BOARD_DATA__*/null"


def top_calls(policy: Policy, obs: mx.array, mask: mx.array, k: int = 3) -> tuple[mx.array, mx.array] | None:
    """(actions [N,k], probabilities [N,k]) of a neural policy, lazily; None for other policies."""
    if not isinstance(policy, NeuralPolicy):
        return None
    probs = mx.softmax(policy.logits(obs, mask), axis=-1)
    idx = mx.argsort(-probs, axis=-1)[:, :k]
    return idx, mx.take_along_axis(probs, idx, axis=1)


def topk_list(idx: np.ndarray, p: np.ndarray, min_p: float = 0.005) -> list[list]:
    """One position's top-k as [[action, prob], ...], dropping calls below `min_p`."""
    return [[int(a), round(float(q), 3)] for a, q in zip(idx, p, strict=True) if q >= min_p]


def hand_suits(hand: np.ndarray) -> list[str]:
    """bool [52] -> ['AKT2', 'Q93', 'J5', '862'] in spades, hearts, diamonds, clubs order."""
    return ["".join(RANKS[r] for r in range(12, -1, -1) if hand[suit + 4 * r]) for suit in (3, 2, 1, 0)]


def match_with_probs(model: Policy, opp: Policy, bank: DealBank, n_boards: int, start: int = 0,
                     batch: int = 8192, seed: int = 0, top: int = 3):
    """`duplicate.match(model, opp, ...)` plus the model's top-k calls at every one of its turns.

    Returns (MatchResult, probs) where probs is None (non-neural model) or a pair of arrays
    (actions [2n, plies, k], probabilities [2n, plies, k]) with rows 0..n-1 = table 1 and n..2n-1 = table 2
    (per chunk, concatenated); entries at plies where the model did not act are meaningless.
    """
    key = mx.random.key(seed)
    imp_chunks, t1, t2, p1, p2 = [], [], [], [], []
    for s0 in range(start, start + n_boards, batch):
        n = min(batch, start + n_boards - s0)
        st = bank.fixed_state(s0, n)
        both = {k: mx.concatenate([v, v]) for k, v in st.items()}
        a_ns = mx.concatenate([mx.ones((n,), mx.bool_), mx.zeros((n,), mx.bool_)])
        key, k = mx.random.split(key)
        steps: list[tuple[mx.array, mx.array]] = []

        def record(state, obs, mask, a_turn, steps=steps):
            tc = top_calls(model, obs, mask, top)
            if tc is not None:
                steps.append(tc)

        final = play(both, a_ns, model, opp, k, on_step=record)
        half1 = {kk: v[:n] for kk, v in final.items()}
        half2 = {kk: v[n:] for kk, v in final.items()}
        imp_chunks.append(np.array(imps(mx, half1["score_ns"] - half2["score_ns"])))
        t1.append(_record(half1))
        t2.append(_record(half2))
        if steps:
            idx = mx.stack([s[0] for s in steps], axis=1)
            pr = mx.stack([s[1] for s in steps], axis=1)
            mx.eval(idx, pr)
            idx, pr = np.array(idx), np.array(pr)
            p1.append((idx[:n], pr[:n]))
            p2.append((idx[n:], pr[n:]))
    res = MatchResult(model.name, opp.name, np.concatenate(imp_chunks).astype(np.float64), _concat(t1),
                      _concat(t2), np.arange(start, start + n_boards))
    if not p1:
        return res, None
    return res, (_cat_plies(p1), _cat_plies(p2))


def _cat_plies(parts: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    width = max(p[0].shape[1] for p in parts)

    def pad(a):
        return np.pad(a, ((0, 0), (0, width - a.shape[1]), (0, 0)))

    return np.concatenate([pad(p[0]) for p in parts]), np.concatenate([pad(p[1]) for p in parts])


def contract_dict(level: int, strain: int, declarer: int, doubled: bool, redoubled: bool, tricks: int | None,
                  score_ns: int) -> dict:
    return {"level": int(level), "strain": int(strain), "declarer": int(declarer),
            "x": bool(doubled and not redoubled), "xx": bool(redoubled),
            "tricks": None if tricks is None else int(tricks), "score": int(score_ns)}


def seat_of(dealer: int, ply: int) -> int:
    return (dealer + ply) % 4


def build_board_data(model: Policy, opponents: list[Policy], bank: DealBank, start: int = 0, n: int = 200,
                     seed: int = 0, top: int = 3) -> dict:
    """Everything the page shows, as JSON-ready Python data."""
    st = bank.fixed_state(start, n)
    hands, dealer, vul = np.array(st["hands"]), np.array(st["dealer"]), np.array(st["vul"])
    tricks = np.array(st["tricks"])
    pts = hcp(hands)
    boards = [{
        "n": start + i + 1, "dealer": int(dealer[i]), "vul": [bool(vul[i, 0]), bool(vul[i, 1])],
        "hands": [hand_suits(hands[i, s]) for s in range(4)], "hcp": [int(x) for x in pts[i]],
        "dd": [int(x) for x in tricks[i]], "pairs": [],
    } for i in range(n)]
    for opp in opponents:
        res, probs = match_with_probs(model, opp, bank, n_boards=n, start=start, seed=seed, top=top)
        for i, b in enumerate(boards):
            tables = []
            for rec, model_ns, pr in ((res.table1, True, None if probs is None else probs[0]),
                                      (res.table2, False, None if probs is None else probs[1])):
                tables.append(_auction(rec, i, tricks[i], pr, model_ns, int(dealer[i])))
            b["pairs"].append({"imps": int(res.imps[i]), "tables": tables})
    return {"model": model.name, "opponents": [o.name for o in opponents], "split": bank.split, "start": start,
            "seed": seed, "boards": boards}


def _auction(rec: TableRecord, i: int, tricks: np.ndarray, probs, model_ns: bool, dealer: int) -> dict:
    n_calls = int(rec.n_calls[i])
    calls = [int(c) for c in rec.calls[i, :n_calls]]
    lv, d = int(rec.level[i]), int(rec.declarer[i])
    t = None if lv == 0 else int(tricks[5 * d + int(rec.strain[i])])
    out = {"calls": calls,
           "contract": contract_dict(lv, rec.strain[i], d, rec.doubled[i], rec.redoubled[i], t, rec.score_ns[i])}
    if probs is not None:
        idx, pr = probs
        side = 0 if model_ns else 1
        out["probs"] = [topk_list(idx[i, j], pr[i, j]) if seat_of(dealer, j) % 2 == side else None
                        for j in range(n_calls)]
    return out


def render_html(data: dict, serve: bool = False) -> str:
    """The page with the board data inlined (a single self-contained file)."""
    page = resources.files("bridgebid.ui").joinpath("static/viewer.html").read_text(encoding="utf-8")
    payload = json.dumps({**data, "serve": serve}, separators=(",", ":"))
    payload = payload.replace("</", "<\\/")  # never close the <script> early
    assert DATA_MARKER in page
    return page.replace(DATA_MARKER, payload)


def default_out(spec: str, name: str) -> Path:
    """runs/<run>/boards.html for a checkpoint under runs/<run>/ckpt/, else runs/viewer/<name>.html."""
    p = Path(spec)
    if p.suffix == ".safetensors" and p.parent.name == "ckpt":
        return p.parent.parent / "boards.html"
    safe = "".join(c if c.isalnum() or c in "-_." else "-" for c in name)
    return Path("runs/viewer") / f"{safe}.html"
