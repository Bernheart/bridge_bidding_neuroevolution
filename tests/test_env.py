"""Batched MLX env == readable reference env, call by call (observations, legal masks, end, score)."""

import mlx.core as mx
import numpy as np
import pytest

from bridgebid.cards import decode_owners, decode_tricks
from bridgebid.env import core
from bridgebid.env.reference import Auction, Board
from bridgebid.policies.random_legal import biased_random_action


def run_lockstep(keys, vals, dealer, vul, seed=0, compiled=False):
    owners, tricks = decode_owners(keys), decode_tricks(vals)
    n = len(keys)
    refs = [Auction(Board(owners[i], tricks[i], int(dealer[i]), (bool(vul[i, 0]), bool(vul[i, 1]))))
            for i in range(n)]
    hands = owners[:, None, :] == np.arange(4)[None, :, None]
    s = core.reset(mx.array(hands), mx.array(tricks.astype(np.int32)), mx.array(dealer.astype(np.int32)),
                   mx.array(vul))
    step = mx.compile(core.step) if compiled else core.step
    rng = np.random.default_rng(seed)
    plies = 0
    while not all(r.done for r in refs):
        mask = np.array(core.legal_mask(s))
        obs = np.array(core.observe(s)).astype(bool)
        cur = np.array(core.current_seat(s))
        actions = np.zeros(n, dtype=np.int32)
        for i, r in enumerate(refs):
            if r.done:
                continue
            assert cur[i] == r.current
            assert (mask[i] == r.legal()).all(), f"deal {i}: legal mask differs after {r.calls}"
            assert (obs[i] == r.observation()).all(), f"deal {i}: observation differs after {r.calls}"
            actions[i] = biased_random_action(rng, mask[i])
            r.step(int(actions[i]))
        s = step(s, mx.array(actions))
        plies += 1
    done = np.array(s["done"])
    score = np.array(s["score_ns"])
    calls = np.array(s["calls"])
    for i, r in enumerate(refs):
        assert done[i]
        assert score[i] == r.score_ns(), f"deal {i}: score {score[i]} != {r.score_ns()} for {r.calls}"
        assert calls[i, : len(r.calls)].tolist() == r.calls
    return refs


@pytest.mark.parametrize("compiled", [False, True])
def test_mlx_env_matches_reference(eval_arrays, compiled):
    a = eval_arrays
    refs = run_lockstep(a["keys"][:400], a["vals"][:400], a["dealer"][:400], a["vul"][:400], seed=int(compiled),
                        compiled=compiled)
    # the random games must actually cover the interesting cases
    calls = [c for r in refs for c in r.calls]
    assert calls.count(1) > 50 and calls.count(2) > 20
    assert sum(r.contract() is None for r in refs) > 5  # some pass-outs
    assert sum(r.contract() is not None and r.contract()[4] for r in refs) > 5  # some redoubled contracts


def test_reset_where_and_terminal_freeze(eval_arrays):
    a = eval_arrays
    owners, tricks = decode_owners(a["keys"][:8]), decode_tricks(a["vals"][:8])
    hands = mx.array(owners[:, None, :] == np.arange(4)[None, :, None])
    s = core.reset(hands, mx.array(tricks.astype(np.int32)), mx.array(np.zeros(8, np.int32)),
                   mx.array(np.zeros((8, 2), bool)))
    for _ in range(4):  # four passes: everything passed out
        s = core.step(s, mx.zeros((8,), mx.int32))
    assert np.array(s["done"]).all() and (np.array(s["score_ns"]) == 0).all()
    s2 = core.step(s, mx.full((8,), 7, mx.int32))  # stepping a finished deal changes nothing
    assert np.array(s2["turn"]).tolist() == np.array(s["turn"]).tolist()
    fresh = core.reset(hands, mx.array(tricks.astype(np.int32)), mx.array(np.zeros(8, np.int32)),
                       mx.array(np.zeros((8, 2), bool)))
    mask = mx.array(np.arange(8) % 2 == 0)
    s3 = core.reset_where(s, mask, fresh)
    assert np.array(s3["done"]).tolist() == [False, True] * 4
