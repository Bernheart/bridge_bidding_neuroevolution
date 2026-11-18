"""Duplicate scoring properties and rule-bot sanity."""

import mlx.core as mx
import numpy as np
import pytest

from bridgebid.cards import pbn_to_owners
from bridgebid.env import core
from bridgebid.policies.base import AlwaysPass, RandomLegal
from bridgebid.policies.rule_bot import RuleBot


@pytest.fixture(scope="module")
def bank():
    from pathlib import Path

    from bridgebid.data.sampler import DealBank

    if not Path("data/processed/eval_keys.npy").exists():
        pytest.skip("dataset missing")
    return DealBank("eval", limit=3000)


def test_duplicate_self_play_is_zero_and_antisymmetric(bank):
    from bridgebid.eval.duplicate import match

    r = match(RuleBot(), RuleBot(), bank, n_boards=1000)
    assert (r.imps == 0).all()
    ab = match(RuleBot(), AlwaysPass(), bank, n_boards=1000)
    ba = match(AlwaysPass(), RuleBot(), bank, n_boards=1000)
    assert (ab.imps == -ba.imps).all()
    assert ab.mean > 2.0  # bidding beats never bidding


def test_random_policy_only_makes_legal_calls(bank):
    st = bank.fixed_state(0, 512)
    pol = RandomLegal()
    key = mx.random.key(0)
    for _ in range(60):
        key, k = mx.random.split(key)
        mask = core.legal_mask(st)
        a = pol.act(st, core.observe(st), mask, k)
        assert np.array(mx.take_along_axis(mask, a[:, None], axis=1)).all()
        st = core.step(st, a)


def _state_for(pbn: str, dealer: int = 0):
    owners = pbn_to_owners(pbn)
    hands = (owners[None, :] == np.arange(4)[:, None])[None]
    return core.reset(mx.array(hands), mx.zeros((1, 20), mx.int32), mx.array([dealer]), mx.zeros((1, 2), mx.bool_))


@pytest.mark.parametrize("north,expected", [
    ("AK2.Q93.QJ54.K32", 7),  # 15 balanced -> 1NT
    ("AK2.KQ3.QJ54.K32", 4),  # 18 balanced -> 1D (better minor)
    ("AKJ72.Q3.Q54.J32", 6),  # 13, five spades -> 1S
    ("AK2.K3.854.KJ432", 3),  # 14, five clubs -> 1C
    ("K2.QJ9873.54.432", 10),  # 6 hearts 6 HCP -> 2H weak
    ("Q72.J53.Q654.432", 0),  # 5 HCP -> pass
])
def test_rule_bot_openings(north, expected):
    from bridgebid.constants import RANKS

    rest = [(suit, r) for suit in "SHDC" for r in RANKS[::-1]
            if r not in north.split(".")["SHDC".index(suit)]]
    others = [rest[0:13], rest[13:26], rest[26:39]]

    def fmt(cards):
        return ".".join("".join(r for s2, r in cards if s2 == s) for s in "SHDC")

    st = _state_for(f"N:{north} {fmt(others[0])} {fmt(others[1])} {fmt(others[2])}", dealer=0)
    a = RuleBot().act(st, core.observe(st), core.legal_mask(st), None)
    assert int(a[0]) == expected
