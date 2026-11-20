"""Weighted random bot: loading, legality, reproducibility, hand examples, tuning smoke, strength (slow)."""

import importlib.util
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from bridgebid.cards import pbn_to_owners
from bridgebid.constants import RANKS, bid_action
from bridgebid.env import core
from bridgebid.policies.base import Policy, load
from bridgebid.policies.rule_bot import auction_features
from bridgebid.policies.weighted_random import WeightedRandom, Weights, load_weights, save_weights

DATA = Path("data/processed")


def _bank(split: str = "eval", limit: int = 10_000):
    from bridgebid.data.sampler import DealBank

    if not (DATA / f"{split}_keys.npy").exists():
        pytest.skip("dataset missing")
    return DealBank(split, limit=limit)


def _deal(seat_hand: str, seat: int = 0) -> str:
    """PBN with `seat_hand` ('S.H.D.C') at `seat`; the other cards dealt in order to the other seats."""
    mine = seat_hand.split(".")
    rest = [(s, r) for s in "SHDC" for r in RANKS[::-1] if r not in mine["SHDC".index(s)]]

    def fmt(cards):
        return ".".join("".join(r for s2, r in cards if s2 == s) for s in "SHDC")

    hands = [fmt(rest[13 * i: 13 * i + 13]) for i in range(3)]
    hands.insert(seat, seat_hand)
    return "N:" + " ".join(hands)


def _state(pbn: str, dealer: int = 0, calls: tuple[int, ...] = ()):
    owners = pbn_to_owners(pbn)
    hands = (owners[None, :] == np.arange(4)[:, None])[None]
    st = core.reset(mx.array(hands), mx.zeros((1, 20), mx.int32), mx.array([dealer]), mx.zeros((1, 2), mx.bool_))
    for c in calls:
        st = core.step(st, mx.array([c]))
    return st


def _probs(st, pol=None) -> np.ndarray:
    pol = pol or WeightedRandom()
    return np.array(pol.probs(st, core.legal_mask(st)))[0]


def test_load_specs(tmp_path):
    pol = load("weighted")
    assert isinstance(pol, Policy) and isinstance(pol, WeightedRandom)
    assert pol.name == "weighted-random" and pol.weights == Weights()
    w = Weights(length=2.5, temperature=0.3)
    save_weights(w, tmp_path / "w.json", note="test")
    tuned = load(f"weighted:{tmp_path / 'w.json'}")
    assert isinstance(tuned, WeightedRandom)
    assert tuned.weights == w == load_weights(tmp_path / "w.json")
    assert tuned.name != pol.name  # both can sit in one ladder
    assert Weights.from_vector(w.to_vector()) == w


def test_in_default_ladder():
    from bridgebid.eval.ladder import DEFAULT_LADDER

    i = DEFAULT_LADDER.index("weighted")
    assert DEFAULT_LADDER[i - 1] == "random" and DEFAULT_LADDER[i + 1] == "pass"


def test_only_legal_calls():
    st = _bank().fixed_state(0, 512)
    pol, key = WeightedRandom(), mx.random.key(0)
    for _ in range(60):
        key, k = mx.random.split(key)
        mask = core.legal_mask(st)
        a = pol.act(st, core.observe(st), mask, k)
        assert np.array(mx.take_along_axis(mask, a[:, None], axis=1)).all()
        st = core.step(st, a)


@pytest.mark.parametrize("n", [512, 1])
def test_same_state_and_key_same_calls(n):
    bank = _bank()

    def run():
        st, key, out = bank.fixed_state(100, n), mx.random.key(7), []
        pol = WeightedRandom()  # a fresh instance: no hidden state
        for _ in range(40):
            key, k = mx.random.split(key)
            a = pol.act(st, core.observe(st), core.legal_mask(st), k)
            out.append(np.array(a))
            st = core.step(st, a)
        return np.stack(out)

    assert (run() == run()).all()


@pytest.mark.parametrize("hand", ["T987.T98.T987.32", "T9876.T98.T98.32", "8765.5432.65432."])
def test_zero_hcp_dealer_passes(hand):
    p = _probs(_state(_deal(hand), dealer=0))
    assert p.argmax() == 0


def test_strong_six_spades_bid_spades():
    p = _probs(_state(_deal("AKQJ32.AK2.K2.32"), dealer=0))
    bids = p[3:]
    assert bids.argmax() % 5 == 3  # the single most likely bid is in spades
    by_strain = bids.reshape(7, 5).sum(0)
    assert by_strain.argmax() == 3


def test_opponents_suit_never_beats_own_long_suit():
    # East holds five spades; North opened 1H: no heart bid may be more likely than the spade bid at its level
    st = _state(_deal("KQ987.A32.K2.432", seat=1), dealer=0, calls=(bid_action(1, 2),))
    p = _probs(st)
    for level in range(2, 8):
        assert p[bid_action(level, 2)] <= p[bid_action(level, 3)]


def test_opponents_suit_property_in_self_play():
    """Same rule on many real auctions: suit s bid by the opponents (not by partner), own longest suit t."""
    st = _bank().fixed_state(0, 512)
    pol, key, checked, ar = WeightedRandom(), mx.random.key(3), 0, np.arange(512)
    for _ in range(12):
        mask = core.legal_mask(st)
        p, m = np.array(pol.probs(st, mask)), np.array(mask)
        f = auction_features(st)
        t = 3 - np.argmax(f["L"][:, ::-1], axis=1)  # longest suit, ties -> higher
        live = ~np.array(st["done"])
        for s in range(4):
            rows = live & f["opp_strains"][:, s] & ~f["partner_strains"][:, s] & (t != s)
            for lv in range(1, 8):
                a_s, a_t = bid_action(lv, s), bid_action(lv, t)
                sel = rows & m[:, a_s] & m[ar, a_t]
                assert (p[sel, a_s] <= p[ar, a_t][sel] + 1e-6).all()
                checked += sel.sum()
        key, k = mx.random.split(key)
        st = core.step(st, pol.act(st, core.observe(st), mask, k))
    assert checked > 100


def test_tuning_smoke_is_reproducible(tmp_path):
    _bank("train", limit=1)
    spec = importlib.util.spec_from_file_location("tune_weighted", Path("scripts/tune_weighted.py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = ["--generations", "2", "--boards", "128", "--lam", "2", "--mu", "1", "--pool", "2000", "--seed", "5",
            "--opponents", "rule", "pass"]
    mod.main([*args, "--out", str(tmp_path / "a.json")])
    mod.main([*args, "--out", str(tmp_path / "b.json")])
    assert (tmp_path / "a.json").read_text() == (tmp_path / "b.json").read_text()
    assert isinstance(load(f"weighted:{tmp_path / 'a.json'}"), WeightedRandom)
    assert mod.generations_for(1, 2000, 8) >= 2  # a 1-minute run is a few generations


@pytest.mark.slow
def test_beats_random_by_five_imps():
    from bridgebid.eval.duplicate import match

    r = match(WeightedRandom(), load("random"), _bank(), n_boards=10_000)
    assert r.mean >= 5.0, str(r)


@pytest.mark.slow
def test_self_play_diagnostics():
    from bridgebid.eval.diagnostics import diagnose

    d = diagnose(WeightedRandom(), _bank(), n_boards=4000)
    assert d["level_hist"][7] / 4000 < 0.05
    assert d["pass_out_rate"] < 0.40
