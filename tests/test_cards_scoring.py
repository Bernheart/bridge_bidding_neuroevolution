"""Card conventions, decoding and scoring. (Old bugs: HCP from the wrong cards, suit-major bid order.)"""

import numpy as np
import pytest

from bridgebid import cards
from bridgebid.constants import action_to_str, bid_action, str_to_action
from bridgebid.scoring import contract_score, imps, score_np


def test_card_index_convention():
    # OpenSpiel order: card = suit + 4 * rank, clubs = 0, rank 12 = ace
    assert cards.pgx_to_os(0) == 51  # pgx 0 = ace of spades
    assert cards.pgx_to_os(1) == 3  # pgx 1 = deuce of spades
    assert cards.pgx_to_os(51) == 44  # pgx 51 = king of clubs
    for c in range(52):
        assert cards.os_to_pgx(cards.pgx_to_os(c)) == c


def test_hcp_counts_honours_not_spot_cards():
    hand = np.zeros(52, bool)
    for rank in (12, 11, 10, 9):  # A K Q J of spades
        hand[3 + 4 * rank] = True
    assert cards.hcp(hand) == 10
    spots = np.zeros(52, bool)
    for rank in (0, 1, 2, 3):  # 2 3 4 5 of spades - the old project scored these
        spots[3 + 4 * rank] = True
    assert cards.hcp(spots) == 0
    assert cards.suit_lengths(hand).tolist() == [0, 0, 0, 4]


def test_balanced():
    assert cards.is_balanced(np.array([4, 3, 3, 3]))
    assert cards.is_balanced(np.array([2, 4, 4, 3]))
    assert cards.is_balanced(np.array([5, 3, 3, 2]))
    assert not cards.is_balanced(np.array([4, 4, 4, 1]))
    assert not cards.is_balanced(np.array([5, 4, 2, 2]))
    assert not cards.is_balanced(np.array([6, 3, 2, 2]))


def test_pack_roundtrip_and_pbn():
    rng = np.random.default_rng(0)
    owners = np.stack([rng.permutation(np.repeat(np.arange(4), 13)) for _ in range(50)]).astype(np.int8)
    keys = cards.encode_owners(owners)
    assert (cards.decode_owners(keys) == owners).all()
    for o in owners[:10]:
        assert (cards.pbn_to_owners(cards.owners_to_pbn(o)) == o).all()
    tricks = rng.integers(0, 14, size=(50, 20)).astype(np.int8)
    assert (cards.decode_tricks(cards.encode_tricks(tricks)) == tricks).all()


def test_pgx_docstring_example():
    # pgx/bridge_bidding.py: _value_to_dds_tricks([4160, 904605, 4160, 904605])
    t = cards.decode_tricks(np.array([[4160, 904605, 4160, 904605]], dtype=np.int32))[0]
    assert t.tolist() == [0, 1, 0, 4, 0, 13, 12, 13, 9, 13, 0, 1, 0, 4, 0, 13, 12, 13, 9, 13]


def test_bid_order_is_bridge_order():
    assert bid_action(1, 0) == 3 and str_to_action("1C") == 3
    assert str_to_action("1NT") == 7
    assert str_to_action("2C") == 8  # 2C is above 1NT
    assert str_to_action("7NT") == 37
    assert str_to_action("1H") < str_to_action("2C")
    assert action_to_str(str_to_action("3NT"), symbols=False) == "3NT"


@pytest.mark.parametrize(
    "contract,vul,x,xx,tricks,expected",
    [
        ("3NT", False, False, False, 9, 400),
        ("3NT", True, False, False, 9, 600),
        ("3NT", True, False, False, 10, 630),
        ("4H", False, False, False, 10, 420),
        ("2S", False, False, False, 9, 140),
        ("1C", False, False, False, 7, 70),
        ("1NT", True, True, False, 6, -200),
        ("1NT", False, True, False, 4, -500),
        ("4S", False, True, False, 6, -800),
        ("4S", True, True, False, 6, -1100),
        ("2H", False, True, False, 8, 470),
        ("1C", False, False, True, 8, 430),  # 1CXX making with an overtrick, non-vul
        ("6NT", False, False, False, 12, 990),
        ("7NT", True, False, True, 13, 2980),
        ("7NT", True, False, True, 0, -7600),
        ("7C", False, False, False, 13, 1440),
    ],
)
def test_known_scores(contract, vul, x, xx, tricks, expected):
    a = str_to_action(contract)
    level, strain = (a - 3) // 5 + 1, (a - 3) % 5
    assert score_np(level, strain, vul, x, xx, tricks) == expected


def test_imp_scale():
    assert imps(np, 10) == 0
    assert imps(np, 20) == 1
    assert imps(np, -50) == -2
    assert imps(np, 420) == 9
    assert imps(np, 430) == 10
    assert imps(np, 3990) == 23
    assert imps(np, 5000) == 24


def test_mlx_scoring_matches_numpy():
    import mlx.core as mx

    lv, st, vu, x, xx, tr = np.meshgrid(np.arange(1, 8), np.arange(5), [0, 1], [0, 1], [0, 1], np.arange(14),
                                        indexing="ij")
    args = [a.ravel() for a in (lv, st, vu, x, xx, tr)]
    ref = contract_score(np, *args)
    got = np.array(contract_score(mx, *[mx.array(a) for a in args]))
    assert (ref == got).all()
    d = np.arange(-5000, 5001, 10)
    assert (imps(np, d) == np.array(imps(mx, mx.array(d)))).all()
