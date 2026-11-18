"""Card conventions and deal decoding.

Internal card index = OpenSpiel order (also used in the observation):
    card = suit + 4 * rank,  suit C, D, H, S = 0..3,  rank 0 = deuce ... 12 = ace.
So the ace of spades is 3 + 4 * 12 = 51 and the deuce of clubs is 0.

Pgx / DD-dataset order (only used when decoding the dataset):
    card = 13 * suit + rank,  suit S, H, D, C = 0..3,  rank 0 = ace, 1 = deuce, ..., 12 = king.

Dataset rows (HF sotetsuk/dds_dataset, packed by Pgx):
    key[i]   for Pgx suit i: 13 base-4 digits, most significant first; digit j = seat owning Pgx card 13*i + j.
    value[d] for declarer seat d (N, E, S, W): 5 hex digits, most significant first = tricks in C, D, H, S, NT.
Decoded DD table: tricks[5 * declarer + strain].
"""

from __future__ import annotations

import numpy as np

from bridgebid.constants import RANKS, SEATS

HCP_BY_RANK = np.array([0] * 9 + [1, 2, 3, 4], dtype=np.int32)  # J, Q, K, A


def pgx_to_os(card_pgx: np.ndarray | int) -> np.ndarray | int:
    suit = 3 - card_pgx // 13
    rank = (card_pgx % 13 + 12) % 13
    return suit + 4 * rank


def os_to_pgx(card_os: np.ndarray | int) -> np.ndarray | int:
    suit = 3 - card_os % 4
    rank = (card_os // 4 + 1) % 13
    return 13 * suit + rank


# PERM_OS_TO_PGX[c_os] = c_pgx
PERM_OS_TO_PGX = np.array([os_to_pgx(c) for c in range(52)], dtype=np.int32)

_KEY_SHIFTS = np.arange(24, -1, -2, dtype=np.int32)  # 13 base-4 digits
_VAL_SHIFTS = np.arange(16, -1, -4, dtype=np.int32)  # 5 hex digits


def decode_owners(keys: np.ndarray) -> np.ndarray:
    """keys [N, 4] int32 -> owner seat of every card, [N, 52] int8 in OpenSpiel card order."""
    digits = (keys[:, :, None] >> _KEY_SHIFTS) & 3  # [N, 4 pgx suits, 13 pgx ranks]
    owner_pgx = digits.reshape(len(keys), 52)
    return owner_pgx[:, PERM_OS_TO_PGX].astype(np.int8)


def encode_owners(owner_os: np.ndarray) -> np.ndarray:
    """Inverse of decode_owners: [N, 52] owners (OpenSpiel order) -> keys [N, 4] int32."""
    owner_pgx = np.empty_like(owner_os, dtype=np.int64)
    owner_pgx[:, PERM_OS_TO_PGX] = owner_os
    digits = owner_pgx.reshape(len(owner_os), 4, 13)
    return (digits << _KEY_SHIFTS.astype(np.int64)).sum(-1).astype(np.int32)


def decode_tricks(values: np.ndarray) -> np.ndarray:
    """values [N, 4] int32 -> DD tricks [N, 20] int8, index 5 * declarer + strain (C, D, H, S, NT)."""
    t = (values[:, :, None] >> _VAL_SHIFTS) & 0xF
    return t.reshape(len(values), 20).astype(np.int8)


def encode_tricks(tricks: np.ndarray) -> np.ndarray:
    t = tricks.reshape(len(tricks), 4, 5).astype(np.int64)
    return (t << _VAL_SHIFTS.astype(np.int64)).sum(-1).astype(np.int32)


def owners_to_hands(owner_os: np.ndarray) -> np.ndarray:
    """[N, 52] owners -> [N, 4, 52] bool hands."""
    return owner_os[:, None, :] == np.arange(4, dtype=owner_os.dtype)[None, :, None]


def hcp(hand: np.ndarray) -> np.ndarray:
    """High-card points of bool hand(s) [..., 52] (A=4, K=3, Q=2, J=1)."""
    return (hand.reshape(*hand.shape[:-1], 13, 4) * HCP_BY_RANK[:, None]).sum((-1, -2))


def suit_lengths(hand: np.ndarray) -> np.ndarray:
    """[..., 52] bool -> [..., 4] lengths in C, D, H, S order."""
    return hand.reshape(*hand.shape[:-1], 13, 4).sum(-2)


def is_balanced(lengths: np.ndarray) -> np.ndarray:
    """4333, 4432 or 5332."""
    s = np.sort(lengths, axis=-1)
    return (s[..., 0] >= 2) & (s[..., 1] >= 3)


def hand_to_str(hand: np.ndarray, sep: str = " ") -> str:
    """bool [52] -> 'S:AK72 H:Q93 D:J5 C:T862' style string (spades first)."""
    parts = []
    for suit, sym in zip((3, 2, 1, 0), "SHDC", strict=True):
        ranks = [RANKS[r] for r in range(12, -1, -1) if hand[suit + 4 * r]]
        parts.append(f"{sym}:{''.join(ranks) or '-'}")
    return sep.join(parts)


def owners_to_pbn(owner_os: np.ndarray) -> str:
    """[52] owners -> PBN deal string 'N:spades.hearts.diamonds.clubs ...' (N, E, S, W)."""
    hands = []
    for seat in range(4):
        suits = []
        for suit in (3, 2, 1, 0):
            suits.append("".join(RANKS[r] for r in range(12, -1, -1) if owner_os[suit + 4 * r] == seat))
        hands.append(".".join(suits))
    return "N:" + " ".join(hands)


def pbn_to_owners(pbn: str) -> np.ndarray:
    first, rest = pbn.split(":", 1)
    start = SEATS.index(first.strip().upper())
    owner = np.full(52, -1, dtype=np.int8)
    for i, hand in enumerate(rest.split()):
        seat = (start + i) % 4
        for suit, cards in zip((3, 2, 1, 0), hand.split("."), strict=True):
            for ch in cards:
                owner[suit + 4 * RANKS.index(ch.upper())] = seat
    if (owner < 0).any():
        raise ValueError("PBN does not contain 52 cards")
    return owner
