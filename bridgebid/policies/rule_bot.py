"""A small hand-written natural bidding system ("mini-SAYC"), vectorised in NumPy over the whole batch.

It is a ladder baseline and a fixed anchor opponent, not a strong player. It only uses its own hand and the
auction. Rules are applied in priority order; the first that fires decides the call; illegal results -> Pass.

Openings: 15-17 bal 1NT, 20-21 bal 2NT, 12+ one of a suit (5-card majors, better minor),
          5-10 HCP with a 6-card D/H/S weak two, 7-card suit three-level preempt.
Responses (opponents silent): raises with 3-card support by strength, 1-level majors over minors,
          NT responses by strength, natural game/invite choices over 1NT/2NT.
Opener's rebid: accept/decline raises and invites, raise partner's major with 4 cards, NT rebids.
Competition: 1NT / suit overcalls with 5+ cards, takeout doubles, simple advances.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from bridgebid.cards import hcp, is_balanced, suit_lengths
from bridgebid.policies.base import Policy

PASS, X = 0, 1
NT1, NT2, NT3, NT4, NT6 = 7, 12, 17, 22, 32


def bid(level, strain):
    return 3 + 5 * (np.asarray(level) - 1) + np.asarray(strain)


def cheapest(strain, last_bid):
    """Cheapest legal bid in `strain` over `last_bid` (bid index, -1 = none); -1 if above 7NT."""
    lv = np.where(last_bid < 0, 1, last_bid // 5 + 1)
    ls = np.where(last_bid < 0, -1, last_bid % 5)
    level = np.where(strain > ls, lv, lv + 1)
    return np.where(level <= 7, bid(level, strain), -1)


def jump(action):
    action = np.asarray(action)
    return np.where((action >= 3) & (action + 5 <= 37), action + 5, -1)


def auction_features(s) -> dict[str, np.ndarray]:
    turn, dealer = np.array(s["turn"]), np.array(s["dealer"])
    width = max(int(turn.max()), 1)  # only the calls made so far
    calls = np.array(s["calls"][:, :width])
    n = len(turn)
    ar = np.arange(n)
    cur = (dealer + turn) % 4
    hand = np.array(s["hands"])[ar, cur]
    lengths = suit_lengths(hand)
    idx = np.arange(width)[None, :]
    valid = idx < turn[:, None]
    rel = (idx - turn[:, None]) % 4  # 0 me, 1 LHO, 2 partner, 3 RHO
    isbid = valid & (calls >= 3)

    def first(m):
        pos = m.argmax(1)
        return np.where(m.any(1), calls[ar, pos], -1), np.where(m.any(1), rel[ar, pos], -1)

    def last(m):
        pos = width - 1 - m[:, ::-1].argmax(1)
        return np.where(m.any(1), calls[ar, pos], -1)

    opening, opener_rel = first(isbid)
    return {
        "ar": ar, "H": hcp(hand), "L": lengths, "bal": is_balanced(lengths),
        "last_bid": np.array(s["last_bid"]), "opening": opening, "opener_rel": opener_rel,
        "partner_first": first(isbid & (rel == 2))[0], "my_first": first(isbid & (rel == 0))[0],
        "partner_last": last(valid & (rel == 2)),
        "my_bids": (isbid & (rel == 0)).sum(1), "partner_bids": (isbid & (rel == 2)).sum(1),
        "opp_bid": (isbid & (rel % 2 == 1)).any(1),
    }


def decide(f: dict[str, np.ndarray], mask: np.ndarray) -> np.ndarray:
    ar, H, L, bal, lb = f["ar"], f["H"], f["L"], f["bal"], f["last_bid"]
    C, D, Hh, S = L[:, 0], L[:, 1], L[:, 2], L[:, 3]
    act = np.full(len(ar), -1)

    def put(cond, action):
        action = np.broadcast_to(np.asarray(action), act.shape)
        m = (act < 0) & cond & (action >= 0) & (action <= 37)
        act[m] = action[m]

    def length(strain):
        return L[ar, np.clip(strain, 0, 3)] * (np.asarray(strain) < 4)

    long_major = np.where(S >= Hh, 3, 2)
    len_major = np.maximum(S, Hh)

    # ---------------- opening ----------------
    op = f["opening"] < 0
    put(op & bal & (H >= 15) & (H <= 17), NT1)
    put(op & bal & (H >= 20) & (H <= 21), NT2)
    open_suit = np.where((S >= 5) & (S >= Hh), 3, np.where(Hh >= 5, 2, np.where((D > C) | ((D == C) & (D >= 4)), 1, 0)))
    put(op & (H >= 12), bid(1, open_suit))
    for s in (3, 2, 1, 0):
        put(op & (H >= 5) & (H <= 10) & (L[:, s] >= 7), bid(3, s))
    for s in (3, 2, 1):
        put(op & (H >= 5) & (H <= 10) & (L[:, s] == 6), bid(2, s))
    put(op, PASS)

    # ---------------- responding (partner opened, opponents silent) ----------------
    P, Ps = f["opening"], (f["opening"] - 3) % 5
    resp = (f["opener_rel"] == 2) & (f["my_first"] < 0) & ~f["opp_bid"] & (f["partner_bids"] == 1)
    r = resp & (P == NT1)
    put(r & (H <= 7) & (len_major >= 5), bid(2, long_major))
    put(r & (H <= 7), PASS)
    put(r & (H <= 9) & (len_major >= 6), bid(3, long_major))
    put(r & (H <= 9), NT2)
    put(r & (H <= 15) & (len_major >= 6), bid(4, long_major))
    put(r & (H <= 15), NT3)
    put(r & (H <= 17), NT4)
    put(r, NT6)
    r = resp & (P == NT2)
    put(r & (H >= 12), NT6)
    put(r & (H >= 4) & (len_major >= 6), bid(4, long_major))
    put(r & (H >= 4), NT3)
    put(r, PASS)
    sup3 = length(Ps) >= 3
    r = resp & ((P == bid(1, 2)) | (P == bid(1, 3)))
    put(r & (H < 6), PASS)
    put(r & sup3 & (H <= 9), bid(2, Ps))
    put(r & sup3 & (H <= 12), bid(3, Ps))
    put(r & sup3, bid(4, Ps))
    put(r & (P == bid(1, 2)) & (S >= 4), bid(1, 3))
    put(r & (H <= 10), NT1)
    put(r & (H <= 12), NT2)
    put(r, NT3)
    r = resp & ((P == bid(1, 0)) | (P == bid(1, 1)))
    put(r & (H < 6), PASS)
    put(r & (S >= 5) & (S > Hh), bid(1, 3))
    put(r & (Hh >= 4), bid(1, 2))
    put(r & (S >= 4), bid(1, 3))
    put(r & (P == bid(1, 0)) & (D >= 4), bid(1, 1))
    put(r & (length(Ps) >= 5) & (H <= 9), bid(2, Ps))
    put(r & (H <= 10), NT1)
    put(r & (H <= 12), NT2)
    put(r, NT3)
    r = resp & (P >= bid(2, 1))  # weak twos and preempts
    put(r & (H >= 15) & sup3 & (Ps >= 2) & (Ps <= 3), bid(4, np.clip(Ps, 0, 3)))
    put(r & (H >= 16) & bal, cheapest(4, lb))
    put(r, PASS)

    # ---------------- opener's rebid (opponents silent) ----------------
    M, Ms = f["my_first"], (f["my_first"] - 3) % 5
    R = f["partner_last"]
    Rs, Rl = (R - 3) % 5, (R - 3) // 5 + 1
    rebid = (f["opener_rel"] == 0) & (f["my_bids"] == 1) & (f["partner_bids"] == 1) & ~f["opp_bid"] & (R >= 3)
    r = rebid & (M == NT1)
    put(r & (R == NT2) & (H >= 16), NT3)
    put(r & (R == NT4) & (H >= 17), NT6)
    put(r & (Rl == 3) & (Rs >= 2) & (Rs <= 3) & (H >= 16), bid(4, Rs))
    put(r, PASS)
    put(rebid & (M == NT2), PASS)
    r = rebid & (Rs == Ms) & (Ms < 4)
    put(r & (Rl == 2) & (H >= 19) & (Ms >= 2), bid(4, Ms))
    put(r & (Rl == 2) & (H >= 19), NT3)
    put(r & (Rl == 2) & (H >= 16), bid(3, Ms))
    put(r & (Rl == 3) & (H >= 14) & (Ms >= 2), bid(4, Ms))
    put(r & (Rl == 3) & (H >= 14), NT3)
    put(r, PASS)
    r = rebid & (Rl == 1) & (Rs < 4) & (Rs != Ms)
    fit = length(Rs) >= 4
    put(r & fit & (H >= 19) & (Rs >= 2), bid(4, Rs))
    put(r & fit & (H >= 16), bid(3, Rs))
    put(r & fit, bid(2, Rs))
    put(r & bal & (H <= 14), NT1)
    put(r & bal & (H >= 18) & (H <= 19), NT2)
    put(r & (length(Ms) >= 6) & (H >= 16), bid(3, Ms))
    put(r & (length(Ms) >= 6), bid(2, Ms))
    put(r & (Rs < 3) & (S >= 4) & (Ms != 3), bid(1, 3))
    put(r & (length(Ms) >= 5), bid(2, Ms))
    put(r, NT1)
    r = rebid & (Rs == 4)
    put(r & (Rl == 1) & (length(Ms) >= 6), bid(2, Ms))
    put(r & (Rl == 1) & (H >= 19), NT3)
    put(r & (Rl == 1) & (H >= 17), NT2)
    put(r & (Rl == 2) & (H >= 14), NT3)
    put(rebid, PASS)

    # ---------------- competition ----------------
    opp_strain = lb % 5
    x_ok = mask[:, X]
    ovc = (f["opener_rel"] % 2 == 1) & (f["my_first"] < 0) & (f["partner_first"] < 0) & (f["partner_last"] != X)
    longest = 3 - np.argmax(L[:, ::-1], axis=1)  # ties -> higher suit
    c = cheapest(longest, lb)
    clevel = (c - 3) // 5 + 1
    put(ovc & bal & (H >= 15) & (H <= 18) & (cheapest(4, lb) == NT1), NT1)
    put(ovc & (length(longest) >= 5) & (longest != opp_strain) & (H >= 8) & (H <= 16) & (clevel == 1), c)
    put(ovc & (length(longest) >= 5) & (longest != opp_strain) & (H >= 11) & (H <= 16) & (clevel == 2), c)
    put(ovc & x_ok & (H >= 12) & (opp_strain < 4) & (length(opp_strain) <= 2), X)
    put(ovc & x_ok & (H >= 17), X)
    put(ovc, PASS)

    adv = (f["opener_rel"] % 2 == 1) & (f["my_first"] < 0) & ((f["partner_first"] >= 0) | (f["partner_last"] == X))
    l2 = L.copy()
    l2[ar, np.clip(opp_strain, 0, 3)] = np.where(opp_strain < 4, -1, l2[ar, np.clip(opp_strain, 0, 3)])
    best = 3 - np.argmax(l2[:, ::-1], axis=1)
    cb = cheapest(best, lb)
    r = adv & (f["partner_last"] == X) & (f["partner_first"] < 0)
    put(r & (H >= 10), jump(cb))
    put(r, cb)
    ps = (f["partner_first"] - 3) % 5
    r = adv & (f["partner_first"] >= 0) & (ps < 4) & (length(ps) >= 3)
    put(r & (H >= 13) & (ps >= 2), bid(4, ps))
    put(r & (H >= 10), jump(cheapest(ps, lb)))
    put(r & (H >= 6), cheapest(ps, lb))

    put(np.ones(len(ar), bool), PASS)
    legal = mask[ar, act]
    return np.where(legal, act, PASS).astype(np.int32)


class RuleBot(Policy):
    name = "rule-bot"

    def act(self, s, obs, mask, key):
        return mx.array(decide(auction_features(s), np.array(mask)))
