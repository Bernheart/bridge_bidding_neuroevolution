"""How a policy bids, from self-play on fixed boards: catches collapse early and shows the learned 'system'."""

from __future__ import annotations

from collections import Counter

import mlx.core as mx
import numpy as np

from bridgebid.cards import hcp
from bridgebid.constants import action_to_str
from bridgebid.data.sampler import DealBank
from bridgebid.eval.duplicate import _record, play
from bridgebid.policies.base import Policy

HCP_BUCKETS = [(0, 5), (6, 10), (11, 12), (13, 15), (16, 17), (18, 19), (20, 37)]
ALARMS = {"pass_out_rate": 0.40, "top_opening_share": 0.80}


def diagnose(policy: Policy, bank: DealBank, n_boards: int = 4000, start: int = 0, seed: int = 0) -> dict:
    st = bank.fixed_state(start, n_boards)
    final = play(st, mx.ones((n_boards,), mx.bool_), policy, policy, mx.random.key(seed))
    rec = _record(final)
    hands = np.array(st["hands"])
    dealer = np.array(st["dealer"])
    tricks = np.array(st["tricks"])
    played = rec.level > 0

    # opening = first non-pass call; system card = dealer's first call by HCP bucket
    first_call = rec.calls[:, 0]
    dealer_hcp = hcp(hands[np.arange(n_boards), dealer])
    openings = [next((int(c) for c in row if c >= 3), -1) for row in rec.calls]
    open_counts = Counter(o for o in openings if o >= 0)
    top_share = max(open_counts.values()) / max(sum(open_counts.values()), 1) if open_counts else 1.0

    made = np.zeros(n_boards, bool)
    d = np.clip(rec.declarer, 0, 3)
    t = tricks[np.arange(n_boards), 5 * d + rec.strain]
    made[played] = t[played] >= rec.level[played] + 6
    game = played & ((rec.level >= 5) | ((rec.level >= 4) & (rec.strain >= 2)) | ((rec.level >= 3) & (rec.strain == 4)))

    card = {}
    for lo, hi in HCP_BUCKETS:
        sel = (dealer_hcp >= lo) & (dealer_hcp <= hi)
        c = Counter(action_to_str(int(a), symbols=False) for a in first_call[sel])
        total = sel.sum()
        card[f"{lo}-{hi}"] = [(k, round(v / total, 3)) for k, v in c.most_common(4)] if total else []

    out = {
        "pass_out_rate": float((~played).mean()),
        "auction_length": float(rec.n_calls.mean()),
        "auction_length_p95": float(np.percentile(rec.n_calls, 95)),
        "doubled_rate": float(rec.doubled[played].mean()) if played.any() else 0.0,
        "redoubled_rate": float(rec.redoubled[played].mean()) if played.any() else 0.0,
        "made_rate": float(made[played].mean()) if played.any() else 0.0,
        "game_or_higher_rate": float(game.mean()),
        "slam_rate": float((rec.level >= 6).mean()),
        "level_hist": np.bincount(rec.level, minlength=8).tolist(),
        "strain_hist": np.bincount(rec.strain[played], minlength=5).tolist(),
        "top_openings": [(action_to_str(k, symbols=False), round(v / max(len(openings), 1), 3))
                         for k, v in open_counts.most_common(6)],
        "top_opening_share": float(top_share),
        "distinct_auctions": len({tuple(r[:n]) for r, n in zip(rec.calls, rec.n_calls, strict=True)}),
        "system_card": card,
    }
    out["alarms"] = [k for k, v in ALARMS.items() if out[k] > v]
    return out


def format_diagnostics(d: dict) -> str:
    lines = [
        f"pass-out {d['pass_out_rate']:.1%} | auction length {d['auction_length']:.1f} "
        f"(p95 {d['auction_length_p95']:.0f}) | distinct auctions {d['distinct_auctions']}",
        f"contracts made (DD) {d['made_rate']:.1%} | game+ {d['game_or_higher_rate']:.1%} | slams "
        f"{d['slam_rate']:.1%} | doubled {d['doubled_rate']:.1%} | redoubled {d['redoubled_rate']:.1%}",
        "top openings: " + ", ".join(f"{k} {v:.0%}" for k, v in d["top_openings"]),
        "dealer's first call by HCP:",
    ]
    for bucket, calls in d["system_card"].items():
        lines.append(f"   {bucket:>6}: " + ", ".join(f"{k} {v:.0%}" for k, v in calls))
    if d["alarms"]:
        lines.append(f"ALARMS: {', '.join(d['alarms'])}")
    return "\n".join(lines)
