"""Terminal viewer: deal diagram, auction grid, final contract and (for neural policies) the top-3 calls."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.text import Text

from bridgebid.cards import hcp
from bridgebid.constants import RANKS, SEATS, action_to_str
from bridgebid.data.sampler import DealBank
from bridgebid.env import core
from bridgebid.policies.base import NeuralPolicy, Policy

SUIT_STYLE = {"♠": "bold blue", "♥": "bold red", "♦": "bold yellow", "♣": "bold green", "NT": "bold"}
SEAT_NAMES = ("North", "East", "South", "West")


def call_text(action: int) -> Text:
    s = action_to_str(action)
    if action < 3:
        return Text(s, style="dim" if action == 0 else "bold magenta")
    sym = s[1:]
    return Text(s[0]) + Text(sym, style=SUIT_STYLE.get(sym, ""))


def hand_lines(hand: np.ndarray) -> list[Text]:
    lines = []
    for suit, sym in zip((3, 2, 1, 0), ("♠", "♥", "♦", "♣"), strict=True):
        ranks = "".join(RANKS[r] for r in range(12, -1, -1) if hand[suit + 4 * r]).replace("T", "10")
        lines.append(Text(sym + " ", style=SUIT_STYLE[sym]) + Text(ranks or "—"))
    return lines


def deal_diagram(hands: np.ndarray) -> list[Text]:
    pts = hcp(hands)
    n, e, s, w = (hand_lines(hands[i]) for i in range(4))
    pad = " " * 18
    out = [Text(pad + f"North ({pts[0]})")] + [Text(pad) + ln for ln in n]
    out.append(Text(f"{'West (' + str(pts[3]) + ')':<36}East ({pts[1]})"))
    for wl, el in zip(w, e, strict=True):
        out.append(wl + Text(" " * max(1, 36 - len(wl.plain))) + el)
    out += [Text(pad + f"South ({pts[2]})")] + [Text(pad) + ln for ln in s]
    return out


def show_boards(ns: Policy, ew: Policy, bank: DealBank, start: int = 0, n: int = 5, top: int = 3,
                console: Console | None = None) -> None:
    console = Console() if console is None else console
    st = bank.fixed_state(start, n)
    hands, dealer, vul = np.array(st["hands"]), np.array(st["dealer"]), np.array(st["vul"])
    tricks = np.array(st["tricks"])
    explain: list[list[str]] = [[] for _ in range(n)]
    key = mx.random.key(0)
    while not mx.all(st["done"]).item():
        key, k1, k2 = mx.random.split(key, 3)
        obs, mask = core.observe(st), core.legal_mask(st)
        cur = np.array(core.current_seat(st))
        ns_turn = mx.array(cur % 2 == 0)
        act = mx.where(ns_turn, ns.act(st, obs, mask, k1), ew.act(st, obs, mask, k2))
        done = np.array(st["done"])
        for pol, side in ((ns, 0), (ew, 1)):
            if isinstance(pol, NeuralPolicy):
                probs = np.array(mx.softmax(pol.logits(obs, mask), axis=-1))
                for i in np.flatnonzero((cur % 2 == side) & ~done):
                    best = np.argsort(-probs[i])[:top]
                    explain[i].append(f"{SEATS[cur[i]]}: " + "  ".join(
                        f"{action_to_str(int(b))} {probs[i, b]:.0%}" for b in best if probs[i, b] >= 0.01))
        st = core.step(st, act)
    info = {k: np.array(v) for k, v in core.contract_info(st).items()}
    calls, n_calls, score = np.array(st["calls"]), np.array(st["turn"]), np.array(st["score_ns"])

    for i in range(n):
        v = {(False, False): "None", (True, False): "NS", (False, True): "EW", (True, True): "Both"}
        console.rule(f"Board {start + i + 1} · Dealer {SEAT_NAMES[dealer[i]]} · Vul {v[tuple(vul[i])]} · "
                     f"NS = {ns.name}, EW = {ew.name}")
        for line in deal_diagram(hands[i]):
            console.print(line)
        grid = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
        order = [3, 0, 1, 2]  # W N E S
        for seat in order:
            grid.add_column(SEAT_NAMES[seat])
        row: list[Text] = [Text("")] * order.index(dealer[i])
        for c in calls[i, : n_calls[i]]:
            row.append(call_text(int(c)))
            if len(row) == 4:
                grid.add_row(*row)
                row = []
        if row:
            grid.add_row(*row, *[Text("")] * (4 - len(row)))
        console.print(grid)
        if info["level"][i] == 0:
            console.print("Passed out.  NS 0")
        else:
            lv, sn, d = info["level"][i], info["strain"][i], info["declarer"][i]
            dbl = "XX" if info["redoubled"][i] else ("X" if info["doubled"][i] else "")
            t = tricks[i, 5 * d + sn]
            res = int(t) - (lv + 6)
            console.print(Text("Contract ") + call_text(int(3 + 5 * (lv - 1) + sn)) + Text(
                f"{dbl} by {SEAT_NAMES[d]}, double-dummy {t} tricks ({'=' if res == 0 else f'{res:+d}'}), "
                f"NS {score[i]:+d}"))
        if explain[i]:
            console.print(Text("model's top calls: ", style="dim") + Text(" | ".join(explain[i]), style="dim"))
