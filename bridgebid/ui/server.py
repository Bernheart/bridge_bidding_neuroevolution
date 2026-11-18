"""Localhost server for `bridgebid show --serve`: the viewer page plus a "Bid this board" API.

The API is stateless: each request sends the board, the user's seat, the opponent and the calls so far. The
server replays them on a one-deal state (rejecting illegal calls), lets the bots bid until it is the user's
turn or the auction ends, and returns the auction, the bots' top calls, the legal calls and, at the end, the
result compared with the model's own auction on that board.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import mlx.core as mx
import numpy as np

from bridgebid.constants import NUM_ACTIONS
from bridgebid.data.sampler import DealBank
from bridgebid.env import core
from bridgebid.policies.base import Policy
from bridgebid.scoring import imps
from bridgebid.ui.boards import contract_dict, render_html, top_calls, topk_list


class BidError(ValueError):
    pass


class BidEngine:
    """Bidding with the model as partner. Board `b` is index b into `data["boards"]`."""

    def __init__(self, model: Policy, opponents: list[Policy], bank: DealBank, data: dict, seed: int = 0):
        self.model, self.opponents, self.bank, self.data, self.seed = model, opponents, bank, data, seed
        self.lock = threading.Lock()  # MLX work happens one request at a time

    def bid(self, board: int, seat: int, opp: int, calls: list[int]) -> dict:
        if not 0 <= board < len(self.data["boards"]):
            raise BidError(f"no board {board}")
        if seat not in range(4):
            raise BidError(f"bad seat {seat}")
        if not 0 <= opp < len(self.opponents):
            raise BidError(f"no opponent {opp}")
        with self.lock:
            return self._bid(board, seat, opp, [int(c) for c in calls])

    def _bid(self, board: int, seat: int, opp_i: int, calls: list[int]) -> dict:
        b = self.data["boards"][board]
        st = self.bank.fixed_state(b["n"] - 1, 1)
        opp = self.opponents[opp_i]
        key = mx.random.key(self.seed)
        probs: list[list | None] = []

        def bots_for(cur: int) -> Policy:
            return self.model if cur % 2 == seat % 2 else opp

        # replay the given calls (all must be legal); bot turns split the key as when they were first bid
        for c in calls:
            if bool(st["done"].item()):
                raise BidError("the auction is already over")
            cur = int(core.current_seat(st).item())
            mask = core.legal_mask(st)
            if not 0 <= c < NUM_ACTIONS or not bool(mask[0, c].item()):
                raise BidError(f"illegal call {c}")
            if cur == seat:
                probs.append(None)
            else:  # a bot's seat: the call must be the one the bot makes (same key stream as when first bid)
                key, k = mx.random.split(key)
                pol, obs = bots_for(cur), core.observe(st)
                if int(pol.act(st, obs, mask, k).item()) != c:
                    raise BidError(f"call {len(probs) + 1} is not what {pol.name} bids")
                probs.append(self._explain(pol, st, mask, obs))
            st = core.step(st, mx.array([c]))
        # bots bid until it is the user's turn or the auction ends
        while not bool(st["done"].item()) and int(core.current_seat(st).item()) != seat:
            cur = int(core.current_seat(st).item())
            key, k = mx.random.split(key)
            obs, mask = core.observe(st), core.legal_mask(st)
            pol = bots_for(cur)
            act = pol.act(st, obs, mask, k)
            probs.append(self._explain(pol, st, mask, obs))
            calls.append(int(act.item()))
            st = core.step(st, act)
        out: dict = {"calls": calls, "probs": probs, "done": bool(st["done"].item())}
        if out["done"]:
            out["result"] = self._result(st, b, seat, opp_i)
        else:
            obs, mask = core.observe(st), core.legal_mask(st)
            out["legal"] = [bool(x) for x in np.array(mask[0])]
            out["hint"] = self._explain(self.model, st, mask, obs)
        return out

    @staticmethod
    def _explain(pol: Policy, st, mask, obs=None) -> list | None:
        tc = top_calls(pol, core.observe(st) if obs is None else obs, mask)
        if tc is None:
            return None
        idx, p = (np.array(x[0]) for x in tc)
        return topk_list(idx, p)

    def _result(self, st, b: dict, seat: int, opp_i: int) -> dict:
        info = {k: int(v[0].item()) for k, v in core.contract_info(st).items()}
        lv, d, sn = info["level"], info["declarer"], info["strain"]
        t = None if lv == 0 else b["dd"][5 * d + sn]
        score = int(st["score_ns"][0].item())
        # the model's table with the model in the user's partnership's seats against the same opponent
        model_table = b["pairs"][opp_i]["tables"][0 if seat % 2 == 0 else 1]
        diff = score - model_table["contract"]["score"]
        you = int(imps(np, diff if seat % 2 == 0 else -diff))
        return {"contract": contract_dict(lv, sn, d, info["doubled"], info["redoubled"], t, score),
                "model_table": 0 if seat % 2 == 0 else 1, "imps": you}


def make_handler(page: str, engine: BidEngine):
    body = page.encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _local_only(self) -> bool:
            """Refuse requests not addressed to this server by name (DNS rebinding) or sent from another site
            (cross-site POSTs). JSON-only POSTs also force a CORS preflight, which this server never grants."""
            port = self.server.server_address[1]
            local = {f"127.0.0.1:{port}", f"localhost:{port}"}
            origin = self.headers.get("Origin")
            if self.headers.get("Host") not in local or (origin is not None and origin.removeprefix("http://")
                                                         not in local):
                self._send(403, b"forbidden", "text/plain")
                return False
            return True

        def do_GET(self):  # noqa: N802
            if not self._local_only():
                return
            if self.path.split("#")[0].split("?")[0] in ("/", "/index.html"):
                self._send(200, body, "text/html; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):  # noqa: N802
            if not self._local_only():
                return
            if self.path != "/api/bid":
                self._send(404, b"not found", "text/plain")
                return
            if self.headers.get_content_type() != "application/json":
                self._send(415, b"send application/json", "text/plain")
                return
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                out = engine.bid(int(req["board"]), int(req["seat"]), int(req["opp"]), list(req.get("calls", [])))
                self._send(200, json.dumps(out).encode(), "application/json")
            except (BidError, KeyError, TypeError, ValueError) as e:
                self._send(400, json.dumps({"error": str(e)}).encode(), "application/json")

        def log_message(self, fmt, *args):  # keep the terminal quiet
            pass

    return Handler


def make_server(model: Policy, opponents: list[Policy], bank: DealBank, data: dict, port: int = 8765,
                seed: int = 0) -> HTTPServer:
    """An HTTP server bound to 127.0.0.1 only (port 0 picks a free port)."""
    engine = BidEngine(model, opponents, bank, data, seed=seed)
    return HTTPServer(("127.0.0.1", port), make_handler(render_html(data, serve=True), engine))
