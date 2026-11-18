"""Board viewer: board data, the HTML page and the bidding API, on synthetic deals (no dataset needed)."""

import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from bridgebid.cards import encode_owners, encode_tricks
from bridgebid.data.convert import board_conditions
from bridgebid.data.sampler import DealBank
from bridgebid.eval.duplicate import match
from bridgebid.models.actor_critic import PolicyNet
from bridgebid.policies.base import AlwaysPass, NeuralPolicy, RandomLegal
from bridgebid.policies.rule_bot import RuleBot
from bridgebid.ui.boards import build_board_data, default_out, render_html
from bridgebid.ui.server import BidEngine, BidError, make_server

N_DEALS = 24


@pytest.fixture(scope="module")
def bank(tmp_path_factory):
    d = tmp_path_factory.mktemp("deals")
    rng = np.random.default_rng(7)
    owners = np.stack([rng.permutation(np.repeat(np.arange(4), 13)) for _ in range(N_DEALS)]).astype(np.int8)
    tricks = rng.integers(0, 14, (N_DEALS, 20))
    dealer, vul = board_conditions(N_DEALS)
    np.save(d / "test_keys.npy", encode_owners(owners))
    np.save(d / "test_vals.npy", encode_tricks(tricks))
    np.save(d / "test_dealer.npy", dealer)
    np.save(d / "test_vul.npy", vul)
    return DealBank("test", data_dir=d)


@pytest.fixture(scope="module")
def model():
    import mlx.core as mx

    mx.random.seed(0)  # weights only; play itself uses explicit keys
    return NeuralPolicy(PolicyNet(width=64, depth=2), name="tiny")


@pytest.fixture(scope="module")
def opponents():
    return [RandomLegal(), AlwaysPass(), RuleBot()]


@pytest.fixture(scope="module")
def data(model, opponents, bank):
    return build_board_data(model, opponents, bank, start=4, n=12, seed=3)


def test_board_data_shape(data, opponents):
    assert [b["n"] for b in data["boards"]] == list(range(5, 17))
    for b in data["boards"]:
        assert len(b["hands"]) == 4
        assert [sum(len(s) for s in h) for h in b["hands"]] == [13] * 4
        assert len(b["dd"]) == 20 and len(b["hcp"]) == 4 and sum(b["hcp"]) == 40
        assert len(b["pairs"]) == len(opponents)
        for p in b["pairs"]:
            assert len(p["tables"]) == 2
            for k, t in enumerate(p["tables"]):
                assert len(t["probs"]) == len(t["calls"])
                for j, pr in enumerate(t["probs"]):
                    model_turn = (b["dealer"] + j) % 2 == k
                    assert (pr is not None) == model_turn
                    if pr:  # greedy model: its call is its top call
                        assert pr[0][0] == t["calls"][j] and len(pr) <= 3
                c = t["contract"]
                if c["level"]:
                    assert c["tricks"] == b["dd"][5 * c["declarer"] + c["strain"]]


def test_imps_equal_duplicate_match(data, model, opponents, bank):
    for i, opp in enumerate(opponents):
        r = match(model, opp, bank, n_boards=12, start=4, seed=3)
        assert [b["pairs"][i]["imps"] for b in data["boards"]] == r.imps.astype(int).tolist()
        assert [b["pairs"][i]["tables"][0]["contract"]["score"] for b in data["boards"]] == r.table1.score_ns.tolist()
        assert [b["pairs"][i]["tables"][1]["calls"] for b in data["boards"]] == [
            r.table2.calls[k, : r.table2.n_calls[k]].tolist() for k in range(12)]


def test_html_is_self_contained(data):
    page = render_html(data)
    assert "/*__BOARD_DATA__*/" not in page
    assert '"serve":false' in page and "src=" not in page and "<link" not in page
    payload = page.split("const D = ", 1)[1].split(";\n", 1)[0]
    assert json.loads(payload)["model"] == "tiny"


def test_default_out():
    assert str(default_out("runs/r1/ckpt/policy.safetensors", "r1")) == "runs/r1/boards.html"
    assert str(default_out("brl:sl", "brl:sl")) == "runs/viewer/brl-sl.html"


def test_bidding_plays_to_the_end_with_legal_calls(data, model, opponents, bank):
    eng = BidEngine(model, opponents, bank, data)
    board, seat, opp = 0, 2, 2
    out = eng.bid(board, seat, opp, [])
    dealer = data["boards"][board]["dealer"]
    for _ in range(80):
        if out["done"]:
            break
        assert (dealer + len(out["calls"])) % 4 == seat  # stops at the user's turn
        assert len(out["probs"]) == len(out["calls"]) and out["hint"]
        legal = [a for a, ok in enumerate(out["legal"]) if ok]
        call = max(a for a in legal if a < 10) if len(out["calls"]) < 8 else 0  # a few bids, then pass
        out = eng.bid(board, seat, opp, out["calls"] + [call])
    assert out["done"]
    r = out["result"]
    assert r["model_table"] == 0
    model_score = data["boards"][board]["pairs"][opp]["tables"][0]["contract"]["score"]
    from bridgebid.scoring import imps

    assert r["imps"] == int(imps(np, r["contract"]["score"] - model_score))
    with pytest.raises(BidError):
        eng.bid(board, seat, opp, out["calls"] + [0])  # auction over
    first = eng.bid(board, seat, opp, [])
    illegal = next(a for a, ok in enumerate(first["legal"]) if not ok) if not all(first["legal"]) else None
    if illegal is not None:
        with pytest.raises(BidError):
            eng.bid(board, seat, opp, first["calls"] + [illegal])


def test_bidding_rejects_made_up_bot_calls(data, model, opponents, bank):
    eng = BidEngine(model, opponents, bank, data)
    for opp in range(len(opponents)):  # incl. the random bot: replay must reproduce its calls
        out = eng.bid(3, 1, opp, [])
        for _ in range(6):
            if out["done"]:
                break
            out = eng.bid(3, 1, opp, out["calls"] + [0])
        dealer = data["boards"][3]["dealer"]
        bot_j = next((j for j in range(len(out["calls"])) if (dealer + j) % 4 != 1), None)
        if bot_j is None:
            continue
        legal_other = 0 if out["calls"][bot_j] != 0 else 3
        fake = out["calls"][:bot_j] + [legal_other]
        with pytest.raises(BidError):
            eng.bid(3, 1, opp, fake)


def test_server_localhost(data, model, opponents, bank):
    srv = make_server(model, opponents, bank, data, port=0)
    assert srv.server_address[0] == "127.0.0.1"
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        page = urllib.request.urlopen(base + "/").read().decode()
        assert '"serve":true' in page
        req = urllib.request.Request(base + "/api/bid", data=json.dumps(
            {"board": 1, "seat": 0, "opp": 1, "calls": []}).encode(), headers={"Content-Type": "application/json"})
        out = json.loads(urllib.request.urlopen(req).read())
        assert "calls" in out and "done" in out
        body = b'{"board": 99, "seat": 0, "opp": 0}'
        json_ct = {"Content-Type": "application/json"}
        port = srv.server_address[1]
        for headers, code in [
            (json_ct, 400),  # bad board
            ({}, 415),  # not JSON: a cross-site "simple" POST
            ({**json_ct, "Origin": "https://evil.example"}, 403),  # another site
            ({**json_ct, "Host": f"evil.example:{port}"}, 403),  # DNS rebinding
            ({**json_ct, "Origin": f"http://localhost:{port}", "Host": f"localhost:{port}"}, 400),  # allowed
        ]:
            with pytest.raises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(urllib.request.Request(base + "/api/bid", data=body, headers=headers))
            assert e.value.code == code, headers
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(urllib.request.Request(base + "/", headers={"Host": "evil.example"}))
        assert e.value.code == 403
    finally:
        srv.shutdown()
        srv.server_close()
