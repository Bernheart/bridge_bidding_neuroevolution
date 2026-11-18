"""Live cross-checks against Pgx and endplay. Run with:  uv run --group oracle pytest -m oracle

ORACLE_GAMES (env var, default 20,000) sets how many random Pgx games are compared call by call.
"""

import os

import mlx.core as mx
import numpy as np
import pytest

from bridgebid.cards import decode_owners, decode_tricks, owners_to_pbn
from bridgebid.env import core
from bridgebid.scoring import contract_score

pytestmark = pytest.mark.oracle


def test_env_matches_pgx_live():
    from bridgebid.env.pgx_oracle import play_pgx

    n = int(os.environ.get("ORACLE_GAMES", 20_000))
    d = "data/processed"
    keys, vals = np.load(f"{d}/train_keys.npy", mmap_mode="r")[:n], np.load(f"{d}/train_vals.npy", mmap_mode="r")[:n]
    keys, vals = np.ascontiguousarray(keys), np.ascontiguousarray(vals)
    rng = np.random.default_rng(7)
    dealer = rng.integers(0, 4, n).astype(np.int32)
    vul = rng.random((n, 2)) < 0.5
    for start in range(0, n, 5000):
        sl = slice(start, start + 5000)
        g = play_pgx(keys[sl], vals[sl], dealer[sl], vul[sl], seed=start)
        owners, tricks = decode_owners(keys[sl]), decode_tricks(vals[sl])
        hands = owners[:, None, :] == np.arange(4)[None, :, None]
        s = core.reset(mx.array(hands), mx.array(tricks.astype(np.int32)), mx.array(dealer[sl]),
                       mx.array(vul[sl]))
        step = mx.compile(core.step)
        for t in range(g["actions"].shape[1]):
            live = g["actions"][:, t] >= 0
            assert (np.array(s["done"]) == ~live).all()
            assert (np.array(core.current_seat(s))[live] == g["seats"][live, t]).all()
            assert (np.array(core.legal_mask(s))[live] == g["masks"][live, t]).all()
            assert (np.array(core.observe(s)).astype(bool)[live] == g["obs"][live, t]).all()
            s = step(s, mx.array(np.maximum(g["actions"][:, t], 0)))
        assert (np.array(s["score_ns"]) == g["score_ns"]).all()


def test_scoring_matches_pgx_and_endplay():
    import jax
    import jax.numpy as jnp
    from endplay.types import Contract, Denom, Penalty, Player, Vul
    from pgx.bridge_bidding import _calc_score

    calc = jax.jit(jax.vmap(_calc_score))
    lv, st, vu, x, xx, tr = np.meshgrid(np.arange(1, 8), np.arange(5), [0, 1], [0, 1], [0, 1], np.arange(14),
                                        indexing="ij")
    lv, st, vu, x, xx, tr = (a.ravel() for a in (lv, st, vu, x, xx, tr))
    ours = contract_score(np, lv, st, vu, x & (1 - xx), xx, tr)
    pgx = np.asarray(calc(jnp.asarray(st), jnp.asarray(lv), jnp.asarray(vu.astype(bool)),
                          jnp.asarray((x & (1 - xx)).astype(bool)), jnp.asarray(xx.astype(bool)), jnp.asarray(tr)))
    assert (ours == pgx).all()
    denoms = [Denom.clubs, Denom.diamonds, Denom.hearts, Denom.spades, Denom.nt]
    for i in range(len(lv)):
        pen = Penalty.redoubled if xx[i] else (Penalty.doubled if x[i] else Penalty.passed)
        c = Contract(level=int(lv[i]), denom=denoms[st[i]], declarer=Player.north, penalty=pen,
                     result=int(tr[i]) - int(lv[i]) - 6)
        assert c.score(Vul.ns if vu[i] else Vul.none) == ours[i], (c, vu[i])


def test_dd_tables_match_endplay():
    from endplay.dds import calc_dd_table
    from endplay.types import Deal, Denom, Player

    k, v = np.load("data/processed/eval_keys.npy")[:300], np.load("data/processed/eval_vals.npy")[:300]
    owners, tricks = decode_owners(k), decode_tricks(v)
    denoms = [Denom.clubs, Denom.diamonds, Denom.hearts, Denom.spades, Denom.nt]
    players = [Player.north, Player.east, Player.south, Player.west]
    for i in range(len(k)):
        tab = calc_dd_table(Deal(owners_to_pbn(owners[i])))
        ref = [tab[dn, p] for p in players for dn in denoms]
        assert tricks[i].tolist() == ref
