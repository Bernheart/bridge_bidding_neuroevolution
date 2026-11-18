"""Replay 1,000 golden Pgx games (tests/fixtures/pgx_games.npz) through the MLX env. No JAX needed.

Every ply must give the same seat to act, legal mask and observation; every game the same NS score.
Regenerate with: uv run --group oracle python scripts/make_fixtures.py
"""

from pathlib import Path

import mlx.core as mx
import numpy as np

from bridgebid.cards import decode_owners, decode_tricks
from bridgebid.env import core

FIX = Path(__file__).parent / "fixtures" / "pgx_games.npz"


def test_mlx_env_replays_pgx_games():
    f = np.load(FIX)
    actions, seats = f["actions"], f["seats"]
    masks = np.unpackbits(f["masks"], axis=-1, count=38).astype(bool)
    obs = np.unpackbits(f["obs"], axis=-1, count=480).astype(bool)
    owners, tricks = decode_owners(f["keys"]), decode_tricks(f["vals"])
    hands = owners[:, None, :] == np.arange(4)[None, :, None]
    s = core.reset(mx.array(hands), mx.array(tricks.astype(np.int32)), mx.array(f["dealer"]), mx.array(f["vul"]))
    step = mx.compile(core.step)
    for t in range(actions.shape[1]):
        live = actions[:, t] >= 0
        assert (np.array(s["done"]) == ~live).all(), f"ply {t}: terminal flags differ"
        assert (np.array(core.current_seat(s))[live] == seats[live, t]).all(), f"ply {t}: seat to act differs"
        assert (np.array(core.legal_mask(s))[live] == masks[live, t]).all(), f"ply {t}: legal masks differ"
        assert (np.array(core.observe(s)).astype(bool)[live] == obs[live, t]).all(), f"ply {t}: obs differ"
        s = step(s, mx.array(np.maximum(actions[:, t], 0)))
    assert np.array(s["done"]).all()
    assert (np.array(s["score_ns"]) == f["score_ns"]).all(), "final scores differ"
