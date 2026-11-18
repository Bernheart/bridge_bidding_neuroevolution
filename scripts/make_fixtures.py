"""Record golden games from Pgx so the default test suite can check the env without JAX.

Run:  uv run --group oracle python scripts/make_fixtures.py
"""

from pathlib import Path

import numpy as np

from bridgebid.env.pgx_oracle import play_pgx

N_GAMES = 1000
OUT = Path("tests/fixtures/pgx_games.npz")


def main() -> None:
    d = Path("data/processed")
    # use validation deals but vary dealer/vulnerability randomly to cover all combinations
    keys = np.load(d / "valid_keys.npy")[:N_GAMES]
    vals = np.load(d / "valid_vals.npy")[:N_GAMES]
    rng = np.random.default_rng(1234)
    dealer = rng.integers(0, 4, N_GAMES).astype(np.int32)
    vul = rng.random((N_GAMES, 2)) < 0.5
    g = play_pgx(keys, vals, dealer, vul, seed=1234)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT, keys=keys, vals=vals, dealer=dealer, vul=vul, actions=g["actions"], score_ns=g["score_ns"],
        seats=g["seats"].astype(np.int8), masks=np.packbits(g["masks"], axis=-1),
        obs=np.packbits(g["obs"], axis=-1),
    )
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB, {N_GAMES} games, "
          f"{(g['actions'] >= 0).sum()} calls, max length {g['actions'].shape[1]})")


if __name__ == "__main__":
    main()
