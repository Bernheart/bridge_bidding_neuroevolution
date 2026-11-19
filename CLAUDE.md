# bridge_bidding_neuroevolution

<!-- claude-kit:begin id=commands v=0.19.0 sha=96e08d7e -->
## Commands

- Test: `uv run pytest`
- Lint: `uv run ruff check .`
- Format: `uv run ruff format .`
<!-- claude-kit:end id=commands -->

<!-- claude-kit:begin id=done v=0.19.0 sha=b1a60dfd -->
## Definition of done

Before you say a change is done, run `uv run ruff check . && uv run pytest` and fix every failure.
If you could not run it, say so instead of reporting success.
<!-- claude-kit:end id=done -->

## MLX setup

- Apple silicon only: everything runs on the Metal GPU through MLX (no torch, no CUDA). Python is pinned to
  3.12 (`.python-version`); `uv sync` installs the `dev` group (pytest, hypothesis, ruff) by default.
- Data and models are local and gitignored (`/data/`, `/models/`, `/runs/`). Fetch them once with
  `uv run bridgebid data` (DD dataset, ~420 MB, into `data/processed/`) and `uv run bridgebid baselines`
  (brl bots, ~75 MB, into `models/brl/`). Tests that need the dataset skip when it is missing: report skips,
  don't count them as passes.
- MLX is lazy: nothing is computed until `mx.eval(...)`, `.item()` or a numpy conversion. Keep the existing
  pattern of one `mx.eval` per step on the state; don't convert to numpy inside hot loops.
- `bridgebid/env/core.py` keeps state as a dict of fixed-shape arrays so `core.step` can be `mx.compile`d:
  keep shapes static and don't branch in Python on array values there. `bridgebid/env/reference.py` is the
  readable spec it is tested against; a rule change goes in both.
- Compiled functions that close over a model pass its state, e.g. `mx.compile(f, inputs=[model.state])`
  (see `bridgebid/policies/base.py`, `bridgebid/trainers/ppo.py`); otherwise they keep using stale weights.
- Randomness uses explicit keys (`mx.random.key(seed)`, then `mx.random.split`). Resume is bit-exact and
  tested, so don't add unseeded or global randomness to trainers.
- `uv run bridgebid train <config> --smoke` is the 1-minute end-to-end check. Start longer runs (`--hours`)
  only when the user says the Mac is free: it is shared with another project.

## Test markers

`pyproject.toml` sets `addopts = "-m 'not oracle and not slow'"`, so plain `uv run pytest` skips both:

- `slow`: long tests, currently the bit-exact PPO resume. Run with `uv run pytest -m slow` (needs the dataset).
- `oracle`: live cross-checks against Pgx and endplay (`tests/test_oracle.py`). Needs the oracle group:
  `uv sync --group oracle && uv run --group oracle pytest -m oracle`. `ORACLE_GAMES` (default 20,000) sets
  how many random games are compared.
- Without JAX, `tests/test_pgx_fixtures.py` replays 1,000 golden Pgx games from
  `tests/fixtures/pgx_games.npz`. Regenerate them with `uv run --group oracle python scripts/make_fixtures.py`.
- Mark new tests that take more than a few seconds `@pytest.mark.slow`, and anything that imports jax, pgx
  or endplay `@pytest.mark.oracle` (`bridgebid/env/pgx_oracle.py` is for tests only).
