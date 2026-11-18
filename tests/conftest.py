from pathlib import Path

import pytest

DATA = Path("data/processed")


@pytest.fixture(scope="session")
def eval_arrays():
    """First 2,000 eval deals as numpy arrays (skips if the dataset has not been downloaded)."""
    import numpy as np

    if not (DATA / "eval_keys.npy").exists():
        pytest.skip("dataset missing: run `uv run bridgebid data`")
    return {name: np.load(DATA / f"eval_{name}.npy")[:2000] for name in ("keys", "vals", "dealer", "vul")}
