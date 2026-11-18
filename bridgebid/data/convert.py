"""Convert the raw DD dataset into training / validation / evaluation splits with integrity checks.

Output (data/processed/):
    train_keys.npy, train_vals.npy                    12.5M deals (2.5M + 10M files), minus any eval overlap
    valid_*.npy                                        100K deals from the test file [100K:200K]
    eval_*.npy                                         100K deals from the test file [0:100K];
                                                       the first 10K are the standard quick-eval set
    {valid,eval}_dealer.npy, {valid,eval}_vul.npy     fixed boards: board i uses the standard 16-board cycle
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bridgebid.cards import decode_owners, decode_tricks
from bridgebid.data.download import RAW_DIR

PROCESSED_DIR = Path("data/processed")

# Standard duplicate board cycle (board 1..16): dealer = (board-1) % 4, vulnerability below.
_VUL_CYCLE = ["-", "NS", "EW", "B", "NS", "EW", "B", "-", "EW", "B", "-", "NS", "B", "-", "NS", "EW"]


def board_conditions(n: int) -> tuple[np.ndarray, np.ndarray]:
    """dealer [n] int8 and vul [n, 2] bool (NS, EW) for boards 1..n."""
    b = np.arange(n)
    dealer = (b % 4).astype(np.int8)
    tag = np.array([_VUL_CYCLE[i % 16] for i in range(16)])[b % 16]
    vul = np.stack([np.isin(tag, ["NS", "B"]), np.isin(tag, ["EW", "B"])], axis=1)
    return dealer, vul


def _load_raw(name: str) -> tuple[np.ndarray, np.ndarray]:
    arr = np.load(RAW_DIR / name, allow_pickle=False)
    assert arr.shape[0] == 2 and arr.shape[2] == 4 and arr.dtype == np.int32, arr.shape
    return arr[0], arr[1]


def check_deals(keys: np.ndarray, vals: np.ndarray, chunk: int = 1_000_000) -> None:
    for s in range(0, len(keys), chunk):
        owners = decode_owners(keys[s : s + chunk])
        counts = np.stack([(owners == seat).sum(1) for seat in range(4)], axis=1)
        assert (counts == 13).all(), "a seat does not hold 13 cards"
        tricks = decode_tricks(vals[s : s + chunk])
        assert tricks.min() >= 0 and tricks.max() <= 13, "trick count out of range"


def _row_view(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()


def convert(out_dir: Path = PROCESSED_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    k1, v1 = _load_raw("dds_results_2.5M.npy")
    k2, v2 = _load_raw("dds_results_10M.npy")
    kt, vt = _load_raw("dds_results_500K.npy")
    train_k, train_v = np.concatenate([k1, k2]), np.concatenate([v1, v2])

    print("checking card counts and trick ranges ...")
    check_deals(train_k, train_v)
    check_deals(kt, vt)

    held_out = kt[:200_000]
    overlap = np.isin(_row_view(train_k), _row_view(held_out))
    n_dup_train = len(train_k) - len(np.unique(_row_view(train_k)))
    print(f"train deals: {len(train_k):,}  duplicates inside train: {n_dup_train:,}  "
          f"overlap with valid/eval: {overlap.sum():,} (removed)")
    train_k, train_v = train_k[~overlap], train_v[~overlap]
    assert len(np.unique(_row_view(held_out))) == len(held_out), "duplicate deals in eval/valid"

    np.save(out_dir / "train_keys.npy", train_k)
    np.save(out_dir / "train_vals.npy", train_v)
    for split, sl in (("eval", slice(0, 100_000)), ("valid", slice(100_000, 200_000))):
        dealer, vul = board_conditions(sl.stop - sl.start)
        np.save(out_dir / f"{split}_keys.npy", kt[sl])
        np.save(out_dir / f"{split}_vals.npy", vt[sl])
        np.save(out_dir / f"{split}_dealer.npy", dealer)
        np.save(out_dir / f"{split}_vul.npy", vul)
    print(f"wrote {out_dir}/ (train {len(train_k):,}, valid 100,000, eval 100,000)")


if __name__ == "__main__":
    convert()
