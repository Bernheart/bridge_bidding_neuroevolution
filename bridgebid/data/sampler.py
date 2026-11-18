"""Deal sources: packed DD data kept on the GPU (32 bytes per deal), decoded on demand."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from bridgebid.cards import PERM_OS_TO_PGX
from bridgebid.data.convert import PROCESSED_DIR, board_conditions
from bridgebid.env import core

_KEY_SHIFTS = mx.arange(24, -1, -2).astype(mx.int32)
_VAL_SHIFTS = mx.arange(16, -1, -4).astype(mx.int32)
_PERM = mx.array(PERM_OS_TO_PGX)


def decode(keys: mx.array, vals: mx.array) -> tuple[mx.array, mx.array]:
    """Packed rows -> hands [n,4,52] bool (OpenSpiel card order), tricks [n,20] int32."""
    n = keys.shape[0]
    owner_pgx = ((keys[:, :, None] >> _KEY_SHIFTS) & 3).reshape(n, 52)
    owner = owner_pgx[:, _PERM]
    hands = owner[:, None, :] == mx.arange(4)[None, :, None]
    tricks = ((vals[:, :, None] >> _VAL_SHIFTS) & 15).reshape(n, 20)
    return hands, tricks


class DealBank:
    """A split of the dataset. Training: random deals with random dealer/vul. Eval: fixed boards."""

    def __init__(self, split: str = "train", data_dir: Path = PROCESSED_DIR, limit: int | None = None):
        keys = np.load(data_dir / f"{split}_keys.npy", mmap_mode="r")
        vals = np.load(data_dir / f"{split}_vals.npy", mmap_mode="r")
        if limit is not None:
            keys, vals = keys[:limit], vals[:limit]
        self.split = split
        self.keys = mx.array(np.ascontiguousarray(keys))
        self.vals = mx.array(np.ascontiguousarray(vals))
        self.size = self.keys.shape[0]
        dealer_path = data_dir / f"{split}_dealer.npy"
        if dealer_path.exists():
            self.dealer = mx.array(np.load(dealer_path)[: self.size].astype(np.int32))
            self.vul = mx.array(np.load(data_dir / f"{split}_vul.npy")[: self.size])
        else:
            d, v = board_conditions(self.size)
            self.dealer, self.vul = mx.array(d.astype(np.int32)), mx.array(v)

    def boards(self, idx: mx.array, random_conditions: bool = False, key: mx.array | None = None):
        """(hands, tricks, dealer, vul) for deal indices `idx`."""
        hands, tricks = decode(self.keys[idx], self.vals[idx])
        if random_conditions:
            k1, k2 = mx.random.split(key)
            n = idx.shape[0]
            dealer = mx.random.randint(0, 4, (n,), key=k1)
            vul = mx.random.bernoulli(0.5, (n, 2), key=k2)
        else:
            dealer, vul = self.dealer[idx], self.vul[idx]
        return hands, tricks, dealer, vul

    def sample_state(self, n: int, key: mx.array) -> core.State:
        """n random deals, random dealer and vulnerability (training)."""
        k1, k2 = mx.random.split(key)
        idx = mx.random.randint(0, self.size, (n,), key=k1)
        return core.reset(*self.boards(idx, random_conditions=True, key=k2))

    def fixed_state(self, start: int, n: int) -> core.State:
        """Boards start..start+n with their fixed dealer/vulnerability (evaluation)."""
        idx = mx.arange(start, start + n)
        return core.reset(*self.boards(idx))
