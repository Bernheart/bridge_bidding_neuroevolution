"""Download the pre-computed double-dummy dataset used by Pgx (HF sotetsuk/dds_dataset, Apache-2.0).

Each file is a numpy array of shape (2, N, 4) int32 = (keys, values); see bridgebid/cards.py for the packing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

REPO_ID = "sotetsuk/dds_dataset"
FILES = {
    "dds_results_2.5M.npy": "train",
    "dds_results_10M.npy": "train",
    "dds_results_500K.npy": "test",
}
RAW_DIR = Path("data/raw")


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download(raw_dir: Path = RAW_DIR) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    raw_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in FILES:
        path = raw_dir / name
        if not path.exists():
            print(f"downloading {name} ...")
            hf_hub_download(REPO_ID, name, repo_type="dataset", local_dir=raw_dir)
        out[name] = path
        print(f"{name}: {path.stat().st_size / 1e6:.0f} MB  sha256={sha256(path)[:16]}…")
    return out


if __name__ == "__main__":
    download()
