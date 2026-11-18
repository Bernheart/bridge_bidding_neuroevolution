"""Run directory: config, metadata, JSONL metrics, TensorBoard, checkpoints.

runs/<YYYYmmdd-HHMMSS>_<trainer>_<name>/
    config.yaml   meta.json   diff.patch (if the git tree was dirty)
    metrics.jsonl tb/         ckpt/        snapshots/   eval/
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path

import mlx.core as mx

from bridgebid.config import RunConfig, to_yaml


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


class Run:
    def __init__(self, cfg: RunConfig, run_dir: str | Path | None = None):
        self.cfg = cfg
        if run_dir is None:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            run_dir = Path("runs") / f"{stamp}_{cfg.trainer}_{cfg.name}{'_smoke' if cfg.smoke else ''}"
            run_dir.mkdir(parents=True, exist_ok=False)
        self.dir = Path(run_dir)
        for sub in ("ckpt", "snapshots", "eval"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        (self.dir / "config.yaml").write_text(to_yaml(cfg))
        meta_path = self.dir / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"sessions": []}
        dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
        session = {"started": datetime.now().isoformat(timespec="seconds"), "git_sha": _git("rev-parse", "HEAD"),
                   "git_dirty": dirty, "mlx": mx.__version__, "python": platform.python_version(),
                   "device": mx.device_info().get("device_name", "?")}
        if dirty:
            (self.dir / f"diff_{len(meta['sessions'])}.patch").write_text(_git("diff"))
        meta["sessions"].append(session)
        meta_path.write_text(json.dumps(meta, indent=2))
        self._metrics = open(self.dir / "metrics.jsonl", "a")
        self._tb = None
        self.t0 = time.time()

    @property
    def tb(self):
        if self._tb is None:
            from tensorboardX import SummaryWriter

            self._tb = SummaryWriter(str(self.dir / "tb"))
        return self._tb

    def log(self, step: int, metrics: dict, prefix: str = "") -> None:
        flat = {f"{prefix}{k}": v for k, v in metrics.items()}
        self._metrics.write(json.dumps({"step": step, "time": round(time.time() - self.t0, 1), **flat}) + "\n")
        self._metrics.flush()
        for k, v in flat.items():
            if isinstance(v, int | float):
                self.tb.add_scalar(k, v, step)
        self.tb.flush()

    def close(self) -> None:
        self._metrics.close()
        if self._tb is not None:
            self._tb.close()
