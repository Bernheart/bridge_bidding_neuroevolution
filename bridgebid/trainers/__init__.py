"""Trainer entry points."""

from __future__ import annotations

from pathlib import Path

from bridgebid.config import RunConfig, load_config


def _trainer_cls(name: str):
    if name == "ppo":
        from bridgebid.trainers.ppo import PPOTrainer

        return PPOTrainer
    if name == "es":
        from bridgebid.trainers.es import ESTrainer

        return ESTrainer
    raise ValueError(f"unknown trainer {name!r}")


def run_training(cfg: RunConfig, hours: float | None = None) -> None:
    _trainer_cls(cfg.trainer)(cfg).train(max_hours=hours)


def resume_training(run_dir: str | Path, overrides: list[str] | None = None, hours: float | None = None) -> None:
    run_dir = Path(run_dir)
    cfg = load_config(str(run_dir / "config.yaml"), overrides)
    _trainer_cls(cfg.trainer)(cfg, run_dir=run_dir, resume=True).train(max_hours=hours)
