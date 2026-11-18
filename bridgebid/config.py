"""Run configuration: typed dataclasses, loaded from YAML with `key=value` command-line overrides.

The fully resolved config is saved in every run directory, so a run can always be reproduced or resumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from omegaconf import OmegaConf


@dataclass
class PPOConfig:
    num_envs: int = 8192
    rollout_len: int = 32  # trained plies per update (+4 look-ahead plies for bootstrapping)
    epochs: int = 4
    minibatch: int = 16384
    lr: float = 1e-4
    lr_anneal: bool = True
    clip: float = 0.2
    vf_coef: float = 0.5
    vf_clip: float = 0.2
    ent_start: float = 0.01
    ent_end: float = 0.001
    ent_decay_frac: float = 0.3
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    width: int = 512
    depth: int = 4
    critic: str = "privileged"  # privileged (sees all hands + DD table, training only) | shared
    critic_width: int = 512
    critic_depth: int = 3
    reward_scale: float = 1000.0
    total_steps: int = 1_000_000_000  # env decisions (all seats)
    # curriculum: probability that one side is forced to pass (uncontested warm-up)
    passive_until: int = 50_000_000
    passive_anneal: int = 50_000_000
    # fictitious self-play against past snapshots (after the warm-up)
    fsp_frac: float = 0.5
    fsp_every: int = 20
    fsp_pool: int = 50
    fsp_groups: int = 4
    eval_every: int = 25
    eval_boards: int = 4000
    eval_opponents: list[str] = field(default_factory=lambda: ["pass", "rule", "brl:sl"])
    ckpt_every: int = 25


@dataclass
class ESConfig:
    population: int = 512  # even (antithetic pairs)
    boards: int = 512  # boards per generation, the same for every member
    chunk: int = 64  # members evaluated per batched forward
    sigma: float = 0.02
    lr: float = 0.01
    weight_decay: float = 0.005
    width: int = 256
    depth: int = 2
    generations: int = 3000
    warmup_generations: int = 300  # opponents always pass
    mix_centre: float = 0.5  # after warm-up: fraction of boards vs the current centre
    mix_pool: float = 0.25  # vs a past centre
    mix_rule: float = 0.25  # vs the rule bot
    snapshot_every: int = 25
    pool_size: int = 20
    kill_margin: float = 1.0  # after warm-up the centre must beat always-pass by this many IMPs/board
    eval_every: int = 25
    eval_boards: int = 4000
    eval_opponents: list[str] = field(default_factory=lambda: ["pass", "rule", "brl:sl"])
    ckpt_every: int = 25


@dataclass
class RunConfig:
    trainer: str = "ppo"  # ppo | es
    name: str = "run"
    seed: int = 0
    smoke: bool = False
    train_split: str = "train"
    train_limit: int | None = None  # use only the first N training deals (debugging)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    es: ESConfig = field(default_factory=ESConfig)


SMOKE = {
    "ppo": {"num_envs": 1024, "rollout_len": 16, "minibatch": 4096, "total_steps": 1024 * 20 * 15,
            "passive_until": 1024 * 20 * 5, "passive_anneal": 1024 * 20 * 3, "fsp_every": 3, "eval_every": 5,
            "eval_boards": 1000, "ckpt_every": 5, "width": 256, "critic_width": 256},
    "es": {"population": 32, "boards": 64, "chunk": 32, "generations": 12, "warmup_generations": 5,
           "snapshot_every": 2, "eval_every": 4, "eval_boards": 1000, "ckpt_every": 4, "kill_margin": 0.0},
    "train_limit": 200_000,
}


def load_config(path: str | None, overrides: list[str] | None = None) -> RunConfig:
    cfg = OmegaConf.structured(RunConfig)
    if path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    if cfg.smoke:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(SMOKE), OmegaConf.from_dotlist(list(overrides or [])))
    return OmegaConf.to_object(cfg)


def to_yaml(cfg: RunConfig) -> str:
    return OmegaConf.to_yaml(OmegaConf.structured(cfg))
