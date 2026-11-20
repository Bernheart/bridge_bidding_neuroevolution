"""Evaluate policies against a fixed ladder of opponents (duplicate IMPs/board)."""

from __future__ import annotations

import json
from pathlib import Path

from bridgebid.data.sampler import DealBank
from bridgebid.eval.duplicate import match
from bridgebid.policies.base import Policy, load

DEFAULT_LADDER = ["weighted", "weighted-tuned", "pass", "rule", "brl:from-scratch-rl", "brl:sl", "brl:pretrained-rl",
                  "brl:pretrained-rl-with-fsp"]
QUICK_LADDER = ["pass", "rule", "brl:sl"]


def available(specs: list[str]) -> list[str]:
    return [s for s in specs if not s.startswith("brl:")
            or Path(f"models/brl/model-{s[4:]}.safetensors").exists()]


def evaluate(policy: Policy, bank: DealBank, opponents: list[str], n_boards: int = 10_000,
             cache: dict[str, Policy] | None = None) -> dict[str, tuple[float, float]]:
    cache = {} if cache is None else cache
    out = {}
    for spec in available(opponents):
        opp = cache.setdefault(spec, load(spec))
        r = match(policy, opp, bank, n_boards=n_boards)
        out[opp.name] = (r.mean, r.se)
    return out


def ladder_matrix(specs: list[str], bank: DealBank, n_boards: int = 10_000) -> dict:
    specs = available(specs)
    pols = {s: load(s) for s in specs}
    rows = {}
    for a in specs:
        rows[pols[a].name] = {}
        for b in specs:
            if a == b:
                rows[pols[a].name][pols[b].name] = (0.0, 0.0)
                continue
            if pols[b].name in rows and pols[a].name in rows[pols[b].name]:
                m, se = rows[pols[b].name][pols[a].name]
                rows[pols[a].name][pols[b].name] = (-m, se)
                continue
            r = match(pols[a], pols[b], bank, n_boards=n_boards)
            rows[pols[a].name][pols[b].name] = (r.mean, r.se)
    return rows


def format_matrix(rows: dict) -> str:
    names = list(rows)
    w = max(len(n) for n in names) + 2
    head = " " * w + "".join(f"{n[:12]:>14}" for n in names)
    lines = [head]
    for a in names:
        cells = "".join(f"{rows[a][b][0]:>+9.2f}±{rows[a][b][1]:.2f}" if a != b else f"{'-':>14}" for b in names)
        lines.append(f"{a:<{w}}{cells}")
    return "\n".join(lines)


def save(rows: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rows, indent=2))
