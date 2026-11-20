"""Tune the weighted random bot's weights with a (1+lambda)-ES against the rule bot.

Each generation: lambda children are sampled around the parent (Gaussian, per-weight step size), the parent and
all children play the same fresh window of fixed **train** deals in duplicate against `rule-bot` with the same
key, and the best replaces the parent if it beat it on those deals. The step size follows a 1/5-style success
rule. Fitness = IMPs/board vs the rule bot. Eval deals are never used.

`--minutes` is turned into a generation count with a fixed cost estimate per board (measured on an M2), so a
run with the same arguments and seed always does the same work and writes the same JSON.

    uv run python scripts/tune_weighted.py --minutes 1 --seed 0 --out runs/weighted_smoke.json   # smoke
    uv run python scripts/tune_weighted.py --minutes 60 --seed 0      # -> results/weighted_tuned.json
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from bridgebid.data.sampler import DealBank
from bridgebid.eval.duplicate import match
from bridgebid.policies.rule_bot import RuleBot
from bridgebid.policies.weighted_random import WeightedRandom, Weights, save_weights

SECONDS_PER_BOARD = 0.55e-3  # one duplicate board (two tables) of weighted vs rule bot, M2 GPU
FREE = {"pass_base", "xx"}  # may take any sign; the others are >= 0 (temperature >= 0.1)


def project(v: np.ndarray) -> np.ndarray:
    names = Weights.names()
    lo = np.array([-np.inf if n in FREE else (0.1 if n == "temperature" else 0.0) for n in names])
    return np.round(np.maximum(v, lo), 4)


def generations_for(minutes: float, boards: int, lam: int) -> int:
    return max(1, int(minutes * 60 / ((lam + 1) * boards * SECONDS_PER_BOARD)))


def tune(generations: int, seed: int = 0, boards: int = 2000, lam: int = 8, sigma: float = 0.15,
         pool: int = 200_000, start: Weights | None = None, log=print) -> dict:
    rng = np.random.default_rng(seed)
    bank = DealBank("train", limit=pool)
    if bank.size < boards:
        raise ValueError(f"--pool ({bank.size} train deals) must be at least --boards ({boards})")
    opp = RuleBot()
    parent = project((start or Weights()).to_vector())
    scale = np.maximum(np.abs(parent), 0.5)  # per-weight step unit
    history = []
    for gen in range(generations):
        t0 = time.perf_counter()
        first = (gen * boards) % max(bank.size - boards, 1)
        kids = [project(parent + sigma * scale * rng.standard_normal(len(parent))) for _ in range(lam)]
        cands = [parent, *kids]
        fit = [match(WeightedRandom(Weights.from_vector(c)), opp, bank, n_boards=boards, start=first,
                     seed=seed * 100_003 + gen).mean for c in cands]
        best = int(np.argmax(fit))
        if best > 0:
            parent = cands[best]
            sigma *= 1.25
        else:
            sigma *= 1.25 ** -0.25
        history.append(round(fit[0], 4))
        log(f"gen {gen + 1:>4}/{generations}  parent {fit[0]:+.3f}  best {fit[best]:+.3f}  "
            f"{'accepted' if best else 'kept'}  sigma {sigma:.3f}  {time.perf_counter() - t0:.1f}s")
    return {"weights": Weights.from_vector(parent), "parent_fitness": history, "generations": generations,
            "seed": seed, "boards": boards, "lambda": lam, "pool": pool}


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--minutes", type=float, default=10.0, help="budget, turned into a fixed generation count")
    p.add_argument("--generations", type=int, help="override the generation count derived from --minutes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/weighted_tuned.json")
    p.add_argument("--boards", type=int, default=2000, help="train deals per generation")
    p.add_argument("--lam", type=int, default=8, help="children per generation")
    p.add_argument("--sigma", type=float, default=0.15, help="initial relative step size")
    p.add_argument("--pool", type=int, default=200_000, help="train deals to draw generation windows from")
    a = p.parse_args(argv)
    gens = a.generations if a.generations is not None else generations_for(a.minutes, a.boards, a.lam)
    t0 = time.perf_counter()
    res = tune(gens, seed=a.seed, boards=a.boards, lam=a.lam, sigma=a.sigma, pool=a.pool)
    w = res.pop("weights")
    save_weights(w, a.out, **res)
    print(f"wrote {a.out} after {gens} generations in {(time.perf_counter() - t0) / 60:.1f} min")
    return {"weights": w, **res}


if __name__ == "__main__":
    main()
