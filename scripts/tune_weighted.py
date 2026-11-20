"""Tune the weighted random bot's weights with a (mu/mu_w, lambda)-ES against a pool of opponents.

Each generation: lambda children are sampled around the current mean (Gaussian in antithetic pairs, per-weight
step unit), the mean and all children play the same fresh window of fixed **train** deals in duplicate against
every opponent with the same keys, and the new mean is the rank-weighted average of the best mu children.
Fitness = IMPs/board averaged over the opponents (default: brl FSP, brl SL and the rule bot, so the weights do not
overfit one system). Eval deals are never used.

Noise-robust by design: recombination averages the noise of single evaluations, the step size follows a fixed
geometric schedule (sigma -> sigma/10) instead of a success rule, and the temperature is held fixed (only the
weights' ratios to it matter, so mutating both lets them all drift upwards together).

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
from bridgebid.policies.base import load
from bridgebid.policies.weighted_random import WeightedRandom, Weights, save_weights

SECONDS_PER_BOARD = 0.45e-3  # one duplicate board (two tables) of the weighted bot vs one opponent, M2 GPU
OPPONENTS = ["brl:pretrained-rl-with-fsp", "brl:sl", "rule"]
FREE = {"pass_base", "xx"}  # may take any sign; the others are >= 0
FIXED = {"temperature"}  # not mutated


def project(v: np.ndarray) -> np.ndarray:
    names = Weights.names()
    lo = np.array([-np.inf if n in FREE else (0.1 if n == "temperature" else 0.0) for n in names])
    return np.round(np.maximum(v, lo), 4)


def generations_for(minutes: float, boards: int, lam: int) -> int:
    return max(1, int(minutes * 60 / ((lam + 1) * boards * SECONDS_PER_BOARD)))


def tune(generations: int, seed: int = 0, boards: int = 2400, lam: int = 8, mu: int = 4, sigma: float = 0.2,
         pool: int = 200_000, opponents: list[str] | None = None, start: Weights | None = None,
         log=print) -> dict:
    """Returns the final mean weights and per-generation fitness of the mean (before its update)."""
    if lam < 2 or lam % 2 or not 1 <= mu <= lam:
        raise ValueError("--lam must be even and >= 2, and 1 <= --mu <= --lam")
    rng = np.random.default_rng(seed)
    bank = DealBank("train", limit=pool)
    if bank.size < boards:
        raise ValueError(f"--pool ({bank.size} train deals) must be at least --boards ({boards})")
    specs = OPPONENTS if opponents is None else opponents
    opps = [load(o) for o in specs]
    per_opp = max(boards // len(opps), 1)
    mean = project((start or Weights()).to_vector())
    scale = np.maximum(np.abs(mean), 0.5)  # per-weight step unit
    scale[[i for i, n in enumerate(Weights.names()) if n in FIXED]] = 0.0
    rw = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    rw /= rw.sum()

    def fitness(v: np.ndarray, first: int, gen: int) -> float:
        pol = WeightedRandom(Weights.from_vector(v))
        return float(np.mean([match(pol, opp, bank, n_boards=per_opp, start=first, seed=seed * 100_003 + 97 * gen + i)
                              .mean for i, opp in enumerate(opps)]))

    history = []
    for gen in range(generations):
        t0 = time.perf_counter()
        sig = sigma * 0.1 ** (gen / max(generations - 1, 1))
        first = (gen * per_opp) % max(bank.size - per_opp, 1)
        z = rng.standard_normal((lam // 2, len(mean)))
        kids = [project(mean + sig * scale * d) for d in np.concatenate([z, -z])]
        fit_mean = fitness(mean, first, gen)
        fit = np.array([fitness(k, first, gen) for k in kids])
        top = np.argsort(-fit, kind="stable")[:mu]
        mean = project(mean + sum(w * (kids[i] - mean) for w, i in zip(rw, top, strict=True)))
        history.append(round(fit_mean, 4))
        log(f"gen {gen + 1:>4}/{generations}  mean {fit_mean:+.3f}  best kid {fit[top[0]]:+.3f}  "
            f"sigma {sig:.3f}  {time.perf_counter() - t0:.1f}s")
    return {"weights": Weights.from_vector(mean), "mean_fitness": history, "generations": generations,
            "seed": seed, "boards": boards, "lambda": lam, "mu": mu, "sigma": sigma, "pool": pool,
            "opponents": list(specs)}


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--minutes", type=float, default=10.0, help="budget, turned into a fixed generation count")
    p.add_argument("--generations", type=int, help="override the generation count derived from --minutes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/weighted_tuned.json")
    p.add_argument("--boards", type=int, default=2400, help="train deals per generation, split over the opponents")
    p.add_argument("--lam", type=int, default=8, help="children per generation (even: antithetic pairs)")
    p.add_argument("--mu", type=int, default=4, help="best children averaged into the new mean")
    p.add_argument("--sigma", type=float, default=0.2, help="initial relative step size (decays to a tenth)")
    p.add_argument("--opponents", nargs="+", default=OPPONENTS, help="policy specs; fitness = mean IMPs/board")
    p.add_argument("--pool", type=int, default=200_000, help="train deals to draw generation windows from")
    a = p.parse_args(argv)
    gens = a.generations if a.generations is not None else generations_for(a.minutes, a.boards, a.lam)
    t0 = time.perf_counter()
    res = tune(gens, seed=a.seed, boards=a.boards, lam=a.lam, mu=a.mu, sigma=a.sigma, pool=a.pool,
               opponents=a.opponents)
    w = res.pop("weights")
    save_weights(w, a.out, **res)
    print(f"wrote {a.out} after {gens} generations in {(time.perf_counter() - t0) / 60:.1f} min")
    return {"weights": w, **res}


if __name__ == "__main__":
    main()
