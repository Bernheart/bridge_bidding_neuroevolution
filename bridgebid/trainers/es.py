"""OpenAI-style evolution strategies (the neuroevolution side of the comparison), in MLX.

Each generation:
  1. sample P/2 noise vectors eps; members are theta +/- sigma * eps (antithetic pairs);
  2. every member plays the SAME M boards in duplicate (it sits NS at table 1 and EW at table 2) against a
     shared opponent per board block: always-pass during the warm-up, then a mix of the current centre,
     a past centre and the rule bot;
  3. fitness = mean IMPs/board; centred ranks; gradient = sum(rank_i * eps_i) / (P * sigma); Adam step.
Members play greedily (argmax). A chunk of members is evaluated at once with stacked weights: one batched
matmul per layer for the whole chunk x all boards x both tables.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from bridgebid.config import RunConfig
from bridgebid.data.sampler import DealBank
from bridgebid.env import core
from bridgebid.models.actor_critic import PolicyNet
from bridgebid.models.io import save_policy
from bridgebid.policies.base import NeuralPolicy
from bridgebid.policies.rule_bot import RuleBot
from bridgebid.scoring import imps
from bridgebid.tracking.run import Run
from bridgebid.trainers.common import key_from_list, key_to_list, periodic_eval, save_json


def _layers(net: PolicyNet) -> list:
    return [*net.torso.layers, net.pi]


def flatten_net(net: PolicyNet) -> mx.array:
    return mx.concatenate([a.reshape(-1) for lyr in _layers(net) for a in (lyr.weight, lyr.bias)])


def net_shapes(net: PolicyNet) -> list[tuple[tuple, tuple]]:
    return [(tuple(lyr.weight.shape), tuple(lyr.bias.shape)) for lyr in _layers(net)]


def load_into(net: PolicyNet, theta: mx.array, shapes) -> None:
    off = 0
    for lyr, (ws, bs) in zip(_layers(net), shapes, strict=True):
        nw, nb = int(np.prod(ws)), int(np.prod(bs))
        lyr.weight = theta[off:off + nw].reshape(ws)
        lyr.bias = theta[off + nw:off + nw + nb].reshape(bs)
        off += nw + nb
    mx.eval(net.parameters())


def stack_params(thetas: mx.array, shapes) -> list[mx.array]:
    """[Pc, D] -> [w0, b0, w1, b1, ...] with w as [Pc, in, out] and b as [Pc, 1, out]."""
    out, off = [], 0
    pc = thetas.shape[0]
    for ws, bs in shapes:
        nw, nb = int(np.prod(ws)), int(np.prod(bs))
        out.append(mx.swapaxes(thetas[:, off:off + nw].reshape(pc, *ws), 1, 2))
        out.append(thetas[:, off + nw:off + nw + nb].reshape(pc, 1, *bs))
        off += nw + nb
    return out


def stacked_forward(x: mx.array, *params: mx.array) -> mx.array:
    """x [Pc, B, in] -> logits [Pc, B, 38]: every member's MLP on its own batch, one matmul per layer."""
    h = x
    for i in range(0, len(params) - 2, 2):
        h = mx.maximum(h @ params[i] + params[i + 1], 0)
    return h @ params[-2] + params[-1]


def es_gradient(fitness: np.ndarray, eps_half: mx.array, sigma: float) -> mx.array:
    """Antithetic OpenAI-ES estimate with centred ranks. fitness [P] (first half +eps, second half -eps)."""
    p = len(fitness)
    ranks = np.empty(p)
    ranks[np.argsort(fitness, kind="stable")] = np.arange(p)
    u = ranks / (p - 1) - 0.5
    half = p // 2
    w = mx.array((u[:half] - u[half:]).astype(np.float32))
    return (w @ eps_half) / (p * sigma)


class ESTrainer:
    def __init__(self, cfg: RunConfig, run_dir: str | Path | None = None, resume: bool = False):
        self.cfg, self.e = cfg, cfg.es
        e = self.e
        assert e.population % 2 == 0 and e.population % e.chunk == 0 and e.chunk % 2 == 0
        self.run = Run(cfg, run_dir)
        self.bank = DealBank(cfg.train_split, limit=cfg.train_limit)
        self.eval_bank = DealBank("eval")
        mx.random.seed(cfg.seed)
        self.net = PolicyNet(e.width, e.depth, value_head=False)  # the centre, as a normal network
        self.opp_net = PolicyNet(e.width, e.depth, value_head=False)  # for past centres
        self.shapes = net_shapes(self.net)
        self.theta = flatten_net(self.net)
        self.dim = self.theta.shape[0]
        self.m = mx.zeros_like(self.theta)
        self.v = mx.zeros_like(self.theta)
        self.t = 0
        self.gen = 0
        self.pool: list[mx.array] = []
        self.key = mx.random.key(cfg.seed)
        self.rule = RuleBot()
        self.eval_cache: dict = {}
        self.killed = False
        self._member_logits = mx.compile(stacked_forward)
        self._step = mx.compile(core.step)
        if resume:
            self._load()

    # ------------------------------------------------------------------ opponents
    def _blocks(self) -> list[tuple[str, int, int]]:
        e, m = self.e, self.e.boards
        if self.gen < e.warmup_generations:
            return [("pass", 0, m)]
        n_c = int(round(m * e.mix_centre))
        n_p = int(round(m * e.mix_pool)) if self.pool else 0
        blocks, b0 = [], 0
        for kind, cnt in (("centre", n_c), ("pool", n_p), ("rule", m - n_c - n_p)):
            if cnt > 0:
                blocks.append((kind, b0, b0 + cnt))
                b0 += cnt
        return blocks

    def _opp_actions(self, kind: str, s: core.State, obs: mx.array, mask: mx.array) -> mx.array:
        if kind == "pass":
            return mx.zeros(obs.shape[:1], mx.int32)
        if kind == "rule":
            return self.rule.act(s, obs, mask, None)
        net = self.net if kind == "centre" else self.opp_net
        return mx.argmax(mx.where(mask, net(obs)[0], -1e9), axis=-1).astype(mx.int32)

    # ------------------------------------------------------------------ evaluation of a chunk of members
    def _play_chunk(self, thetas: mx.array, boards: core.State, blocks) -> mx.array:
        """Duplicate IMPs/board for each member in `thetas` [Pc, D] on `boards` [M]. Returns [Pc]."""
        pc, m = thetas.shape[0], boards["dealer"].shape[0]
        params = stack_params(thetas, self.shapes)
        s = {k: mx.broadcast_to(v[None, None], (pc, 2, *v.shape)).reshape(pc * 2 * m, *v.shape[1:])
             for k, v in boards.items()}
        member_ns = mx.broadcast_to(mx.array([True, False])[None, :, None], (pc, 2, m)).reshape(-1)
        for ply in range(320):
            obs, mask = core.observe(s), core.legal_mask(s)
            member_turn = ((core.current_seat(s) % 2) == 0) == member_ns
            logits = self._member_logits(obs.reshape(pc, 2 * m, -1), *params).reshape(pc * 2 * m, -1)
            a_mem = mx.argmax(mx.where(mask, logits, -1e9), axis=-1).astype(mx.int32)
            parts = []
            for kind, b0, b1 in blocks:
                sel = {k: v.reshape(pc, 2, m, *v.shape[1:])[:, :, b0:b1].reshape(-1, *v.shape[1:])
                       for k, v in s.items()}
                o = obs.reshape(pc, 2, m, -1)[:, :, b0:b1].reshape(-1, obs.shape[-1])
                mk = mask.reshape(pc, 2, m, -1)[:, :, b0:b1].reshape(-1, mask.shape[-1])
                parts.append(self._opp_actions(kind, sel, o, mk).reshape(pc, 2, b1 - b0))
            a_opp = mx.concatenate(parts, axis=2).reshape(-1)
            s = self._step(s, mx.where(member_turn, a_mem, a_opp))
            mx.eval(s)
            if ply % 4 == 3 and mx.all(s["done"]).item():
                break
        score = s["score_ns"].reshape(pc, 2, m)
        return imps(mx, score[:, 0] - score[:, 1]).astype(mx.float32).mean(axis=1)

    # ------------------------------------------------------------------ one generation
    def generation(self) -> dict:
        e = self.e
        half = e.population // 2
        self.key, k_eps, k_boards = mx.random.split(self.key, 3)
        eps = mx.random.normal((half, self.dim), key=k_eps)
        boards = self.bank.sample_state(e.boards, k_boards)
        blocks = self._blocks()
        if any(k == "pool" for k, _, _ in blocks):
            rng = np.random.default_rng((self.cfg.seed, self.gen))
            load_into(self.opp_net, self.pool[int(rng.integers(len(self.pool)))], self.shapes)
        fit = []
        signs = mx.concatenate([mx.ones((half,)), -mx.ones((half,))])
        for c0 in range(0, e.population, e.chunk):
            idx = mx.arange(c0, c0 + e.chunk)
            thetas = self.theta[None] + e.sigma * signs[idx][:, None] * eps[idx % half]
            fit.append(self._play_chunk(thetas, boards, blocks))
        fitness = np.array(mx.concatenate(fit))
        grad = es_gradient(fitness, eps, e.sigma)  # ascent direction
        # Adam (maximising) with decoupled weight decay
        self.t += 1
        b1, b2 = 0.9, 0.999
        self.m = b1 * self.m + (1 - b1) * grad
        self.v = b2 * self.v + (1 - b2) * grad * grad
        mhat, vhat = self.m / (1 - b1 ** self.t), self.v / (1 - b2 ** self.t)
        self.theta = self.theta + e.lr * mhat / (mx.sqrt(vhat) + 1e-8) - e.lr * e.weight_decay * self.theta
        mx.eval(self.theta, self.m, self.v)
        load_into(self.net, self.theta, self.shapes)
        return {"fitness_mean": float(fitness.mean()), "fitness_max": float(fitness.max()),
                "fitness_std": float(fitness.std()), "grad_norm": float(mx.linalg.norm(grad).item()),
                "theta_norm": float(mx.linalg.norm(self.theta).item()),
                "opponents": "+".join(k for k, _, _ in blocks)}

    # ------------------------------------------------------------------ checkpoints
    def save(self) -> None:
        d = self.run.dir / "ckpt"
        mx.save_safetensors(str(d / "es_state.safetensors"),
                            {"theta": self.theta, "m": self.m, "v": self.v,
                             **{f"pool_{i:04d}": p for i, p in enumerate(self.pool)}})
        save_policy(self.net, d / "policy", meta={"generation": self.gen})
        save_json({"gen": self.gen, "t": self.t, "key": key_to_list(self.key), "pool": len(self.pool),
                   "killed": self.killed}, d / "trainer.json")

    def _load(self) -> None:
        d = self.run.dir / "ckpt"
        meta = json.loads((d / "trainer.json").read_text())
        st = mx.load(str(d / "es_state.safetensors"))
        self.theta, self.m, self.v = st["theta"], st["m"], st["v"]
        self.pool = [st[f"pool_{i:04d}"] for i in range(meta["pool"])]
        self.gen, self.t, self.key = meta["gen"], meta["t"], key_from_list(meta["key"])
        self.killed = meta.get("killed", False)
        load_into(self.net, self.theta, self.shapes)
        print(f"resumed {self.run.dir} at generation {self.gen}")

    # ------------------------------------------------------------------ main loop
    def evaluate(self) -> dict:
        load_into(self.net, self.theta, self.shapes)
        pol = NeuralPolicy(self.net, name=f"g{self.gen}")
        res = periodic_eval(pol, self.eval_bank, self.e.eval_opponents, self.e.eval_boards, self.eval_cache)
        save_policy(self.net, self.run.dir / "snapshots" / f"g{self.gen:05d}",
                    meta={"generation": self.gen, **{k: v for k, v in res.items() if k.startswith("imps")}})
        return res

    def train(self, max_generations: int | None = None, max_hours: float | None = None) -> None:
        """Train to the configured budget, or stop (with a checkpoint) after max_generations / max_hours."""
        session_start, start_count = time.time(), self.gen
        e = self.e
        n_games = e.population * e.boards * 2
        print(f"ES run {self.run.dir}: {e.generations} generations, population {e.population}, "
              f"{e.boards} boards x 2 tables -> {n_games:,} auctions per generation, {self.dim:,} parameters")
        try:
            while self.gen < e.generations and not self.killed:
                if max_generations is not None and self.gen - start_count >= max_generations:
                    break
                if max_hours is not None and time.time() - session_start > max_hours * 3600:
                    print(f"time budget of {max_hours} h reached")
                    break
                t0 = time.time()
                stats = self.generation()
                self.gen += 1
                stats.update(gen_s=time.time() - t0, auctions_per_s=n_games / (time.time() - t0))
                if e.snapshot_every and self.gen % e.snapshot_every == 0:
                    self.pool = (self.pool + [self.theta])[-e.pool_size:]
                self.run.log(self.gen, stats, prefix="train/")
                print(f"g{self.gen:5d}  fitness mean {stats['fitness_mean']:+6.2f} max {stats['fitness_max']:+6.2f}"
                      f"  |theta| {stats['theta_norm']:.1f}  vs {stats['opponents']}  {stats['gen_s']:.1f}s",
                      flush=True)
                kill_check = self.gen == e.warmup_generations and e.kill_margin > 0
                if self.gen % e.eval_every == 0 or self.gen == e.generations or kill_check:
                    ev = self.evaluate()
                    self.run.log(self.gen, ev, prefix="eval/")
                    print("   eval: " + "  ".join(f"{k}={v}" for k, v in ev.items() if k.startswith("imps"))
                          + f"  passout={ev['pass_out_rate']:.2f} top={ev['top_openings']}"
                          + (f"  ALARMS: {ev['alarms']}" if ev["alarms"] else ""), flush=True)
                    if kill_check and ev.get("imps_vs_always-pass", 0.0) < e.kill_margin:
                        self.killed = True
                        print(f"   KILL CRITERION: centre beats always-pass by only "
                              f"{ev.get('imps_vs_always-pass', 0.0):+.2f} < {e.kill_margin} IMPs/board after the "
                              f"warm-up. Stopping (resume with es.kill_margin=0 to continue anyway).")
                if self.gen % e.ckpt_every == 0:
                    self.save()
        except KeyboardInterrupt:
            print("\ninterrupted - saving checkpoint ...")
        self.save()
        self.run.close()
        print(f"checkpoint saved. Resume with:  uv run bridgebid resume {self.run.dir}")
