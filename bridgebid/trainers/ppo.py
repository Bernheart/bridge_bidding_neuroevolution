"""PPO self-play for competitive bidding, in MLX.

* One policy plays all four seats; every seat's decision is a training sample (unless it belongs to a forced-
  pass side during the warm-up, or to a past snapshot during fictitious self-play).
* Reward: final duplicate score of the contract from the acting side's point of view / reward_scale. Bidding
  has no intermediate rewards, so with gamma = 1 every decision's return is the final score.
* GAE with stride 4: the "next state" of a decision is the same player's next turn, 4 calls later.
  Each update plays rollout_len + 4 calls; the last 4 only supply bootstrap values.
* Critic: 'privileged' sees all hands and the DD table (training only - the actor never does), or 'shared'
  (value head on the actor).
* Curriculum: for the first passive_until steps, one random side per deal is forced to pass (uncontested
  bidding), then that probability anneals to 0 and fictitious self-play against past snapshots starts.
"""

from __future__ import annotations

import time
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from bridgebid.config import RunConfig
from bridgebid.constants import OBS_DIM
from bridgebid.data.sampler import DealBank
from bridgebid.env import core
from bridgebid.models.actor_critic import PolicyNet, ValueNet
from bridgebid.models.io import save_policy
from bridgebid.policies.base import NeuralPolicy
from bridgebid.tracking.run import Run
from bridgebid.trainers.common import (
    PRIV_DIM,
    critic_input,
    key_from_list,
    key_to_list,
    linear,
    load_tree,
    periodic_eval,
    privileged_extra,
    save_json,
    save_tree,
)


class Learner(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.actor = PolicyNet(c.width, c.depth, value_head=(c.critic == "shared"))
        if c.critic == "privileged":
            self.critic = ValueNet(OBS_DIM + PRIV_DIM, c.critic_width, c.critic_depth)

    def value(self, obs: mx.array, extra: mx.array, v_shared: mx.array | None) -> mx.array:
        if "critic" in self:
            return self.critic(critic_input(obs, extra))
        return v_shared


class PPOTrainer:
    def __init__(self, cfg: RunConfig, run_dir: str | Path | None = None, resume: bool = False):
        self.cfg, self.c = cfg, cfg.ppo
        c = self.c
        self.run = Run(cfg, run_dir)
        self.bank = DealBank(cfg.train_split, limit=cfg.train_limit)
        self.eval_bank = DealBank("eval")
        self.priv = c.critic == "privileged"
        mx.random.seed(cfg.seed)
        self.model = Learner(c)
        mx.eval(self.model.parameters())
        self.opt = optim.Adam(learning_rate=c.lr)
        self.opt.init(self.model.trainable_parameters())
        self.opp_nets = [PolicyNet(c.width, c.depth, value_head=False) for _ in range(c.fsp_groups)]
        self.key = mx.random.key(cfg.seed)
        self.update = 0
        self.steps = 0
        self.pool: list = []
        self.pool_saved = 0
        self.steps_per_update = c.num_envs * (c.rollout_len + 4)
        self.total_updates = max(1, c.total_steps // self.steps_per_update)
        self.eval_cache: dict = {}
        self._build()
        if resume:
            self._load()
        else:
            self.key, k1, k2 = mx.random.split(self.key, 3)
            self.env = self.bank.sample_state(c.num_envs, k1)
            self.roles = self._new_roles(k2)

    # ------------------------------------------------------------------ schedules
    def p_passive(self) -> float:
        c = self.c
        if self.steps < c.passive_until:
            return 1.0
        return linear(1.0, 0.0, (self.steps - c.passive_until) / max(c.passive_anneal, 1))

    def fsp_active(self) -> bool:
        return self.p_passive() == 0.0 and len(self.pool) > 0 and self.c.fsp_frac > 0

    def n_pool_envs(self) -> int:
        if not self.fsp_active():
            return 0
        g = self.c.fsp_groups
        return int(self.c.fsp_frac * self.c.num_envs) // g * g

    def _new_roles(self, key: mx.array) -> dict[str, mx.array]:
        n = self.c.num_envs
        k1, k2 = mx.random.split(key)
        team = mx.random.randint(0, 2, (n,), key=k1)
        passive = mx.where(mx.random.uniform(shape=(n,), key=k2) < self.p_passive(), team, -1)
        in_pool = (mx.arange(n) < self.n_pool_envs()) & (passive < 0)
        return {"passive": passive.astype(mx.int32), "opp": mx.where(in_pool, 1 - team, -1).astype(mx.int32)}

    # ------------------------------------------------------------------ compiled pieces
    def _build(self) -> None:
        c, model = self.c, self.model

        def act(obs, extra, mask, key):
            logits, v_sh = model.actor(obs)
            logits = mx.where(mask, logits, -1e9)
            a = mx.random.categorical(logits, key=key)
            logp = mx.take_along_axis(logits - mx.logsumexp(logits, axis=-1, keepdims=True), a[:, None], axis=1)
            return a.astype(mx.int32), logp[:, 0], model.value(obs, extra, v_sh)

        self._act = mx.compile(act, inputs=[model.state])
        self._opp_act = []
        for net in self.opp_nets:
            def opp(obs, mask, key, net=net):
                return mx.random.categorical(mx.where(mask, net(obs)[0], -1e9), key=key).astype(mx.int32)
            self._opp_act.append(mx.compile(opp, inputs=[net.state]))
        self._env_step = mx.compile(core.step)

        def loss_fn(model, obs, extra, mask, act, old_logp, old_v, adv, ret, ent_coef):
            x = obs.astype(mx.float32)
            logits, v_sh = model.actor(x)
            logits = mx.where(mask, logits, -1e9)
            logp_all = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            logp = mx.take_along_axis(logp_all, act[:, None], axis=1)[:, 0]
            ratio = mx.exp(logp - old_logp)
            adv = (adv - adv.mean()) / (mx.sqrt(mx.var(adv)) + 1e-8)
            pg = -mx.minimum(ratio * adv, mx.clip(ratio, 1 - c.clip, 1 + c.clip) * adv).mean()
            ent = -mx.where(mask, mx.exp(logp_all) * logp_all, 0.0).sum(-1).mean()
            v = model.value(x, extra, v_sh)
            v_clip = old_v + mx.clip(v - old_v, -c.vf_clip, c.vf_clip)
            vf = 0.5 * mx.maximum((v - ret) ** 2, (v_clip - ret) ** 2).mean()
            loss = pg - ent_coef * ent + c.vf_coef * vf
            kl = ((ratio - 1) - (logp - old_logp)).mean()
            clipfrac = (mx.abs(ratio - 1) > c.clip).astype(mx.float32).mean()
            return loss, (pg, vf, ent, kl, clipfrac)

        grad_fn = nn.value_and_grad(model, loss_fn)
        state = [model.state, self.opt.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def train_step(obs, extra, mask, act, old_logp, old_v, adv, ret, ent_coef):
            (loss, aux), grads = grad_fn(model, obs, extra, mask, act, old_logp, old_v, adv, ret, ent_coef)
            grads, gnorm = optim.clip_grad_norm(grads, c.max_grad_norm)
            self.opt.update(model, grads)
            return (loss, gnorm, *aux)

        self._train_step = train_step

    # ------------------------------------------------------------------ rollout
    def rollout(self) -> tuple[dict[str, mx.array], dict[str, float]]:
        c = self.c
        n, plies = c.num_envs, c.rollout_len + 4
        rng = np.random.default_rng((self.cfg.seed, self.update))
        n_pool = self.n_pool_envs()
        group = n_pool // c.fsp_groups if n_pool else 0
        if n_pool:
            for net, i in zip(self.opp_nets, rng.integers(0, len(self.pool), c.fsp_groups), strict=True):
                net.update(self.pool[i])
        keys = ("obs", "extra", "mask", "act", "logp", "val", "team", "learn", "ended", "score")
        buf: dict[str, list] = {k: [] for k in keys}
        ep_n = ep_len = ep_pass = ep_abs = mx.array(0.0)
        for _ in range(plies):
            self.key, k_act, k_opp, k_new, k_roles = mx.random.split(self.key, 5)
            s = self.env
            obs, mask = core.observe(s), core.legal_mask(s)
            team = core.current_seat(s) % 2
            extra = privileged_extra(s) if self.priv else mx.zeros((n, 1), mx.uint8)
            a, logp, v = self._act(obs, extra, mask, k_act)
            passive = self.roles["passive"] == team
            opp = self.roles["opp"] == team
            if n_pool:
                ks = mx.random.split(k_opp, c.fsp_groups)
                a_opp = [f(obs[g * group:(g + 1) * group], mask[g * group:(g + 1) * group], ks[g])
                         for g, f in enumerate(self._opp_act)]
                a_opp = mx.concatenate(a_opp + [mx.zeros((n - n_pool,), mx.int32)])
                a = mx.where(opp, a_opp, a)
            a = mx.where(passive, 0, a)
            s = self._env_step(s, a)
            ended, score = s["done"], s["score_ns"]
            ep_n = ep_n + ended.sum()
            ep_len = ep_len + mx.where(ended, s["turn"], 0).sum()
            ep_pass = ep_pass + (ended & (s["last_bid"] < 0)).sum()
            ep_abs = ep_abs + mx.where(ended, mx.abs(score), 0).sum()
            self.env = core.reset_where(s, ended, self.bank.sample_state(n, k_new))
            new = self._new_roles(k_roles)
            self.roles = {k: mx.where(ended, new[k], v_) for k, v_ in self.roles.items()}
            for k, val in zip(keys, (obs.astype(mx.bool_), extra, mask, a, logp, v, team, ~passive & ~opp, ended,
                                     score), strict=True):
                buf[k].append(val)
            mx.eval(self.env, self.roles, a, logp, v, ep_n)
        data = {k: mx.stack(v) for k, v in buf.items()}
        eps = max(ep_n.item(), 1.0)
        stats = {"episodes": int(ep_n.item()), "ep_len": ep_len.item() / eps, "ep_passout": ep_pass.item() / eps,
                 "ep_abs_score": ep_abs.item() / eps, "pool_envs": n_pool}
        return data, stats

    def advantages(self, data: dict[str, mx.array]) -> tuple[np.ndarray, np.ndarray]:
        """Stride-4 GAE (gamma = 1). Returns (advantages, returns) of shape [rollout_len, N]."""
        c = self.c
        T = c.rollout_len
        ended, score = np.array(data["ended"]), np.array(data["score"]).astype(np.float32)
        team, val = np.array(data["team"]), np.array(data["val"]).astype(np.float32)
        n = ended.shape[1]
        ar = np.arange(n)
        term = np.zeros((T, n), bool)
        rew = np.zeros((T, n), np.float32)
        for t in range(T):
            w = ended[t:t + 4]
            term[t] = w.any(0)
            e = t + w.argmax(0)
            sign = np.where(team[t] == 0, 1.0, -1.0)
            rew[t] = np.where(term[t], score[e, ar] * sign / c.reward_scale, 0.0)
        adv = np.zeros((T + 4, n), np.float32)
        for t in reversed(range(T)):
            delta = rew[t] + (~term[t]) * val[t + 4] - val[t]
            adv[t] = delta + c.gae_lambda * (~term[t]) * adv[t + 4]
        adv = adv[:T]
        return adv, adv + val[:T]

    # ------------------------------------------------------------------ update
    def learn(self, data: dict[str, mx.array]) -> dict[str, float]:
        c = self.c
        T, n = c.rollout_len, c.num_envs
        adv, ret = self.advantages(data)
        learn = np.array(data["learn"][:T]).reshape(-1)
        idx = np.flatnonzero(learn)
        flat = {k: data[k][:T].reshape((T * n,) + data[k].shape[2:]) for k in
                ("obs", "extra", "mask", "act", "logp", "val")}
        flat["adv"], flat["ret"] = mx.array(adv.reshape(-1)), mx.array(ret.reshape(-1))
        frac = self.update / self.total_updates
        ent_coef = mx.array(linear(c.ent_start, c.ent_end, frac / max(c.ent_decay_frac, 1e-9)))
        self.opt.learning_rate = c.lr * (1 - frac) if c.lr_anneal else c.lr
        rng = np.random.default_rng((self.cfg.seed, self.update, 1))
        mb = min(c.minibatch, len(idx))
        sums = np.zeros(7)
        count = 0
        stopped = False
        for _ in range(c.epochs):
            perm = rng.permutation(idx)
            for i in range(len(perm) // mb):
                b = mx.array(perm[i * mb:(i + 1) * mb])
                out = self._train_step(*(flat[k][b] for k in ("obs", "extra", "mask", "act", "logp", "val", "adv",
                                                               "ret")), ent_coef)
                mx.eval(out)
                vals = np.array([o.item() for o in out])
                sums += vals
                count += 1
                if vals[5] > 1.5 * c.target_kl:
                    stopped = True
                    break
            if stopped:
                break
        m = sums / max(count, 1)
        v, r = np.array(data["val"][:T]).reshape(-1)[idx], ret.reshape(-1)[idx]
        ev = 1 - np.var(r - v) / (np.var(r) + 1e-8)
        return {"loss": m[0], "grad_norm": m[1], "pg_loss": m[2], "vf_loss": m[3], "entropy": m[4],
                "approx_kl": m[5], "clipfrac": m[6], "explained_var": float(ev), "minibatches": count,
                "kl_stop": int(stopped), "learner_frac": len(idx) / (T * n), "ent_coef": ent_coef.item(),
                "lr": float(self.opt.learning_rate.item())}

    # ------------------------------------------------------------------ checkpoints
    def _ckpt_dir(self) -> Path:
        return self.run.dir / "ckpt"

    def save(self) -> None:
        d = self._ckpt_dir()
        save_tree(self.model.parameters(), d / "model.safetensors")
        save_tree(self.opt.state, d / "opt.safetensors")
        save_policy(self.model.actor, d / "policy", meta={"update": self.update, "steps": self.steps})
        mx.savez(str(d / "env.npz"), **self.env, **{f"role_{k}": v for k, v in self.roles.items()})
        (d / "pool").mkdir(exist_ok=True)
        for i in range(self.pool_saved, len(self.pool)):
            save_tree(self.pool[i], d / "pool" / f"{i:04d}.safetensors")
        self.pool_saved = len(self.pool)
        save_json({"update": self.update, "steps": self.steps, "key": key_to_list(self.key),
                   "pool": len(self.pool)}, d / "trainer.json")

    def _load(self) -> None:
        import json

        d = self._ckpt_dir()
        meta = json.loads((d / "trainer.json").read_text())
        self.model.update(load_tree(d / "model.safetensors"))
        self.opt.state = load_tree(d / "opt.safetensors")
        env = dict(mx.load(str(d / "env.npz")))
        self.roles = {k[5:]: env.pop(k) for k in list(env) if k.startswith("role_")}
        self.env = env
        self.pool = [load_tree(d / "pool" / f"{i:04d}.safetensors") for i in range(meta["pool"])]
        self.pool_saved = len(self.pool)
        self.update, self.steps = meta["update"], meta["steps"]
        self.key = key_from_list(meta["key"])
        mx.eval(self.model.parameters(), self.opt.state)
        self._build()  # compiled steps must capture the restored optimizer state object
        print(f"resumed {self.run.dir} at update {self.update} ({self.steps:,} steps)")

    # ------------------------------------------------------------------ main loop
    def evaluate(self) -> dict:
        pol = NeuralPolicy(self.model.actor, name=f"u{self.update}")
        res = periodic_eval(pol, self.eval_bank, self.c.eval_opponents, self.c.eval_boards, self.eval_cache)
        save_policy(self.model.actor, self.run.dir / "snapshots" / f"u{self.update:05d}",
                    meta={"update": self.update, "steps": self.steps, **{k: v for k, v in res.items()
                                                                          if k.startswith("imps")}})
        return res

    def train(self, max_updates: int | None = None, max_hours: float | None = None) -> None:
        """Train to the configured budget, or stop (with a checkpoint) after max_updates / max_hours."""
        session_start, start_count = time.time(), self.update
        c = self.c
        print(f"PPO run {self.run.dir}: {self.total_updates} updates x {self.steps_per_update:,} steps")
        try:
            while self.update < self.total_updates:
                if max_updates is not None and self.update - start_count >= max_updates:
                    break
                if max_hours is not None and time.time() - session_start > max_hours * 3600:
                    print(f"time budget of {max_hours} h reached")
                    break
                t0 = time.time()
                data, stats = self.rollout()
                t1 = time.time()
                metrics = self.learn(data)
                t2 = time.time()
                self.update += 1
                self.steps += self.steps_per_update
                if c.fsp_every and self.update % c.fsp_every == 0:
                    self.pool.append(dict(self.model.actor.parameters()))
                    self.pool = self.pool[-c.fsp_pool:]
                    self.pool_saved = min(self.pool_saved, len(self.pool))
                metrics.update(stats, steps=self.steps, p_passive=self.p_passive(), pool=len(self.pool),
                               sps=self.steps_per_update / (t2 - t0), rollout_s=t1 - t0, learn_s=t2 - t1)
                self.run.log(self.steps, metrics, prefix="train/")
                print(f"u{self.update:5d} {self.steps / 1e6:8.1f}M  sps {metrics['sps']:7.0f}  "
                      f"len {stats['ep_len']:5.2f} passout {stats['ep_passout']:.2f} |score| "
                      f"{stats['ep_abs_score']:6.1f}  ent {metrics['entropy']:.3f} kl {metrics['approx_kl']:.4f} "
                      f"ev {metrics['explained_var']:.2f}  p_passive {metrics['p_passive']:.2f}", flush=True)
                if self.update % c.eval_every == 0 or self.update == self.total_updates:
                    ev = self.evaluate()
                    self.run.log(self.steps, ev, prefix="eval/")
                    print("   eval: " + "  ".join(f"{k}={v}" for k, v in ev.items() if k.startswith("imps"))
                          + f"  passout={ev['pass_out_rate']:.2f} top={ev['top_openings']}"
                          + (f"  ALARMS: {ev['alarms']}" if ev["alarms"] else ""), flush=True)
                if self.update % c.ckpt_every == 0:
                    self.save()
        except KeyboardInterrupt:
            print("\ninterrupted - saving checkpoint ...")
        self.save()
        self.run.close()
        print(f"checkpoint saved. Resume with:  uv run bridgebid resume {self.run.dir}")
