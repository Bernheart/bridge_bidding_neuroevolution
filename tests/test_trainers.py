"""Trainer maths: stacked population forward, ES gradient estimate, stride-4 GAE, exact resume."""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from bridgebid.models.actor_critic import PolicyNet
from bridgebid.trainers.es import es_gradient, flatten_net, load_into, net_shapes, stack_params, stacked_forward


def test_stacked_forward_equals_per_member_forward():
    net = PolicyNet(width=32, depth=2, value_head=False)
    mx.eval(net.parameters())
    shapes = net_shapes(net)
    theta = flatten_net(net)
    thetas = theta[None] + 0.1 * mx.random.normal((4, theta.shape[0]), key=mx.random.key(1))
    x = (mx.random.uniform(shape=(4, 16, 480), key=mx.random.key(2)) > 0.9).astype(mx.float32)
    stacked = stacked_forward(x, *stack_params(thetas, shapes))
    for p in range(4):
        load_into(net, thetas[p], shapes)
        assert mx.allclose(stacked[p], net(x[p])[0], atol=1e-4).item()
    load_into(net, theta, shapes)
    assert mx.allclose(flatten_net(net), theta).item()


def test_es_gradient_points_uphill():
    dim, half, sigma = 50, 500, 0.1
    target = mx.random.normal((dim,), key=mx.random.key(0))
    theta = mx.zeros((dim,))
    eps = mx.random.normal((half, dim), key=mx.random.key(1))
    members = mx.concatenate([theta + sigma * eps, theta - sigma * eps])
    fitness = -np.array(((members - target) ** 2).sum(-1))
    g = es_gradient(fitness, eps, sigma)
    true = 2 * (target - theta)
    cos = (g @ true) / (mx.linalg.norm(g) * mx.linalg.norm(true))
    assert cos.item() > 0.8


def test_stride4_gae_telescopes_with_lambda_one():
    """With lambda = 1 and gamma = 1: A_t + V_t = final reward whenever the episode ends inside the rollout."""
    from types import SimpleNamespace

    from bridgebid.trainers.ppo import PPOTrainer

    rng = np.random.default_rng(0)
    T, n = 24, 64
    ended = rng.random((T + 4, n)) < 0.12
    score = (rng.integers(-20, 20, (T + 4, n)) * 50).astype(np.int32)
    team = np.zeros((T + 4, n), np.int32)
    for i in range(n):  # seats rotate; a new deal starts after each end with a random dealer
        seat = rng.integers(4)
        for t in range(T + 4):
            team[t, i] = seat % 2
            seat = rng.integers(4) if ended[t, i] else seat + 1
    val = rng.normal(size=(T + 4, n)).astype(np.float32)
    trainer = SimpleNamespace(c=SimpleNamespace(rollout_len=T, reward_scale=1000.0, gae_lambda=1.0))
    data = {k: mx.array(v) for k, v in dict(ended=ended, score=score, team=team, val=val).items()}
    adv, ret = PPOTrainer.advantages(trainer, data)
    checked = 0
    for i in range(n):
        for t in range(T):
            e = t + int(np.argmax(ended[t:, i])) if ended[t:, i].any() else None
            if e is None:
                continue
            last_chain = t + 4 * ((e - t) // 4)
            if last_chain < T:
                sign = 1 if team[t, i] == 0 else -1
                assert abs(ret[t, i] - sign * score[e, i] / 1000.0) < 1e-4
                checked += 1
    assert checked > 200


@pytest.mark.slow
def test_ppo_resume_is_exact(tmp_path):
    if not Path("data/processed/train_keys.npy").exists():
        pytest.skip("dataset missing")
    from bridgebid.config import load_config
    from bridgebid.trainers.ppo import PPOTrainer

    over = ["ppo.num_envs=256", "ppo.rollout_len=8", "ppo.minibatch=512", "ppo.width=64", "ppo.critic_width=64",
            "ppo.total_steps=12288", "ppo.passive_until=3072", "ppo.passive_anneal=3072", "ppo.fsp_every=1",
            "ppo.eval_every=1000", "ppo.ckpt_every=1000", "ppo.eval_opponents=[pass]", "ppo.eval_boards=200",
            "train_limit=5000"]
    cfg = load_config("configs/ppo_base.yaml", over)
    a = PPOTrainer(cfg, run_dir=tmp_path / "a")
    a.train()
    b = PPOTrainer(cfg, run_dir=tmp_path / "b")
    b.train(max_updates=2)  # e.g. Ctrl-C or --hours: checkpoint mid-run
    b2 = PPOTrainer(cfg, run_dir=tmp_path / "b", resume=True)
    b2.train()
    assert a.update == b2.update == 4
    from mlx.utils import tree_flatten

    pa, pb = dict(tree_flatten(a.model.parameters())), dict(tree_flatten(b2.model.parameters()))
    assert all(mx.array_equal(pa[k], pb[k]).item() for k in pa)
