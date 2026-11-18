"""Throughput benchmark for the shapes this project uses, on the MLX GPU (and CPU for comparison).

Run:  uv run python scripts/bench_mlx.py [--quick]

Measures
  1. ES population forward: P stacked MLPs (480-256-256-38) on B boards each, one batched matmul per layer.
  2. Policy forward (4x512, 4x1024) at large batch: env-step inference cost.
  3. Forward + backward + Adam at PPO minibatch sizes: training cost.
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

OBS, ACT = 480, 38


def timeit(fn, reps: int) -> float:
    fn()  # warm-up (and compile)
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / reps


def bench_population(dtype, quick: bool) -> list[tuple]:
    rows = []
    sizes = [480, 256, 256, ACT]
    for P in ([128, 512] if quick else [128, 256, 512]):
        for B in [1024, 4096]:
            ws = [(mx.random.normal((P, i, o)) * 0.05).astype(dtype) for i, o in zip(sizes[:-1], sizes[1:])]
            x = (mx.random.uniform(shape=(P, B, OBS)) > 0.9).astype(dtype)

            @mx.compile
            def fwd(x, *ws):
                h = x
                for w in ws[:-1]:
                    h = mx.maximum(h @ w, 0)
                return mx.argmax(h @ ws[-1], axis=-1)

            def run():
                mx.eval(fwd(x, *ws))

            dt = timeit(run, 5)
            rows.append(("ES pop fwd", f"P={P} B={B}", str(dtype).split(".")[-1], dt * 1e3, P * B / dt))
    return rows


class MLP(nn.Module):
    def __init__(self, width: int, depth: int):
        super().__init__()
        dims = [OBS] + [width] * depth
        self.layers = [nn.Linear(i, o) for i, o in zip(dims[:-1], dims[1:])]
        self.head = nn.Linear(width, ACT)

    def __call__(self, x):
        for layer in self.layers:
            x = nn.relu(layer(x))
        return self.head(x)


def bench_policy(quick: bool) -> list[tuple]:
    rows = []
    for width in [512, 1024]:
        model = MLP(width, 4)
        mx.eval(model.parameters())
        fwd = mx.compile(model)
        for B in ([8192, 65536] if quick else [8192, 32768, 65536]):
            x = (mx.random.uniform(shape=(B, OBS)) > 0.9).astype(mx.float32)
            dt = timeit(lambda: mx.eval(fwd(x)), 10)
            rows.append(("policy fwd", f"4x{width} B={B}", "float32", dt * 1e3, B / dt))

        opt = optim.Adam(learning_rate=3e-4)

        def loss_fn(m, x, a):
            logits = m(x)
            return nn.losses.cross_entropy(logits, a, reduction="mean")

        loss_and_grad = nn.value_and_grad(model, loss_fn)
        state = [model.state, opt.state]

        @mx.compile
        def train_step(x, a):
            loss, grads = loss_and_grad(model, x, a)
            opt.update(model, grads)
            return loss

        train_step = mx.compile(train_step, inputs=state, outputs=state)
        for B in ([16384] if quick else [8192, 16384, 32768]):
            x = (mx.random.uniform(shape=(B, OBS)) > 0.9).astype(mx.float32)
            a = mx.random.randint(0, ACT, (B,))
            dt = timeit(lambda: mx.eval(train_step(x, a), state), 10)
            rows.append(("fwd+bwd+Adam", f"4x{width} B={B}", "float32", dt * 1e3, B / dt))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="also run the ES population benchmark on CPU")
    args = ap.parse_args()

    print(f"MLX {mx.__version__}  device: {mx.device_info().get('device_name', '?')}")
    rows = []
    rows += bench_population(mx.float32, args.quick)
    rows += bench_population(mx.float16, args.quick)
    if args.cpu:
        with mx.stream(mx.cpu):
            rows += [(r[0] + " [CPU]", *r[1:]) for r in bench_population(mx.float32, True)]
    rows += bench_policy(args.quick)

    print(f"\n{'benchmark':<22}{'shape':<22}{'dtype':<10}{'ms/call':>10}{'rows/s':>14}")
    for name, shape, dtype, ms, thr in rows:
        print(f"{name:<22}{shape:<22}{dtype:<10}{ms:>10.2f}{thr / 1e6:>12.2f} M")


if __name__ == "__main__":
    main()
