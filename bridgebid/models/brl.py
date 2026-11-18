"""Download and convert the published brl checkpoints (Kita et al. 2024, Apache-2.0) to MLX safetensors.

Input:  models/brl/raw/<name>.pkl  (download from github.com/harukaki/brl/tree/main/bridge_models)
Output: models/brl/<name>.safetensors + .json

The pickles hold haiku parameter dicts of jax arrays. They are read with a *restricted* unpickler that can
only rebuild numpy arrays; any other global in the file raises an error, so no foreign code runs.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

RAW = Path("models/brl/raw")
OUT = Path("models/brl")
NAMES = ["model-sl", "model-from-scratch-rl", "model-pretrained-rl", "model-pretrained-rl-with-fsp",
         "model-pretrained-rl-with-pfsp"]
# Published duplicate IMPs/board vs WBridge5 (Kita et al. 2024; 'from-scratch' from the paper's ablation).
VS_WBRIDGE5 = {"model-sl": -0.56, "model-from-scratch-rl": -0.64, "model-pretrained-rl": 0.88,
               "model-pretrained-rl-with-fsp": 1.24, "model-pretrained-rl-with-pfsp": None}


def _numpy_reconstruct():
    try:
        from numpy._core.multiarray import _reconstruct
    except ImportError:  # numpy < 2
        from numpy.core.multiarray import _reconstruct
    return _reconstruct


def _reconstruct_array(fun, args, arr_state, aval_state):
    arr = fun(*args)
    arr.__setstate__(arr_state)
    return np.asarray(arr)


class SafeUnpickler(pickle.Unpickler):
    ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"): _numpy_reconstruct(),
        ("numpy._core.multiarray", "_reconstruct"): _numpy_reconstruct(),
        ("numpy", "ndarray"): np.ndarray,
        ("numpy", "dtype"): np.dtype,
        ("jax._src.array", "_reconstruct_array"): _reconstruct_array,
    }

    def find_class(self, module, name):
        try:
            return self.ALLOWED[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(f"refusing to load {module}.{name}") from None


def load_brl_params(path: Path) -> dict[str, dict[str, np.ndarray]]:
    with open(path, "rb") as f:
        params = SafeUnpickler(f).load()
    expected = {f"actor_critic/linear{'' if i == 0 else f'_{i}'}" for i in range(6)}
    assert set(params) == expected, sorted(params)
    return params


def brl_forward_numpy(params, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = obs.astype(np.float32)
    for i in range(4):
        p = params[f"actor_critic/linear{'' if i == 0 else f'_{i}'}"]
        x = np.maximum(x @ p["w"] + p["b"], 0)
    logits = x @ params["actor_critic/linear_4"]["w"] + params["actor_critic/linear_4"]["b"]
    value = (x @ params["actor_critic/linear_5"]["w"] + params["actor_critic/linear_5"]["b"])[:, 0]
    return logits, value


def to_mlx(params):
    import mlx.core as mx

    from bridgebid.models.actor_critic import PolicyNet

    model = PolicyNet(width=1024, depth=4, value_head=True)
    names = [f"actor_critic/linear{'' if i == 0 else f'_{i}'}" for i in range(6)]
    for layer, name in zip(model.torso.layers, names[:4], strict=True):
        layer.weight = mx.array(params[name]["w"].T)
        layer.bias = mx.array(params[name]["b"])
    model.pi.weight, model.pi.bias = mx.array(params[names[4]]["w"].T), mx.array(params[names[4]]["b"])
    model.v.weight, model.v.bias = mx.array(params[names[5]]["w"].T), mx.array(params[names[5]]["b"])
    mx.eval(model.parameters())
    return model


BASE_URL = "https://raw.githubusercontent.com/harukaki/brl/main/bridge_models"


def download() -> None:
    import urllib.request

    RAW.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        dst = RAW / f"{name}.pkl"
        if not dst.exists():
            print(f"downloading {name}.pkl (~15 MB) ...")
            urllib.request.urlretrieve(f"{BASE_URL}/{name}.pkl", dst)


def main() -> None:
    import mlx.core as mx

    from bridgebid.models.io import save_policy

    rng = np.random.default_rng(0)
    probe = (rng.random((2048, 480)) < 0.05).astype(np.float32)
    for name in NAMES:
        src = RAW / f"{name}.pkl"
        if not src.exists():
            print(f"skip {name}: {src} missing")
            continue
        params = load_brl_params(src)
        model = to_mlx(params)
        ref_logits, ref_value = brl_forward_numpy(params, probe)
        logits, value = model(mx.array(probe))
        err = max(np.abs(np.array(logits) - ref_logits).max(), np.abs(np.array(value) - ref_value).max())
        assert err < 1e-3, f"{name}: port mismatch {err}"
        save_policy(model, OUT / name, meta={"source": "github.com/harukaki/brl (Apache-2.0)",
                                             "vs_wbridge5_imps": VS_WBRIDGE5[name]})
        print(f"{name}: ported (max |diff| vs numpy forward = {err:.2e})")


if __name__ == "__main__":
    download()
    main()
