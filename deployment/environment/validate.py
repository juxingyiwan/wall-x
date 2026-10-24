"""Check the shared environment without opening cameras, CAN or robot devices."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import json
import subprocess
import sys
import zipfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--lerobot-wheel", type=Path, help="Optional pristine reference wheel")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "deployment/client/src"))
    report = {"prefix": sys.prefix, "python": sys.version, "checks": {}, "errors": []}

    def check(name, operation):
        try:
            report["checks"][name] = operation()
        except Exception as exc:
            report["errors"].append(f"{name}: {type(exc).__name__}: {exc}")

    expected = {
        "torch": "2.10.0",
        "transformers": "4.57.6",
        "huggingface-hub": "0.35.3",
        "lerobot": "0.4.4",
        "numpy": "2.2.6",
        "rerun-sdk": "0.26.2",
        "lerobot-robot-seeed-b601": "1.0.4",
    }

    def versions():
        actual = {name: metadata.version(name) for name in expected}
        assert actual == expected, actual
        assert sys.version_info[:2] == (3, 10), sys.version
        return actual

    check("versions", versions)

    def pip_check():
        result = subprocess.run(
            [sys.executable, "-m", "pip", "check"], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout.strip()

    check("pip_check", pip_check)

    def editable_packages():
        editable = []
        for dist in metadata.distributions():
            direct = json.loads(dist.read_text("direct_url.json") or "{}")
            if direct.get("dir_info", {}).get("editable"):
                editable.append(dist.metadata["Name"])
        assert not editable, editable
        return "none"

    check("editable_packages", editable_packages)

    def pristine_lerobot():
        wheel = args.lerobot_wheel
        dist = metadata.distribution("lerobot")
        count = 0
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                if name.startswith("lerobot/") and name.endswith(".py"):
                    actual = Path(dist.locate_file(name)).read_bytes()
                    assert actual == archive.read(name), name
                    count += 1
        return {
            "python_files": count,
            "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        }

    if args.lerobot_wheel:
        check("pristine_lerobot", pristine_lerobot)
    else:
        report["checks"]["pristine_lerobot"] = "not checked: pass --lerobot-wheel"
    for name in (
        "numpy",
        "cv2",
        "numba",
        "scipy",
        "pyarrow",
        "av",
        "torchcodec",
        "flash_attn",
        "peft",
        "rerun",
        "lerobot_robot_seeed_b601",
        "wallx_client.cli",
        "wall_x.model.qact.qwen2_5.modeling_qwen2_5_vl_act",
        "wall_x._vendor.harrix.serving.policy.wall_x_policy",
    ):
        check(name, lambda name=name: str(importlib.import_module(name).__file__))

    def extension():
        from wall_x.model.core.ops._cuda_ext import load

        return str(load().__file__)

    check("wallx_cuda_extension", extension)

    def cuda_device():
        import torch

        available = torch.cuda.is_available()
        if args.require_cuda:
            assert available, "CUDA is required for this validation"
        if not available:
            return {"available": False, "execution": "not tested"}
        from wall_x.model.core.ops import rmsnorm

        # The upstream export has no Rmsnorm CUDA wrapper. Its ordinary PyTorch
        # fallback is supported; test an actually exported CUDA op separately.
        return {
            "available": True,
            "name": torch.cuda.get_device_name(),
            "rmsnorm_backend": rmsnorm.backend,
        }

    def cuda_rope():
        import torch

        from wall_x.model.core.ops import rope

        torch.manual_seed(7)
        cases = []
        for dtype, tolerance in [(torch.float16, 3e-3), (torch.bfloat16, 2e-2)]:
            for interleave in (False, True):
                q, k = [torch.randn(1, 32, 4, 64, device="cuda", dtype=dtype) for _ in range(2)]
                angles = torch.randn(1, 32, 32, device="cuda", dtype=torch.float32)
                cos, sin = angles.cos(), angles.sin()
                reference = rope.call_with_backend("pytorch", q, k, cos, sin, interleave)
                # The CUDA wrapper updates q/k in place. Independent copies are
                # essential; comparing against already-mutated inputs is invalid.
                actual = rope.call_with_backend(
                    "cuda_inline", q.clone(), k.clone(), cos, sin, interleave, inference=True
                )
                for got, wanted in zip(actual, reference, strict=True):
                    torch.testing.assert_close(got, wanted, rtol=tolerance, atol=tolerance)
                cases.append({"dtype": str(dtype), "interleave": interleave})
        torch.cuda.synchronize()
        return {"backend": "cuda_inline", "cases": cases, "passed": True}

    def cuda_flash_attention():
        import torch
        from flash_attn import flash_attn_func
        from torch.nn.attention import SDPBackend, sdpa_kernel

        torch.manual_seed(7)
        q, k, v = [torch.randn(1, 32, 4, 64, device="cuda", dtype=torch.float16) for _ in range(3)]
        actual = flash_attn_func(q, k, v)
        # Force an independent mathematical reference, not another fused backend.
        with sdpa_kernel(SDPBackend.MATH):
            reference = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            ).transpose(1, 2)
        torch.testing.assert_close(actual, reference, rtol=3e-3, atol=3e-3)
        torch.cuda.synchronize()
        return {"reference": "SDPBackend.MATH", "passed": True}

    check("cuda_device", cuda_device)
    if report["checks"].get("cuda_device", {}).get("available"):
        check("cuda_rope", cuda_rope)
        check("cuda_flash_attention", cuda_flash_attention)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return int(bool(report["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
