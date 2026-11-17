#!/usr/bin/env python3
"""[TRAIN] [JETSON] Environment check for edge-ai-engine.

    python scripts/check_env.py --role train
    python scripts/check_env.py --role jetson
    python scripts/check_env.py --role train --json results/env/train.json

Prints PASS/WARN/FAIL per check and exits non-zero if any check FAILs.
Read-only: never downloads models, never touches the arm.
"""
from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Check:
    name: str
    status: str  # PASS | WARN | FAIL
    detail: str


def _run(cmd: List[str]) -> Optional[str]:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def check_platform(role: str) -> Check:
    machine = platform.machine()
    expected = "aarch64" if role == "jetson" else "x86_64"
    ok = machine == expected
    return Check("platform", "PASS" if ok else "FAIL", f"{machine} (expected {expected} for role={role})")


def check_python(role: str) -> Check:
    v = sys.version_info
    want = (3, 12) if role == "train" else (3, 10)
    ok = (v.major, v.minor) >= want
    return Check("python", "PASS" if ok else "FAIL", f"{v.major}.{v.minor}.{v.micro} (need >= {want[0]}.{want[1]})")


def check_disk(min_gb: float) -> Check:
    free = shutil.disk_usage(REPO_ROOT).free / 1e9
    status = "PASS" if free >= min_gb else ("WARN" if free >= min_gb / 3 else "FAIL")
    return Check("disk_free", status, f"{free:.1f} GB free at {REPO_ROOT} (want >= {min_gb:.0f} GB)")


def check_torch() -> List[Check]:
    try:
        import torch
    except ImportError as e:
        return [Check("torch", "FAIL", f"not importable: {e}")]
    out = [Check("torch", "PASS", f"{torch.__version__} (CUDA build {torch.version.cuda})")]
    if not torch.cuda.is_available():
        out.append(Check("cuda", "FAIL", "torch.cuda.is_available() is False"))
        return out
    props = torch.cuda.get_device_properties(0)
    out.append(Check("cuda", "PASS", f"{props.name}, {props.total_memory / 2**30:.1f} GiB, SM{props.major}{props.minor}"))
    # A real kernel launch catches driver/runtime mismatches that is_available() misses.
    try:
        a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()
        val = float((a @ a).float().abs().mean())
        out.append(Check("cuda_matmul", "PASS", f"fp16 matmul ok (mean |x| = {val:.2f})"))
    except Exception as e:  # noqa: BLE001
        out.append(Check("cuda_matmul", "FAIL", repr(e)))
    out.append(Check("bf16", "PASS" if torch.cuda.is_bf16_supported() else "WARN",
                     f"bf16 supported: {torch.cuda.is_bf16_supported()}"))
    return out


def check_module(name: str, required: bool = True) -> Check:
    try:
        mod = importlib.import_module(name)
        return Check(name, "PASS", getattr(mod, "__version__", "importable"))
    except Exception as e:  # noqa: BLE001
        return Check(name, "FAIL" if required else "WARN", f"not importable: {e}")


def check_hf_login() -> Check:
    try:
        from huggingface_hub import get_token, whoami
    except ImportError:
        return Check("hf_login", "WARN", "huggingface_hub not installed")
    if not get_token():
        return Check("hf_login", "WARN", "not logged in — run `hf auth login` (needed for gated repos)")
    try:
        return Check("hf_login", "PASS", f"logged in as {whoami()['name']}")
    except Exception as e:  # noqa: BLE001
        return Check("hf_login", "WARN", f"token present but whoami failed: {e}")


def check_ffmpeg() -> Check:
    path = shutil.which("ffmpeg")
    if not path:
        return Check("ffmpeg", "WARN", "not found — dataset video decoding may fail")
    first = (_run(["ffmpeg", "-version"]) or "").splitlines()[:1]
    return Check("ffmpeg", "PASS", first[0] if first else path)


def check_nvidia_smi() -> Check:
    out = _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    return Check("nvidia_driver", "PASS" if out else "FAIL", out or "nvidia-smi failed")


def check_jetson_release() -> Check:
    p = Path("/etc/nv_tegra_release")
    if not p.exists():
        return Check("l4t", "FAIL", "/etc/nv_tegra_release missing — not a Jetson?")
    return Check("l4t", "PASS", p.read_text().splitlines()[0].strip())


def check_tensorrt() -> Check:
    try:
        import tensorrt as trt
        return Check("tensorrt", "PASS", trt.__version__)
    except ImportError:
        trtexec = shutil.which("trtexec") or ("/usr/src/tensorrt/bin/trtexec"
                                               if Path("/usr/src/tensorrt/bin/trtexec").exists() else None)
        if trtexec:
            return Check("tensorrt", "PASS", f"python module missing, trtexec at {trtexec}")
        return Check("tensorrt", "FAIL", "neither python tensorrt nor trtexec found")


def check_cxx_toolchain() -> List[Check]:
    out = []
    for tool in ("g++", "cmake"):
        ver = _run([tool, "--version"])
        out.append(Check(tool, "PASS" if ver else "FAIL", ver.splitlines()[0] if ver else "not found"))
    nvcc = shutil.which("nvcc") or ("/usr/local/cuda/bin/nvcc" if Path("/usr/local/cuda/bin/nvcc").exists() else None)
    out.append(Check("nvcc", "PASS" if nvcc else "FAIL", nvcc or "not found (add /usr/local/cuda/bin to PATH)"))
    return out


def run(role: str, min_disk_gb: float) -> List[Check]:
    checks = [check_platform(role), check_python(role), check_disk(min_disk_gb)]
    if role == "train":
        checks.append(check_nvidia_smi())
        checks += check_torch()
        checks += [check_module("lerobot"), check_module("transformers"), check_module("num2words"),
                   check_ffmpeg(), check_hf_login()]
    else:
        checks += [check_jetson_release(), check_tensorrt(), *check_cxx_toolchain(),
                   check_module("lerobot", required=False), check_hf_login()]
    return checks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", choices=["train", "jetson"], required=True)
    ap.add_argument("--min-disk-gb", type=float, default=None,
                    help="free-space threshold (default: 50 train, 20 jetson)")
    ap.add_argument("--json", type=Path, help="also write results to this JSON file")
    args = ap.parse_args()
    min_gb = args.min_disk_gb if args.min_disk_gb is not None else (50 if args.role == "train" else 20)

    checks = run(args.role, min_gb)
    width = max(len(c.name) for c in checks)
    for c in checks:
        print(f"[{c.status:4}] {c.name:<{width}}  {c.detail}")
    n_fail = sum(c.status == "FAIL" for c in checks)
    n_warn = sum(c.status == "WARN" for c in checks)
    print(f"\n{len(checks)} checks: {n_fail} FAIL, {n_warn} WARN")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"role": args.role, "python": sys.version,
                                         "checks": [asdict(c) for c in checks]}, indent=2))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
