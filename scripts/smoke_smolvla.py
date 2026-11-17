#!/usr/bin/env python3
"""[TRAIN] [JETSON] SmolVLA smoke test: load a checkpoint and produce one action chunk from dummy inputs.

    python scripts/smoke_smolvla.py                       # lerobot/smolvla_base on CUDA
    python scripts/smoke_smolvla.py --precision bf16-amp --iters 10

Precision modes:
  stock     as LeRobot loads it: VLM + expert in bf16, small projections fp32 (LeRobot 0.7.0 default)
  fp32      every weight cast to fp32 — the true reference for accuracy comparisons
  bf16-amp  fp32 weights + torch.autocast(bf16)
  fp16-amp  fp32 weights + torch.autocast(fp16)

Proves the install works end to end (download, processors, tokenizer, forward pass). Inputs are
random, so the actions are meaningless — this is NOT a benchmark (see bench/ for that).
Never touches the arm.
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

AMP = {"bf16-amp": torch.bfloat16, "fp16-amp": torch.float16}
PRECISIONS = ["stock", "fp32", *AMP]


def dummy_observation(policy: SmolVLAPolicy, task: str, seed: int) -> dict:
    """Random observation matching the checkpoint's declared input features (unbatched, like a robot)."""
    g = torch.Generator().manual_seed(seed)
    obs: dict = {"task": task}
    for key, feat in policy.config.input_features.items():
        shape = tuple(feat.shape)
        if feat.type.name == "VISUAL":
            obs[key] = torch.rand(shape, generator=g)  # float image in [0, 1], CHW
        else:
            obs[key] = torch.randn(shape, generator=g)
    return obs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="lerobot/smolvla_base")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--precision", choices=PRECISIONS, default="fp32")
    ap.add_argument("--iters", type=int, default=5, help="timed chunk predictions after 1 warm-up")
    ap.add_argument("--task", default="Pick up the cube and place it in the bowl.")
    args = ap.parse_args()

    t0 = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(args.model)
    # LeRobot 0.7.0 loads the VLM and expert in bf16 by default ("stock"). Casting the whole model to
    # bf16/fp16 with .to(dtype) fails (internal tensors are created in fp32), so reduced precision
    # uses fp32 weights + torch.autocast instead.
    if args.precision == "stock":
        policy.to(device=args.device)
    else:
        policy.to(device=args.device, dtype=torch.float32)
    policy.eval()
    amp = args.precision in AMP
    pre, post = make_pre_post_processors(policy.config, pretrained_path=args.model,
                                         preprocessor_overrides={"device_processor": {"device": args.device}})
    print(f"loaded {args.model} in {time.perf_counter() - t0:.1f}s on {args.device} "
          f"(precision={args.precision})")
    n_params = sum(p.numel() for p in policy.parameters())
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    dtypes: dict = {}
    for prm in policy.parameters():
        dtypes[str(prm.dtype).replace("torch.", "")] = dtypes.get(str(prm.dtype).replace("torch.", ""), 0) + prm.numel()
    print(f"params: {n_params / 1e6:.0f}M total, {n_train / 1e6:.0f}M trainable, by dtype: "
          + ", ".join(f"{k} {v / 1e6:.0f}M" for k, v in dtypes.items()))
    print("inputs:", {k: tuple(v.shape) for k, v in policy.config.input_features.items()})
    print("output:", {k: tuple(v.shape) for k, v in policy.config.output_features.items()},
          f"chunk_size={policy.config.chunk_size} denoise_steps={policy.config.num_steps}")

    def one_chunk(seed: int) -> torch.Tensor:
        batch = pre(dummy_observation(policy, args.task, seed))
        # Seeded noise so runs (and dtypes) are comparable.
        cfg = policy.config
        noise = torch.randn((1, cfg.chunk_size, cfg.max_action_dim),
                            generator=torch.Generator().manual_seed(seed)).to(args.device)
        with torch.autocast(device_type=args.device.split(":")[0], dtype=AMP.get(args.precision, torch.bfloat16), enabled=amp):
            chunk = policy.predict_action_chunk(batch, noise=noise)
        return post(chunk.float())

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()  # peak includes the resident weights
    chunk = one_chunk(0)  # warm-up
    times = []
    for i in range(args.iters):
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        t = time.perf_counter()
        chunk = one_chunk(i + 1)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1e3)

    print(f"action chunk: shape={tuple(chunk.shape)} dtype={chunk.dtype} finite={bool(torch.isfinite(chunk).all())}")
    print(f"first action: {[round(x, 3) for x in chunk[0, 0].float().tolist()]}")
    if times:
        print(f"chunk latency (indicative only): median {statistics.median(times):.1f} ms over {len(times)} iters")
    if args.device.startswith("cuda"):
        print(f"peak CUDA memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    ok = bool(torch.isfinite(chunk).all())
    print("SMOKE TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
