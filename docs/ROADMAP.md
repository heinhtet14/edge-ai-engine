# Roadmap

Each milestone ends with a review stop. Machine tags: **[TRAIN]** workstation, **[JETSON]** Orin Nano.

## Phase A — SmolVLA

### M0 · Repo + environment — [TRAIN] [JETSON]
- Repo layout, README, CLAUDE.md, website skeleton ✅
- `scripts/train/setup.sh`, `scripts/jetson/setup.sh` with pinned versions (JetPack-compatible PyTorch wheels)
- `check_env.py`: CUDA, cuDNN, TensorRT, torch device, camera enumeration, free-disk guard
- **Done when:** a dummy SmolVLA forward pass runs on both machines.

### M1 · Safety layer — [JETSON]
- Joint position/velocity/torque limits from config, per-step action-delta clamp, workspace bounds
- Keyboard e-stop (torque off), watchdog (stale frame/policy → hold), `--dry-run` mode
- Wiring notes for a physical power-cut switch
- **Done when:** dry-run streams actions with the arm unpowered; e-stop latency measured.

### M2 · Arm bring-up — [JETSON]
- Motor ID/baud setup, leader + follower calibration (files versioned), teleop test, wrist camera check
- **Done when:** smooth teleop at 30 Hz with camera preview.

### M3 · Public dataset selection + visualization — [TRAIN]
- Shortlist public SO-101 datasets (prefer wrist-only; otherwise handle the missing view)
- Inspect, visualize, safely replay on the arm, fix the held-out split
- **Done when:** dataset choice approved with evidence.

### M4 · Fine-tune SmolVLA — [TRAIN]
- Config-driven BF16 fine-tuning, checkpoints, TensorBoard logs, offline eval
- Zero-shot `smolvla_base` reference
- **Done when:** validation loss converges; offline action error recorded.

### M5 · Deploy — [JETSON]
- Sync and async/chunked control loops with per-stage timing
- **Done when:** FP32 policy runs closed-loop through the safety layer.

### M6 · Benchmark harness — [JETSON]
- `bench/run_all.sh`: latency p50/p95 per stage + per chunk, control Hz, peak memory,
  tegrastats power, offline accuracy → JSON → CSV/markdown table + plots + `site/data/results.json`
- Fixed power mode, warm-up, versions + git SHA recorded per run
- **Done when:** one command reproduces the FP32 row from a clean checkout.

### M7 · Optimization ladder — [JETSON]
PyTorch FP32 → PyTorch FP16/BF16 → **TensorRT FP16** → **TensorRT INT8 (mixed precision)** →
INT4 (TensorRT Model Optimizer, if practical) → ONNX Runtime + TensorRT backend (comparison) →
chunk-size × denoising-step sweeps on the best engine. An ADR precedes each method choice.

### M8 · Write-up A
Results table, accuracy-vs-speed plot, real-arm trials (N=10) for FP32 and best config.

## Phase B — pi0

### M9 · pi0 on the Orin Nano
Memory budget analysis → public checkpoint (or 4-bit LoRA locally if it fits 16 GB) →
INT8/INT4 weight quantization → TensorRT → same control loop + harness → feasibility verdict with numbers.

## Phase C — World models (if time)

### M10 · Scoping
Present options that fit local compute (e.g. action-conditioned world model as offline evaluator,
synthetic demo generation); user chooses before anything is built.
