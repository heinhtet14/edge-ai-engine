# AGENTS.md — edge-ai-engine

Instructions for AI coding agents (Claude Code, Codex, Cursor, Copilot, …) working in this repo.
Human contributors: this is also the fastest way to learn the project's rules.

## What this project is

Run vision-language-action (VLA) robot policies on an **NVIDIA Jetson Orin Nano 8 GB** controlling an
**SO-ARM101** arm, then optimize and benchmark every step. Order: **SmolVLA → π0/π0.5 → world models**.

It is a **portfolio and learning project**: clear docs and reproducible numbers matter as much as code.
The owner is learning — explain trade-offs, don't just make choices silently.

## Read first

| File | What it gives you |
|---|---|
| [README.md](README.md) | Goals, hardware, architecture diagram, status |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Milestones and acceptance criteria |
| [docs/decisions/](docs/decisions/) | Accepted design decisions (ADRs) — do not contradict them without a new ADR |
| [docs/research/smolvla.md](docs/research/smolvla.md) | SmolVLA architecture, training, fine-tuning, dataset findings |
| [docs/research/tensorrt-inference.md](docs/research/tensorrt-inference.md) | Prior art and techniques for TensorRT VLA inference on Jetson |
| [docs/learning/](docs/learning/) | Plain-language explanations written for the owner |
| [docs/jetson-quirks.md](docs/jetson-quirks.md) | Every Jetson install gotcha hit so far |

## Architecture (agreed)

```
[TRAIN machine, Python]                         [JETSON, Python + C++]
LeRobot fine-tuning ─▶ export package ──copy──▶ C++ TensorRT inference (pybind11/IPC TBD)
  • ONNX per stage                                   ▲ observations     │ action chunks
  • quantized ONNX (ModelOpt)                        │                  ▼
  • golden test data                            Python control loop (LeRobot)
  • model_meta.json                               • motors, calibration, teleop
                                                  • SAFETY LAYER → SO-ARM101
```

- **Python** owns: training, dataset handling, offline eval, ONNX export + quantization, motor control,
  calibration, teleop, the control loop, the **safety layer**, and benchmark reporting/plots.
- **C++** owns: **inference only** — TensorRT engines (vision encoder, LM prefix, action-expert step),
  the denoising loop, CUDA graphs, GPU preprocessing, the async action-chunk queue, per-stage timing.
- **The boundary is a set of files** (the "export package"). C++ must not reimplement model logic by
  guesswork — everything it needs (normalization stats, image size, chunk size, denoise schedule,
  pre-tokenized task text) comes from `model_meta.json`.
- C++ inference **never commands motors directly**. All motion goes through the Python safety layer.
- TensorRT is the primary optimization path ([ADR-0001](docs/decisions/0001-tensorrt-as-primary-optimization-path.md)).

> Note: `docs/ROADMAP.md` milestones are being re-sequenced to add the export package and the C++
> runtime. Where the roadmap and this section disagree, **this section wins**.

## Machines

Tag every script, doc command, and CI job with where it runs:

- **[TRAIN]** — x86_64 Linux + NVIDIA GPU. Fine-tuning, offline eval, ONNX export, quantization.
  Reference: Ubuntu 22.04, RTX 2000 Ada 16 GB. Collaborators may differ — don't assume GPU size;
  make batch size etc. configurable.
- **[JETSON]** — Orin Nano 8 GB (aarch64). Arm control, TensorRT engine builds, inference, benchmarks.
  Reference: JetPack 6.2 (L4T 36.4, CUDA 12.6, TensorRT 10.3, Python 3.10).

Check `uname -m` (`aarch64` = Jetson) before running machine-specific steps. Never assume the stock
PyPI `torch` wheel works on the Jetson.

## Hard rules

1. **Safety first.** Never run anything that moves the arm unless (a) the same command passed with
   `--dry-run`, and (b) a human explicitly confirmed in the current session. Never weaken limits,
   the e-stop, or the watchdog to make something work.
2. **No hand-edited results.** Numbers in README, docs, or `site/` come only from files written by `bench/`.
3. **Parity before deployment.** Every exported/optimized model must match the FP32 PyTorch reference
   (golden test data) within tolerance on the robot's used action dims before it controls the arm.
4. **Pin versions.** Every dependency pinned. Record each Jetson install quirk in `docs/jetson-quirks.md`.
5. **Never commit** model weights, ONNX files, TensorRT engines, datasets, calibration caches, or
   tokens. TensorRT engines are device- and version-specific: always build them on the Jetson.
6. **Decisions need an ADR.** Before choosing a quantization method, export approach, or anything that
   changes the architecture above, write a short trade-off note in `docs/decisions/NNNN-title.md` and get
   the owner's approval.
7. **Check disk space** before large downloads (models, datasets, Docker images); never fill the disk.
8. **Stop for review** at the end of each milestone.

## Open decisions — do not decide these unilaterally

- Python ↔ C++ link: pybind11 module vs. shared-memory IPC.
- Exporter: write our own vs. build on [eetmie/jetson-orin-nano-vla](https://github.com/eetmie/jetson-orin-nano-vla).
- Phase B target: π0 vs. π0.5.
- JetPack: stay on 6.2 vs. upgrade to 7.2.x.
- Camera strategy for wrist-only hardware vs. two-camera public datasets (decided in M3).

If your task depends on one of these, ask the owner instead of picking.

## Data & evaluation

- **Public Hugging Face datasets only** — no own data collection. The real arm has **one wrist camera**.
- Accuracy for every config = offline action error on a fixed held-out split vs. the FP32 reference.
- Real-arm trials (N=10) only for the FP32 baseline and the best optimized config, clearly labeled.
- Benchmarks: fixed power mode (`nvpmodel` + `jetson_clocks`), warm-up, 60 s screening + paired 300 s
  runs, record versions + git SHA per run. See the method in
  [docs/learning/01-eetmie-optimization-notes.md](docs/learning/01-eetmie-optimization-notes.md).

## Repo layout

```
src/edge_ai_engine/   Python package (control loop, safety layer, export, bench reporting)
configs/              robot, camera, safety limits, training, benchmark configs
scripts/jetson/       [JETSON] setup, bring-up, engine builds, deploy
scripts/train/        [TRAIN] setup, fine-tuning, offline eval, export
bench/                single-command reproducible benchmark harness
results/              benchmark outputs (JSON/CSV) and generated tables/plots
docs/                 roadmap, ADRs, research, learning log, Jetson quirks
site/                 interactive project website (plain HTML/CSS/JS, no build step)
```

The C++ runtime directory will be added with its milestone; follow the layout in that milestone's ADR.

## Conventions

- **Python:** 3.10-compatible (JetPack system Python), type hints, small modules, configs over constants.
- **C++:** C++17, CMake, TensorRT 10 API (`enqueueV3`, named tensors), RAII for CUDA/TensorRT resources.
- **Docs:** keep `README.md` roadmap table, `docs/ROADMAP.md`, and `site/data/progress.json` in sync
  when a milestone changes status. New explanations for the owner go in `docs/learning/NN-topic.md`;
  external research goes in `docs/research/` with **every claim linked to its source**, and numbers
  marked as "reported" until reproduced here.
- **Website:** `site/` reads JSON from `site/data/`; preview with `python3 -m http.server -d site 8000`.
  `site/data/results.json` is written by the bench pipeline only.
- **Git:** conventional commit messages (`feat:`, `fix:`, `docs:`, `chore:` …), one logical change per
  commit. Use your own git identity. Never force-push `main`.
