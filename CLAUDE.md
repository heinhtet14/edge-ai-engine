# CLAUDE.md — edge-ai-engine

Context for Claude Code sessions working in this repo.

## Project
Run VLA policies (SmolVLA first, then pi0, then world models) on an NVIDIA Jetson Orin Nano 8 GB
controlling an SO-ARM101 arm; optimize and benchmark every step. Portfolio + learning project:
clarity of docs and reproducibility of numbers matter as much as the code.

## Machines
- **[TRAIN]** x86 workstation, Ubuntu 22.04, RTX 2000 Ada 16 GB. Fine-tuning, offline eval, ONNX export.
  Disk is nearly full — check free space before any download; never fill the disk.
- **[JETSON]** Orin Nano 8 GB, JetPack 6.2 (L4T 36.4, CUDA 12.6, TensorRT 10.3, Python 3.10).
  Bring-up, deployment, TensorRT engine builds, benchmarks. A separate Claude Code session runs there.
- Tag every script and README command with the machine it runs on. Check `uname -m`
  (`aarch64` = Jetson) before running machine-specific steps.

## Hard rules
- **Safety:** never run anything that moves the arm without (1) a passing `--dry-run` of the same
  command and (2) explicit user confirmation in this session. All motion goes through the safety layer.
- **No hand-edited results.** Numbers in README/site come only from files written by `bench/`.
- **Pin versions.** Every dependency pinned; record each Jetson install quirk in `docs/jetson-quirks.md`.
- **Never commit** model weights, ONNX files, TensorRT engines, datasets, or tokens (see `.gitignore`).
  TensorRT engines are device- and version-specific: always build them on the Jetson.
- Before choosing a quantization or export method, write a short trade-off note as an ADR in
  `docs/decisions/` and get user approval.
- Stop for user review at the end of each milestone (see `docs/ROADMAP.md`).

## Data & evaluation
- Public Hugging Face datasets only (no own data collection). Single wrist camera on the real arm.
- Accuracy for every config = offline action error on a fixed held-out split vs. the FP32 reference.
  Real-arm trials (N=10) only for the FP32 baseline and the best optimized config.

## Docs & website
- Keep `README.md` roadmap table, `docs/ROADMAP.md`, and `site/data/progress.json` in sync when a
  milestone changes status.
- `site/` is plain HTML/CSS/JS with no build step; it reads JSON from `site/data/`.
  Benchmark results will be copied to `site/data/results.json` by the bench pipeline.

## Git
- Commit identity is set in the local repo config (GitHub noreply). Don't change it.
- Conventional, descriptive commit messages; one logical change per commit.
