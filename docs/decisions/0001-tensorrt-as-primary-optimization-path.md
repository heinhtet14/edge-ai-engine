# ADR-0001: TensorRT as the primary optimization path

- **Status:** Accepted (2026-11-09)
- **Context milestone:** M7

## Context
On the Jetson Orin Nano, plain PyTorch FP16 leaves most of the available speedup on the table.
TensorRT (shipped with JetPack 6.2 as TensorRT 10.3) provides kernel fusion, tuned FP16/INT8
kernels, and lower memory use — critical on 8 GB of memory shared between CPU and GPU.

VLA policies such as SmolVLA and pi0 do not export as a single graph: a vision-language prefix
runs once per chunk and produces a KV cache, then a flow-matching action expert loops ~10 times.

## Options considered
| Path | Effort | Notes |
|---|---|---|
| ONNX → TensorRT engine (`trtexec` / Python API) | High | Best performance; per-layer precision control |
| ONNX Runtime + TensorRT execution provider | Medium | Falls back to CUDA for unsupported ops |
| Torch-TensorRT | Low | Fastest to try; Jetson wheel support is unreliable |

## Decision
Use **ONNX → TensorRT engines** as the main path, with ONNX Runtime + TensorRT EP as a fallback
and comparison point. Export three separate engines:
1. vision encoder
2. language-model prefix (outputs KV cache)
3. one action-expert denoising step (looped from Python, later CUDA graphs)

## Consequences
- Per-stage timing in benchmarks; mixed precision possible (e.g. INT8 vision encoder, FP16 action expert).
- Engines must be built on the Jetson; they are never committed.
- INT8 requires a calibration set (a few hundred frames from the held-out split).
- INT4 for pi0 to be validated against TensorRT 10.3 on Orin before committing (separate ADR).
