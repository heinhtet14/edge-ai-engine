# Research: TensorRT inference for VLAs on the Jetson Orin Nano

*Compiled 2026-11-09. Numbers are as reported by each source and have not been reproduced by this project yet.*

## Key takeaways

1. Prior open-source work already runs SmolVLA **and π0.5** with TensorRT on an 8 GB Orin Nano.
2. **π0.5 fits in FP16 (~6.4 GB)** — quantization is an optimization for speed and memory headroom, not a requirement to fit.
3. **No published INT8/INT4 VLA results on the Orin Nano.** The closest prior work is explicitly FP16-only. That gap is this project's niche.
4. The Orin GPU (SM87, Ampere) has **no FP8**, so NVIDIA's Thor FP8/NVFP4 recipes don't transfer; INT8 and INT4 are the options.
5. Those Orin Nano results ran on **JetPack 7.2.1** (TensorRT 10.16, CUDA 13), not JetPack 6.2 (TensorRT 10.3).

## Reported numbers

### Orin Nano 8 GB
| Model | Runtime | p50 | Memory | Source |
|---|---|---|---|---|
| SmolVLA (2 cams, 10 steps, chunk 50) | TensorRT FP16, split engines + CUDA graph + Triton kernels | ~105 ms (9.3 Hz) | 1.78 GB | [eetmie/jetson-orin-nano-vla][eetmie] |
| SmolVLA | ONNX Runtime + TensorRT EP (older) | ~190 ms (5.25 Hz) | 1.87 GB | [eetmie/smolvla-base-onnx][eetmie-hf] |
| π0.5 LIBERO (3.3B) | TensorRT FP16 | ~438 ms (2.3 Hz) | 6.39 GB | [eetmie/jetson-orin-nano-vla][eetmie] |
| π0.5 base (3.3B) | TensorRT FP16 | ~643 ms (1.55 Hz) | 6.18 GB | [eetmie/jetson-orin-nano-vla][eetmie] |
| SmolVLA | ONNX Runtime + TensorRT FP16 / INT8 | ~25 / ~20 ms | — | [jared-hpc][jared] — method unclear, treat as unverified |

### Other boards (context)
| Board | Model | Runtime | p50 | Source |
|---|---|---|---|---|
| AGX Orin 32 GB | SmolVLA | PyTorch, MAXN | 739 ms | [furuya02][furuya] |
| AGX Orin 64 GB | π0.5 (2 cams) | BF16 → INT8 W8A8 custom CUDA (FlashRT) | 193 → 124 ms | [FlashRT][flashrt] |
| AGX Orin | π0.5 | PyTorch eager → torch.compile | 949 → 394 ms | [NVIDIA forum][forum] |
| Thor | π0.5 | PyTorch BF16 → TensorRT FP8 → FP8+NVFP4 | 132 → 54 → 49 ms | [Jetson AI Lab][thor] |

## Techniques that matter

- **Split engines** (vision → LM prefix → denoise step) and **replay the chain as one CUDA graph**. Monolithic engines don't build in 8 GB for the larger models. CUDA graphs on the denoise session alone: 3–3.8× ([jared-hpc][jared]).
- **Hoist observation-only work** out of the denoising loop (cross-attention K/V).
- **Clean the export**: e.g. LeRobot's `apply_rope` exports as `ScatterND`; rewriting with `torch.cat` saved ~7.5 ms.
- **FP32 accumulation**: TensorRT often picks FP16-accumulating GEMMs; forcing FP32 accumulation cost ~2 % speed but cut SmolVLA action error ~31 %.
- **GPU preprocessing**: SmolVLA preprocessing 3.9 → 0.5 ms.
- **Mandatory parity checks** against the stock policy. One π0.5 exporter had six silent bugs ([jared-hpc][jared]); an open LeRobot issue reports wrong SmolVLA actions on Orin with JetPack 6.2 + torch 2.5 ([#3636][issue]).

Detailed walkthrough: [docs/learning/01-eetmie-optimization-notes.md](../learning/01-eetmie-optimization-notes.md).

## Quantization on Orin

- **INT8**: NVIDIA TensorRT Model Optimizer inserts Q/DQ nodes into the ONNX file in Python; TensorRT builds the engine on the device ([ModelOpt ONNX PTQ][modelopt], [support matrix][modelopt-matrix]). Fits our "Python exports, C++ runs" split.
- **INT4**: ModelOpt's own README warns INT4 TensorRT engines are **not yet faster than FP16**. Expect memory savings, not speed.
- FlashRT reached INT8 W8A8 on AGX Orin with custom CUTLASS kernels ([FlashRT][flashrt]).

## Python control ↔ C++ inference options

| Option | Notes |
|---|---|
| **pybind11 module** | One process; C++ releases the GIL and runs its own async inference thread. Microsecond-level call overhead. |
| Shared-memory IPC | Separate C++ process, actions via `/dev/shm`; isolates crashes (see [LITHE][lithe]). More plumbing. |
| gRPC / ZeroMQ | Most flexible, slowest, most code. |

## Reading list (suggested order)

1. [eetmie/jetson-orin-nano-vla][eetmie] + [optimization notes][eetmie-opt] — closest prior work.
2. [Deploying VLA models on Jetson — what fits, what TensorRT buys you][jared]
3. [FlashRT Orin deployment guide][flashrt] — C++/CUDA VLA runtime, INT8 on Orin.
4. [OpenPi π0.5 on Jetson Thor][thor] — NVIDIA's ONNX → ModelOpt → `trtexec` pipeline.
5. [ModelOpt ONNX PTQ README][modelopt] and [support matrix][modelopt-matrix].
6. [TensorRT Edge-LLM][edgellm] ([Jetson tutorial][edgellm-tut]) — NVIDIA's C++ LLM/VLM runtime; reference design.
7. [TensorRT 10 C++ API patterns (enqueueV3)][trt-cpp] and [cyrusbehr/tensorrt-cpp-api][cyrusbehr].
8. [Arm: export SmolVLA to ONNX][arm]
9. [VLA benchmarks across Jetson boards (NVIDIA forum)][forum] — benchmark methodology.
10. [LeRobot issue #3636][issue] — SmolVLA misbehaving on Orin.

Also: [furuya02/jetson-smolvla-edge-bench][furuya], [AdaVLA (adaptive denoising steps)][adavla], [LITHE (Python/C++ shared memory)][lithe].

[eetmie]: https://github.com/eetmie/jetson-orin-nano-vla
[eetmie-opt]: https://github.com/eetmie/jetson-orin-nano-vla/blob/main/docs/optimization.md
[eetmie-hf]: https://huggingface.co/eetmie/smolvla-base-onnx
[jared]: https://jared-hpc.com/posts/vla-physical-ai-tensorrt-tether/
[flashrt]: https://github.com/flashrt-project/FlashRT/blob/main/docs/deployment_orin.md
[thor]: https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/
[modelopt]: https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/onnx_ptq/README.md
[modelopt-matrix]: https://nvidia.github.io/Model-Optimizer/guides/0_support_matrix.html
[edgellm]: https://nvidia.github.io/TensorRT-Edge-LLM/overview.html
[edgellm-tut]: https://www.jetson-ai-lab.com/tutorials/tensorrt-edge-llm/
[trt-cpp]: https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-8x-to-10x-c-api-patterns.html
[cyrusbehr]: https://www.github.com/cyrusbehr/tensorrt-cpp-api
[arm]: https://learn.arm.com/learning-paths/cross-platform/smolvla-onnx-conversion/convert-and-validate-onnx/
[forum]: https://forums.developer.nvidia.com/t/vla-models-0-5-gr00t-smolvla-measured-on-jetson-agx-thor-agx-orin-and-orin-nano-latency-energy-memory-per-policy-call/387191
[issue]: https://github.com/huggingface/lerobot/issues/3636
[furuya]: https://github.com/furuya02/jetson-smolvla-edge-bench
[adavla]: https://arxiv.org/pdf/2608.29208
[lithe]: https://arxiv.org/pdf/2603.07442
