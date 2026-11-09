# Learning 01: Understanding the eetmie optimization notes

*A plain-language walkthrough of [eetmie/jetson-orin-nano-vla — docs/optimization.md](https://github.com/eetmie/jetson-orin-nano-vla/blob/main/docs/optimization.md),
the closest prior work to this project. Written 2026-11-09.*

Their SmolVLA went from **126.8 ms → 104.6 ms** (~17 %) on the Orin Nano — not from one trick, but
from many 1–8 ms wins.

---

## Part 1 — Background

```
PyTorch model ──export──▶ ONNX graph ──TensorRT build──▶ engine ──run──▶ actions
 (Python code)            (a file listing          (compiled GPU program
                           every math operation)    for THIS device only)
```

- **ONNX graph** — the model written out as a flowchart of operations (MatMul, Add, Softmax…).
  Harmless-looking PyTorch code can turn into slow operations here.
- **TensorRT build** — merges operations ("fusion") and benchmarks several GPU implementations
  ("**tactics**") per operation, keeping the fastest. The result is an **engine**.
- **Kernel** — one small program running on the GPU (e.g. one matrix multiply).
- **GEMM** — matrix multiply; most of a transformer's time goes here and to attention.
- **Triton** — a Python-like language for writing custom GPU kernels, plugged into TensorRT.
- **CUDA graph** — record a whole sequence of kernel launches once, replay it with one call;
  removes CPU launch overhead.

**The key fact about the Orin Nano: memory speed, not compute, is usually the limit.** ~100 GB/s
(eetmie measured 96 GB/s for a plain copy). At batch size 1 the GPU mostly *waits for weights to
arrive from memory*.

**Precision:**
- **FP32** — accurate, slow. **FP16** — half the memory, much faster; max value 65,504, and coarse
  at large values (near 6,000, consecutive FP16 numbers are 4 apart).
- **Accumulation** — a matrix multiply sums thousands of products. Inputs can be FP16 while the
  *running total* stays FP32. Summing in FP16 loses accuracy, like adding cents to a total rounded
  to whole dollars.

---

## Part 2 — The 17 findings, grouped

### A. Clean up the exported graph
| # | Finding | Plain terms |
|---|---|---|
| 1 | `ScatterND` in the ONNX | LeRobot's rotary-position code writes into tensor slices (`x[..., :n] = ...`), which exports as a slow scatter. Rebuilding with `torch.cat` — same math — saves **7.5 ms** and 100 MB. Biggest single win. |
| 9 | Conv patch embedding on old kernels | Rewrite "cut image into patches" as reshape + MatMul: −0.8 ms. |
| 2, 13 | Masks built from constants | Masks that never change at run time: drop the hidden rows at export instead of computing and masking. −6–9 ms on GR00T. |
| 4 | Computing rows that get thrown away | Compute only the rows the next stage uses. |

→ The exporter is where much of the speed is won.

### B. Don't repeat work
| # | Finding | Plain terms |
|---|---|---|
| 3 | Observation-only work inside the denoising loop | ~10 denoising steps per chunk. Anything depending only on the image (cross-attention K/V) is computed **once**: −2.1 ms. |
| 17 | Re-laying-out a cache every step | EVO1 reshuffled its cache 256× per inference; once is enough: −19.8 ms. |

→ For every operation in the loop, ask: does this change between steps?

### C. Move less data
| # | Finding | Plain terms |
|---|---|---|
| 5 | FP32 at engine boundaries | Engine A converts FP16 → FP32 for the hand-off, engine B converts straight back: wasted traffic and RAM. Keep FP16 across the boundary. |
| 14 | FP32 → FP16 → FP32 round trips at outputs | An accurate FP32 result was squeezed through FP16 for nothing. Keeping FP32 cut GR00T error ~45 %, free. |
| 16 | One buffer per output | Alternate two reused buffers: tens of MB saved — matters on 8 GB. |

### D. Image preprocessing
| # | Finding | Plain terms |
|---|---|---|
| 6 | numpy preprocessing, cameras one after another | GPU preprocessing: SmolVLA **3.9 → 0.5 ms**, π0.5 28.5 → 9.7 ms. Our C++ runtime gets this naturally (upload `uint8` once, everything on GPU). |

### E. Custom kernels (advanced, last 5–10 %)
| # | Finding | Plain terms |
|---|---|---|
| 7, 15 | TensorRT attention slow on short/odd sequences | Triton attention reading data in its existing layout; for causal masks, skip fully-masked blocks. |
| 8 | Tile size | Kernels work in tiles (e.g. 64×64); re-test after any change — 128×64 saved 2.6 ms. |
| 11 | Huge GEMV accumulating in FP16 | FP32 accumulation: same speed, **half the error**. |
| 12 | Conv in the "wrong" memory layout | Native-layout kernel avoids transposes: −11 ms on X-VLA. |

### F. What didn't help
- **#10 builder optimization level 3–5**: < 1 %, mostly noise.
- **Custom GEMM for GR00T**: TensorRT already reads weights at 86–91 GB/s against a ~96 GB/s
  ceiling — nothing left (the memory-limit fact again).
- **Fused QKV projection**: −0.38 ms for +9.3 MB RAM → rejected. Rule of thumb: **~+100 MB is worth 5 ms**.
  Memory is a budget you spend.

---

## Part 3 — Precision lessons

1. **TensorRT silently picks FP16-accumulating GEMMs** (~25 % faster), with no builder flag to forbid
   it in 10.16. Fix: write MatMuls as `Cast→FP32 → MatMul → Cast→FP16`; TensorRT then uses
   FP16-input / FP32-accumulate kernels. SmolVLA: **+2.3 % time, −31 % error** — a textbook
   results-table row.
2. **More precision isn't automatically better**: FP32 attention scores helped GR00T, X-VLA, π0.5,
   but made SmolVLA's vision *worse*. Measure per model.
3. **Watch for large values**: π0.5's residual stream reaches ~6,000, where FP16 steps by 4. Keeping
   it FP32 halved the error, free. **Critical for INT8**: large-value layers are the ones that break
   under quantization.
4. **Compare against FP32, not another fast version**: two FP16 builds can differ by more than the
   tolerance. Gate against stock FP32, several inputs, only the action dims the robot uses —
   padded dims can hide real errors.

---

## Part 4 — Their method (we adopt it)

1. Profile with **Nsight Systems**; read the ONNX and TensorRT's per-layer tactics before writing code.
2. Change **one thing** at a time, into a new engine cache.
3. Gate on **stored FP32 reference actions** plus varied inputs.
4. **Screen at 60 s, confirm with paired 300 s runs** (thermals and clocks make short runs noisy).
5. Measure **process memory (RSS)**, not just board RAM.

---

## Part 5 — What this means for edge-ai-engine

| Their finding | Our plan |
|---|---|
| Split engines + chain replayed as one CUDA graph | Core design of our **C++ runtime** |
| Export cleanups (#1, #3, #5, #9, #14) | Built into **our exporter** from day one — reproduce 126.8 → ~105 ms ourselves |
| FP32 reference-action gate | "Golden test data" gate before anything moves the arm |
| 60 s / 300 s runs, Nsight, RSS | Benchmark harness protocol |
| Custom Triton kernels | Optional and late |
| **FP16 only** + memory bandwidth is the limit | **Our contribution**: INT8 weights = half the bytes, INT4 = a quarter — attacking exactly what's left. Precision lessons tell us which layers to keep in higher precision. |

**Portfolio story:** reproduce the best published FP16 result on the Orin Nano with our own C++
runtime, go beyond it with INT8/INT4 (speed *and* accuracy), and drive a real arm with it.
