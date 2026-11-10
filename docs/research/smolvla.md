# Research: SmolVLA — architecture, training, and fine-tuning

*Compiled 2026-11-10. Numbers are as reported by each source; not yet reproduced by this project.
LeRobot docs were at v0.7.0 at the time of writing — CLI names change between versions, so pin the version in M0.*

## TL;DR

- **SmolVLA** = ~450M-parameter VLA from Hugging Face: a **frozen SmolVLM2** vision-language backbone
  (first 16 LLM layers only) + a **~100M flow-matching action expert** that outputs **50-action chunks**
  in **10 denoising steps**. Images are 512×512 → **64 visual tokens per frame**.
- **Fine-tuning recipe (official):** start from `lerobot/smolvla_base`, batch 64, **20k steps**
  (~4 h on one A100), map your camera names to `camera1/2/3` with `--rename_map`.
- **Data matters more than hyperparameters:** ~50 episodes minimum, *several demos per task variation*,
  one consistent strategy, dense coverage of a small workspace. 25 episodes was not enough in the official tests.
- **For us (wrist-camera only, public data):** the official SO-101 dataset has **no wrist camera**.
  Candidates with a wrist view exist (see [Public datasets](#public-datasets-for-our-setup)), but expect
  a sizeable gap between offline accuracy and real-arm success.

---

## Required reading (in order)

| # | Read | Why |
|---|---|---|
| 1 | [SmolVLA blog post (Hugging Face)][blog] | The 15-minute overview: architecture, community data, async inference, results. Start here. |
| 2 | [SmolVLA paper — arXiv 2506.01844][paper] ([HTML][paper-html]) | Exact architecture, training hyperparameters, and **ablations** (§ below) that tell us which knobs matter for optimization. |
| 3 | [LeRobot SmolVLA docs][docs] | The official fine-tuning and deployment commands. Our M4 starts from this. |
| 4 | [`lerobot/smolvla_base` model card][card] | Checkpoint details, optional policy flags (`chunk_size`, `n_action_steps`, gradient checkpointing). |
| 5 | [Fine-tuning SmolVLA for cube pick-and-place on SO-101 (ggando)][ggando] | **Best practitioner write-up**: 4 data rounds, what failed and why, real success rates. |
| 6 | [Arm Learning Path: fine-tune SmolVLA for SO-101][arm-ft] | Full reproducible command incl. `--policy.empty_cameras`, loss curve, checkpoint layout. |
| 7 | [LeRobot async inference docs][async] | Policy-server / robot-client split; the design our C++ runtime replaces. |
| 8 | [LeRobot paper — arXiv 2602.22818][lerobot-paper] | Framework context; sync vs. async numbers on SO-100. |
| 9 | [Training notebook (Colab)][colab] | Quick way to see the training loop end-to-end. |

Optional: [Seeed Studio SO-Arm LeRobot tutorial][seeed], [Sa74ll/smolvla-so100-finetune][sa74ll],
[huangbj16/smolvla_so101][huangbj], [viam-labs/smolvla-so101-training][viam],
[openelab fine-tuning guide][openelab] (sections on episode counts, LoRA, common errors).

---

## Architecture

| Component | Detail | Source |
|---|---|---|
| Backbone | SmolVLM2-500M (SigLIP vision encoder + SmolLM2 decoder), **frozen** during training | [paper][paper-html], [card][card] |
| Layer skipping | Only the **first 16 LLM layers** are used (N = L/2); roughly halves compute | [paper][paper-html] |
| Vision | Global image only (no tiling), **512×512**, pixel shuffle → **64 tokens / frame** | [paper][paper-html] |
| State | Linear projection of the 6-D joint state → **1 token** in the VLM prefix | [paper][paper-html] |
| Action expert | ~100M params, hidden size **0.75×** the VLM's; **interleaved cross-attention + causal self-attention** blocks | [paper][paper-html] |
| Objective | **Flow matching** (predict ε − Aₜ), τ ~ Beta distribution (as in π0) | [paper][paper-html] |
| Inference | **10 flow-matching steps**, chunk size **n = 50** | [paper][paper-html] |
| Size | ~450M total; weights shipped as F32 and BF16 safetensors; Apache-2.0 | [card][card] |

**Why this matters for TensorRT:** the frozen VLM prefix (vision + 16 LM layers) runs **once per chunk**;
the action expert runs **10×**, cross-attending to the prefix's K/V. That's exactly the
three-engine split in [ADR-0001](../decisions/0001-tensorrt-as-primary-optimization-path.md), and why
"hoist observation-only work out of the loop" (see [learning 01](../learning/01-eetmie-optimization-notes.md)) pays off.

## Pretraining

| Item | Value | Source |
|---|---|---|
| Data | **481 community datasets**, 22.9K episodes, 10.6M frames (SO-100 arm, 30 FPS) | [paper][paper-html] |
| Label cleanup | Task strings rewritten by Qwen2.5-VL-3B into short verb-led instructions | [blog][blog] |
| Camera mapping | `OBS_IMAGE_1/2/3` in priority order **top → wrist → side** | [paper][paper-html] |
| Steps / batch | 200k steps, global batch 256 | [paper][paper-html] |
| Optimizer | AdamW (β1 0.9, β2 0.95); 100-step warmup; cosine decay 1e-4 → 2.5e-6 | [paper][paper-html] |
| Precision | bfloat16 + `torch.compile`, 4 GPUs via accelerate; ~30k GPU-hours total | [paper][paper-html] |

## Results worth knowing

| Benchmark | SmolVLA 0.45B | Notes | Source |
|---|---|---|---|
| LIBERO avg | 87.3 % | competitive with much larger VLAs | [paper][paper-html] |
| Meta-World avg | 57.3 % | | [paper][paper-html] |
| SO-100 real (pick-place / stack / sort) | 75 / 90 / 70 % → **78.3 % avg** | **51.7 % without pretraining** | [paper][paper-html] |
| SO-101 pick-place-lego | **90 %** in-distribution / 50 % out-of-distribution | ACT: 70 / 40 % | [paper][paper-html] |
| vs. π0 | ~40 % faster to train, ~6× less memory | | [paper][paper-html] |

### Ablations (LIBERO) — knobs relevant to our optimization ladder
| Knob | Finding | Implication for us |
|---|---|---|
| Attention | Interleaved CA+SA 85.5 % vs CA-only 79.0 % vs SA-only 74.5 % | Keep architecture as-is |
| LM layers | 16: 78.5 %, 24: 79.5 %, 32: 80.3 % | 16 layers is already the speed/accuracy sweet spot |
| Objective | Flow matching 80.25 % vs L1 regression 75.25 % | Denoising loop stays; we optimize it, not remove it |
| **Chunk size** | 10: **84.0 %**, 50: 80.3 %; 1 and 100 worse | Our chunk sweep should include 10 and 50 |
| **Executed steps per chunk** | 1: 80.3 %, **10: 82.8 %**, 30: 70.8 %, 50: 51.8 % | Re-planning often helps → latency matters, not just Hz |
| Expert width | 0.75×: 77.5 %, 1.0×: 82.3 %, 0.5×: 80.3 % | (Paper's own numbers don't favor 0.75× — noted) |

### Limitations (from the paper)
Single robot type in pretraining (SO-100); ~23k trajectories (vs ~1M for OpenVLA); backbone pretrained
mainly on documents/OCR; weak on long-horizon tasks; imitation learning only.

---

## Fine-tuning

### Official recipe
```bash
# [TRAIN]  — LeRobot docs (v0.7.0)
pip install -e ".[smolvla]"

lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=lerobot/svla_so100_pickplace \
  --policy.repo_id=${HF_USER}/my_smolvla \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --batch_size=64 \
  --steps=20000 \
  --output_dir=outputs/train/my_smolvla \
  --job_name=my_smolvla_training \
  --policy.device=cuda
```
- `--rename_map` is **required**: `smolvla_base` expects `camera1/2/3`; without it training fails with a feature mismatch.
- `--policy.empty_cameras=N` lets expected camera slots be absent ([Arm guide][arm-ft]) — relevant for a wrist-only setup.
- Optional: `--policy.dtype=bfloat16`, `--policy.chunk_size`, `--policy.n_action_steps`,
  `--policy.gradient_checkpointing=true` ([card][card]).

### Hyperparameters across sources
| Source | Steps | Batch | Hardware / time | Notes |
|---|---|---|---|---|
| [LeRobot docs][docs] | 20k | 64 | 1× A100, ~4 h | "Tune steps to your use case" |
| [Model card][card] | 100k | 4 | — | bf16; different example |
| [Paper][paper-html] | 100k (sim) / 200k (real) | 64 | — | Fine-tuning in the paper's experiments |
| [ggando][ggando] | 20k | 64 | RTX 3090, ~10 h | lr 1e-4 cosine; loss 0.005, still decreasing at 20k |
| [Arm guide][arm-ft] | 20k | 64 | DGX Spark, ~30 h | final loss 0.017 |

**What's trained:** action expert + projections (~50M params); vision encoder and LM stay frozen ([ggando][ggando], [Arm][arm-ft]).

**Small GPUs:** community reports say defaults assume 16–24 GB. Under memory pressure: smaller batch +
gradient accumulation, bf16, gradient checkpointing, shorter chunk, or LoRA via PEFT
([search summary sources][ggando], [openelab][openelab]). *Our RTX 2000 Ada has 16 GB — batch size to be found empirically in M4.*

### Data lessons (the part that actually decides success)
- **~50 episodes minimum**; official tests: 50 episodes over 5 cube positions × 10 each worked, **25 was not enough** ([docs][docs]).
- **Repeat each variation several times** (e.g. 10 demos per cube position) ([docs][docs]).
- **Dense beats wide:** 50 demos over a 30 cm area → grasping at air; 75 demos over 10 cm → ~80 % ([ggando][ggando]).
- **One consistent strategy:** mixing grasp styles dropped success to 20–80 %; "the policy learns exactly what you show it, including your bad habits" ([ggando][ggando]).
- **Overhead camera helps**: dual-camera 5/5 vs wrist-only 4/5 in a small comparison ([ggando][ggando]).
- **Consistency at deployment:** lighting changes and recalibrating without retraining hurt performance ([ggando][ggando]).
- **Loss ≠ success:** low training loss doesn't guarantee task success ([Arm][arm-ft]).

---

## Inference & deployment

```bash
# [JETSON] — reference Python deployment (LeRobot docs v0.7.0)
lerobot-rollout \
  --strategy.type=base \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower \
  --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --rename_map='{"observation.images.wrist": "observation.images.camera1"}' \
  --task="<same task string as the dataset>" \
  --policy.path=${HF_USER}/my_smolvla
```
- Same camera names, resolution, `--rename_map` and **task string** as in training.
- **RTC (real-time chunking)** for slow hardware: `--inference.type=rtc --inference.rtc.execution_horizon=10 --inference.rtc.max_guidance_weight=10.0` — smooths transitions between chunks. Relevant on the Orin Nano.
- `--strategy.type=episodic` records evaluation episodes with scene resets.

### Async inference
- Client sends a new observation when the action queue drops below a threshold *g* (blog: ~70 %);
  near-duplicate observations are skipped; overlapping chunks are blended ([paper][paper-html], [blog][blog]).
- Results: **similar success (78.3 % sync vs 73.3 % async) but ~30 % faster task completion**
  (9.7 s vs 13.75 s; 19 vs 9 cycles in a fixed window) ([paper][paper-html]).
- LeRobot's implementation uses a separate policy-server process over gRPC ([async docs][async]);
  module paths changed across versions (`lerobot.scripts.server.*` → `lerobot.async_inference.*`).
  Our C++ runtime will implement the same idea in-process.

---

## Public datasets for our setup

Checked via each dataset's `meta/info.json` on 2026-11-10:

| Dataset | Robot | Episodes / frames | Cameras | Fit for wrist-only |
|---|---|---|---|---|
| [`lerobot/svla_so101_pickplace`][ds-so101] | so100_follower* | 50 / 11,939 | `up`, `side` | ❌ **no wrist camera** |
| [`lerobot/svla_so100_pickplace`][ds-so100] | so100 | 50 / 19,631 | `top`, `wrist` | ✅ wrist view; the official fine-tuning example; SO-100 not SO-101 |
| [`CoRL2026-CSI/SO101-pnp_ma_100epi`][ds-corl] | so101_follower | 100 / 98,187 | `top`, `left_wrist` | ✅ SO-101, 100 episodes, wrist view |
| [`Dillonjohnson/pick_place_v2`][ds-dillon] | SO-101 | 45 / ~31k | front, wrist | ⚠️ gated; task label says "cube", description says "puck" |
| [`upna/pickup_blue_block_into_the_yellow_tray`][ds-upna] | SO-101 | small | front, wrist | ⚠️ metadata not readable at check time |
| [`geonmin-kim/merged_all7`][ds-merged] | mixed | 673 | incl. wrist | ⚠️ merged sources; inspect before use |

\* listed `robot_type` in metadata. All are LeRobot v3.0 format, 30 FPS, 640×480, 6-D state/action.

**Wrist-only options** (decided in M3 with evidence):
1. Train with only the wrist view mapped to `camera1` and `--policy.empty_cameras` for the rest — matches our hardware exactly.
2. Train with both views; at deployment feed a blank top view — mismatched, likely worse.
3. Add a cheap fixed top camera — matches the datasets and the "overhead camera helps" finding.

---

## Implications for edge-ai-engine

| Finding | What we do |
|---|---|
| Frozen 16-layer prefix runs once per chunk; expert runs 10× | Three TensorRT engines; hoist prefix K/V out of the loop |
| 512×512 → 64 tokens/frame | Fixed input shapes → static TensorRT profiles; GPU preprocessing in C++ |
| Chunk 10 and 10 executed steps did best on LIBERO | Chunk-size × executed-steps sweep in M8, measured on accuracy *and* latency |
| Async ≈ same success, ~30 % faster tasks; RTC exists for slow hardware | Async action queue in the C++ runtime; RTC as a reference to compare against |
| 20k steps × batch 64 ≈ 4 h A100 / 10 h 3090 | Expect a long run on our 16 GB RTX 2000 Ada; checkpoint every 5k steps |
| Official SO-101 dataset has no wrist camera | Shortlist `svla_so100_pickplace` and `CoRL2026-CSI/SO101-pnp_ma_100epi` for M3 |
| Data mismatch dominates real success | Report offline accuracy for every row; real-arm success only as a labeled, secondary metric |

### Open questions
- Does SO-100 data transfer to our SO-101 (same STS3215 servos; calibration/normalization conventions differ)?
- Largest batch size that fits on 16 GB with a frozen VLM in bf16?
- Does the open Orin numerics issue ([lerobot#3636][issue3636]) affect our PyTorch baseline on JetPack 6.2?

---

## Sources

[blog]: https://huggingface.co/blog/smolvla
[paper]: https://arxiv.org/abs/2506.01844
[paper-html]: https://arxiv.org/html/2506.01844v1
[docs]: https://huggingface.co/docs/lerobot/smolvla
[card]: https://huggingface.co/lerobot/smolvla_base
[ggando]: https://ggando.com/blog/smolvla-so101/
[arm-ft]: https://learn.arm.com/learning-paths/laptops-and-desktops/finetune-smolvla-lerobot/6-finetune-smolvla/
[async]: https://huggingface.co/docs/lerobot/main/async
[lerobot-paper]: https://arxiv.org/pdf/2602.22818
[colab]: https://colab.research.google.com/github/huggingface/notebooks/blob/main/lerobot/training-smolvla.ipynb
[seeed]: https://wiki.seeedstudio.com/lerobot_so100m_new/
[sa74ll]: https://github.com/Sa74ll/smolvla-so100-finetune
[huangbj]: https://github.com/huangbj16/smolvla_so101
[viam]: https://github.com/viam-labs/smolvla-so101-training
[openelab]: https://openelab.io/blogs/learn/how-to-fine-tune-smolvla-on-so-101-with-lerobot
[issue3636]: https://github.com/huggingface/lerobot/issues/3636
[ds-so101]: https://huggingface.co/datasets/lerobot/svla_so101_pickplace
[ds-so100]: https://huggingface.co/datasets/lerobot/svla_so100_pickplace
[ds-corl]: https://huggingface.co/datasets/CoRL2026-CSI/SO101-pnp_ma_100epi
[ds-dillon]: https://huggingface.co/datasets/Dillonjohnson/pick_place_v2
[ds-upna]: https://huggingface.co/datasets/upna/pickup_blue_block_into_the_yellow_tray
[ds-merged]: https://huggingface.co/datasets/geonmin-kim/merged_all7

- SmolVLA blog — <https://huggingface.co/blog/smolvla>
- SmolVLA paper — <https://arxiv.org/abs/2506.01844>
- LeRobot SmolVLA docs — <https://huggingface.co/docs/lerobot/smolvla>
- `smolvla_base` model card — <https://huggingface.co/lerobot/smolvla_base>
- ggando, Fine-tuning SmolVLA on SO-101 — <https://ggando.com/blog/smolvla-so101/>
- Arm Learning Path, fine-tune SmolVLA for SO-101 — <https://learn.arm.com/learning-paths/laptops-and-desktops/finetune-smolvla-lerobot/6-finetune-smolvla/>
- LeRobot async inference docs — <https://huggingface.co/docs/lerobot/main/async>
- LeRobot paper — <https://arxiv.org/pdf/2602.22818>
- Training notebook — <https://colab.research.google.com/github/huggingface/notebooks/blob/main/lerobot/training-smolvla.ipynb>
- Seeed Studio SO-Arm LeRobot tutorial — <https://wiki.seeedstudio.com/lerobot_so100m_new/>
- Community repos: [Sa74ll/smolvla-so100-finetune][sa74ll], [huangbj16/smolvla_so101][huangbj], [viam-labs/smolvla-so101-training][viam]
- openelab guide — <https://openelab.io/blogs/learn/how-to-fine-tune-smolvla-on-so-101-with-lerobot>
- LeRobot issue #3636 — <https://github.com/huggingface/lerobot/issues/3636>
