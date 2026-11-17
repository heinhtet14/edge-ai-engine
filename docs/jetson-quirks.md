# Jetson install quirks

A running log of every Jetson-specific gotcha hit during this project, with the fix and the
exact versions involved. Newest first.

**Target:** Jetson Orin Nano 8 GB · JetPack 6.2 · L4T 36.4.x · CUDA 12.6 · TensorRT 10.3 · Python 3.10

| Date | Problem | Fix | Versions |
|---|---|---|---|
| — | *(entries added from M0 onward)* | | |

## Known up front (to verify in M0)
- **PyTorch:** the standard PyPI wheels are not built for Jetson. Use NVIDIA/Jetson AI Lab wheels
  that match JetPack 6.2 exactly; `torchvision` must match the `torch` build.
- **LeRobot vs. Python 3.10 — confirmed conflict (2026-11-17):** LeRobot `>=0.5.0` requires
  Python `>=3.12` (0.4.4 is the last release supporting 3.10; checked on PyPI). JetPack 6.2 ships 3.10.
  Options, to decide before Jetson setup: (a) pin LeRobot 0.4.4 on the Jetson (only used there for motors/
  control; inference is C++), (b) a Python 3.12 env on the Jetson (needs Jetson torch wheels for cp312),
  (c) upgrade to JetPack 7.2.x if it ships Python 3.12.
- **Power mode:** benchmarks require a fixed mode — `sudo nvpmodel -m <MAXN SUPER id>` and
  `sudo jetson_clocks`. Record both in every benchmark run.
- **Storage:** use NVMe, not SD card; add swap for pi0 experiments.
