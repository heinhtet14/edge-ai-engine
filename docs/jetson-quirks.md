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
- **LeRobot vs. Python 3.10:** check LeRobot's required Python version against JetPack's system Python.
  Fallbacks: a `jetson-containers` image, or a separate Python build.
- **Power mode:** benchmarks require a fixed mode — `sudo nvpmodel -m <MAXN SUPER id>` and
  `sudo jetson_clocks`. Record both in every benchmark run.
- **Storage:** use NVMe, not SD card; add swap for pi0 experiments.
