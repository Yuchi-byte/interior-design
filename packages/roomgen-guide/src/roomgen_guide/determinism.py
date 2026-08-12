"""Best-effort run-to-run determinism on top of the seeded generators."""

from __future__ import annotations

import os

import torch

# cuBLAS reuses one workspace across streams unless told to allocate per-stream
# ones, which makes a reduction's summation order depend on what else was in
# flight. This is the value PyTorch documents for a deterministic GEMM.
CUBLAS_WORKSPACE_CONFIG: str = ":4096:8"


def enable_determinism() -> None:
    """Ask torch for deterministic kernels wherever it has them.

    Seeding is not the whole story on CUDA: a seeded generator fixes the noise,
    but atomics and autotuned reductions still let two runs of the same seed
    disagree in the last bits, which a guided sampler integrates over its whole
    trajectory. This narrows that variance; it does not guarantee it away.

    ``warn_only`` is deliberate. Some ops on the reward's own path — bilinear
    interpolation's backward in the depth preprocessing among them — ship no
    deterministic kernel, and a hard error there would simply forbid guidance;
    torch warns instead, naming the op, which is the list worth acting on.

    Call before the first CUDA allocation of the process:
    ``CUBLAS_WORKSPACE_CONFIG`` is read when the cuBLAS handle is created, so
    setting it after the first matmul has no effect on that handle.
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
