import os
from collections.abc import Iterator

import pytest
import torch

from roomgen_guide.determinism import CUBLAS_WORKSPACE_CONFIG, enable_determinism


@pytest.fixture
def restored_torch_flags() -> Iterator[None]:
    # The flags are process-global, so a test that sets them has to put them
    # back or every later test runs under different kernels than CI does.
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn = torch.backends.cudnn.deterministic
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.deterministic = cudnn


def test_enabling_determinism_sets_the_flags_and_the_cublas_workspace(
    restored_torch_flags: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False

    enable_determinism()

    assert torch.are_deterministic_algorithms_enabled()
    assert torch.is_deterministic_algorithms_warn_only_enabled()
    assert torch.backends.cudnn.deterministic
    # cuBLAS reads this at handle creation, which is why the call has to come
    # before the process's first CUDA allocation rather than beside the sampler.
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == CUBLAS_WORKSPACE_CONFIG


def test_warn_only_keeps_an_op_without_a_deterministic_kernel_running(
    restored_torch_flags: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bilinear interpolation's backward is on the reward's own path through the
    # depth preprocessing; erroring there would forbid guidance outright, so the
    # request must degrade to a warning and still produce a gradient.
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)
    enable_determinism()
    image = torch.rand(1, 3, 8, 8, requires_grad=True)

    resized = torch.nn.functional.interpolate(
        image, size=(16, 16), mode="bilinear", align_corners=False
    )
    resized.sum().backward()

    assert image.grad is not None
    assert torch.isfinite(image.grad).all()
