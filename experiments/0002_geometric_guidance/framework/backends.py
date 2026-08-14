"""Building the model stack a variant asks for: the real checkpoints, or the weightless doubles."""

from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
from diffusers import DDIMScheduler

from framework.config import TORCH_DTYPES, AlignmentBackendConfig, BackendConfig
from roomgen_geometry.depth.base import DepthEstimator
from roomgen_geometry.depth.depth_anything import DepthAnythingV2
from roomgen_geometry.depth.depth_anything_v3 import DepthAnythingV3Metric
from roomgen_geometry.segmentation.base import Segmenter
from roomgen_geometry.segmentation.eomt import EoMT
from roomgen_guide.determinism import enable_determinism
from roomgen_guide.renovate import load_pipeline
from roomgen_guide.sampling.pipeline import Img2ImgPipeline

# The two test-double packages reach their suites through the root pytest
# `pythonpath`; a script gets there the same way, and importing them beats
# maintaining a second, subtly different set of fakes for the smoke variant.
_FAKE_PACKAGES: Final = (
    Path(__file__).resolve().parents[3] / "packages" / "roomgen-geometry" / "tests",
    Path(__file__).resolve().parents[3] / "packages" / "roomgen-guide" / "tests",
)


@dataclass(frozen=True)
class Backends:
    """Everything a run needs to sample and to score, already on the target device."""

    pipeline: Img2ImgPipeline
    depth_estimator: DepthEstimator
    segmenter: Segmenter
    device: torch.device
    dtype: torch.dtype


@dataclass(frozen=True)
class AlignmentBackends:
    """The benchmark's stack plus the metric judge, and the two named for what they do.

    `metric_depth` (DA3 metric) both steers the guided arms and scores the
    study's primary matrix; `judge_depth` (V2, affine-invariant) never steers
    anything and scores every arm under the full fit — the second judge that
    keeps the comparison from being read through one backend's eyes.
    """

    pipeline: Img2ImgPipeline
    metric_depth: DepthEstimator
    judge_depth: DepthEstimator
    segmenter: Segmenter
    device: torch.device
    dtype: torch.dtype


def build_alignment_backends(
    config: AlignmentBackendConfig, *, cache_dir: Path
) -> AlignmentBackends:
    """The alignment study's stack: everything the benchmark loads, plus DA3 metric."""
    base = build_backends(config, cache_dir=cache_dir)
    metric_depth: DepthEstimator
    match config.kind:
        case "real":
            metric_depth = DepthAnythingV3Metric(
                config.metric_depth_checkpoint,
                device=base.device,
                dtype=TORCH_DTYPES[config.depth_dtype],
                cache_dir=cache_dir,
                input_size=config.metric_input_size,
            )
        case "fake":
            # `build_backends` has already put the doubles on the path. A
            # deliberately different affine frame from the judge's double, so a
            # smoke run exercises exactly the disagreement the modes are about.
            from roomgen_fakes import FakeDepthEstimator  # noqa: PLC0415

            metric_depth = FakeDepthEstimator(scale=2.0, shift=0.25)
    return AlignmentBackends(
        pipeline=base.pipeline,
        metric_depth=metric_depth,
        judge_depth=base.depth_estimator,
        segmenter=base.segmenter,
        device=base.device,
        dtype=base.dtype,
    )


def build_backends(config: BackendConfig, *, cache_dir: Path) -> Backends:
    """The stack `config` selects, loaded from `cache_dir` when it needs weights."""
    if config.deterministic:
        # Before the weights land on the device: `CUBLAS_WORKSPACE_CONFIG` is
        # read when the cuBLAS handle is created, and building the stack is the
        # first thing in the process that touches CUDA.
        enable_determinism()
    device = torch.device(config.device)
    dtype = TORCH_DTYPES[config.dtype]
    match config.kind:
        case "real":
            return _real_backends(
                config, device=device, dtype=dtype, cache_dir=cache_dir
            )
        case "fake":
            return _fake_backends(config, device=device, dtype=dtype)


def _real_backends(
    config: BackendConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
    cache_dir: Path,
) -> Backends:
    pipeline = load_pipeline(
        config.diffusion_checkpoint, device=device, dtype=dtype, cache_dir=cache_dir
    )
    if config.scheduler == "ddim":
        # SD1.5 ships PNDM, whose update carries history across steps; DDIM's is
        # the deterministic closed form over the same epsilon parameterisation
        # that the guided loop's x̂₀ correction is derived from.
        #
        # diffusers ships `py.typed` but leaves `from_config`'s `**kwargs`
        # unannotated, which makes the constructor an untyped call — the same
        # gap `load_pipeline` hits on `from_pretrained`.
        pipeline.scheduler = DDIMScheduler.from_config(  # type: ignore[no-untyped-call]
            pipeline.scheduler.config
        )
    elif "generator" not in inspect.signature(pipeline.scheduler.step).parameters:
        raise ValueError(
            f"backends.scheduler='pipeline' keeps {type(pipeline.scheduler).__name__},"
            " whose step() cannot take the loop's generator; use 'ddim'"
        )
    return Backends(
        pipeline=pipeline,
        depth_estimator=DepthAnythingV2(
            config.depth_checkpoint,
            device=device,
            dtype=TORCH_DTYPES[config.depth_dtype],
            cache_dir=cache_dir,
        ),
        segmenter=EoMT(
            config.segmentation_checkpoint,
            device=device,
            dtype=TORCH_DTYPES[config.segmentation_dtype],
            cache_dir=cache_dir,
        ),
        device=device,
        dtype=dtype,
    )


def _fake_backends(
    config: BackendConfig, *, device: torch.device, dtype: torch.dtype
) -> Backends:
    if config.scheduler != "pipeline":
        raise ValueError(
            "the fake stack ships its own scheduler, so backends.scheduler must be "
            f"'pipeline', not {config.scheduler!r}"
        )
    for package in _FAKE_PACKAGES:
        if str(package) not in sys.path:
            sys.path.insert(0, str(package))
    # Local by necessity, not by taste: the doubles are only importable once the
    # lines above have put their packages on the path, and a `real` run must not
    # need them on the path at all.
    from roomgen_fakes import FakeDepthEstimator, FakeSegmenter  # noqa: PLC0415
    from roomgen_guide_fakes import FakePipeline  # noqa: PLC0415

    return Backends(
        pipeline=FakePipeline(),
        depth_estimator=FakeDepthEstimator(),
        segmenter=FakeSegmenter(),
        device=device,
        dtype=dtype,
    )
