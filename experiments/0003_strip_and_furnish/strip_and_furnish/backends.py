"""Building the model stack a variant asks for: the real checkpoints, or the weightless doubles."""

from __future__ import annotations

import inspect
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
from diffusers import DDIMScheduler

from roomgen_geometry.depth.base import DepthEstimator
from roomgen_geometry.depth.depth_anything import DepthAnythingV2
from roomgen_geometry.segmentation.base import Segmenter
from roomgen_geometry.segmentation.eomt import EoMT
from roomgen_guide.determinism import enable_determinism
from roomgen_guide.eval import (
    ClipPromptAdherence,
    ContentsChangeAdherence,
    PromptAdherence,
)
from roomgen_guide.guidance.clip import (
    ClipTextEnergies,
    ClipTextGuidance,
    load_clip_text_energies,
)
from roomgen_guide.renovate import load_pipeline
from roomgen_guide.sampling.inpaint import InpaintPipeline, load_inpaint_pipeline
from roomgen_guide.sampling.pipeline import Img2ImgPipeline
from strip_and_furnish.config import TORCH_DTYPES, AdherenceConfig, BackendConfig

# The two test-double packages reach their suites through the root pytest
# `pythonpath`; a script gets there the same way, and importing them beats
# maintaining a second, subtly different set of fakes for the smoke variant.
_FAKE_PACKAGES: Final = (
    Path(__file__).resolve().parents[3] / "packages" / "roomgen-geometry" / "tests",
    Path(__file__).resolve().parents[3] / "packages" / "roomgen-guide" / "tests",
)

# The text energy runs in float32 whatever the sampler's dtype: it is a second
# model on the gradient path, and its cosine is what the composite's weight is
# sized against, so quantising it would move the weight rather than the render.
_TEXT_ENERGY_DTYPE: Final = torch.float32

TextEnergy = Callable[[str], ClipTextGuidance]
"""One prompt in, the energy that pulls a render towards it out."""

BuildInpainter = Callable[[], InpaintPipeline]
"""Stage 1's pipeline, built when the strip runs and dropped when it is over."""


@dataclass(frozen=True)
class Backends:
    """Everything the two stages need, already on the target device.

    The inpainter is a factory rather than a loaded pipeline: it is the one
    model nothing after stage 1 uses, and a guided step's VAE backward wants
    every gigabyte the two stages are not both holding.
    """

    pipeline: Img2ImgPipeline
    build_inpainter: BuildInpainter
    depth_estimator: DepthEstimator
    segmenter: Segmenter
    text_energy: TextEnergy
    device: torch.device
    dtype: torch.dtype


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


def build_adherence(
    config: AdherenceConfig, *, device: torch.device, cache_dir: Path | None
) -> PromptAdherence:
    """The prompt-adherence metric `config` selects, on a checkpoint no arm optimised."""
    match config.kind:
        case "clip":
            return ClipPromptAdherence(
                config.checkpoint, device=device, cache_dir=cache_dir
            )
        case "contents_change":
            return ContentsChangeAdherence()


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
        # unannotated, which makes the constructor an untyped call.
        pipeline.scheduler = DDIMScheduler.from_config(  # type: ignore[no-untyped-call]
            pipeline.scheduler.config
        )
    elif "generator" not in inspect.signature(pipeline.scheduler.step).parameters:
        raise ValueError(
            f"backends.scheduler='pipeline' keeps {type(pipeline.scheduler).__name__},"
            " whose step() cannot take the loop's generator; use 'ddim'"
        )

    def build_inpainter() -> InpaintPipeline:
        return load_inpaint_pipeline(
            config.inpainting_checkpoint,
            device=device,
            dtype=dtype,
            cache_dir=cache_dir,
        )

    # One CLIP for every scene: the image tower is what each of them
    # differentiates through, and only the frozen text embedding differs.
    energies: ClipTextEnergies = load_clip_text_energies(
        config.text_energy_checkpoint,
        device=device,
        dtype=_TEXT_ENERGY_DTYPE,
        cache_dir=cache_dir,
    )

    return Backends(
        pipeline=pipeline,
        build_inpainter=build_inpainter,
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
        text_energy=energies,
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
    from roomgen_guide_fakes import (  # noqa: PLC0415
        FakeInpaintPipeline,
        FakePipeline,
        fake_clip_text_guidance,
    )

    def text_energy(prompt: str) -> ClipTextGuidance:
        # Named rather than returned inline: the doubles are outside mypy's
        # `files`, so their return type arrives as `Any` and the annotation here
        # is what pins it back to the Protocol the real stack satisfies.
        energy: ClipTextGuidance = fake_clip_text_guidance(
            prompt, device=device, dtype=dtype
        )
        return energy

    def build_inpainter() -> InpaintPipeline:
        inpainter: InpaintPipeline = FakeInpaintPipeline()
        return inpainter

    return Backends(
        pipeline=FakePipeline(),
        build_inpainter=build_inpainter,
        depth_estimator=FakeDepthEstimator(),
        segmenter=FakeSegmenter(),
        text_energy=text_energy,
        device=device,
        dtype=dtype,
    )
