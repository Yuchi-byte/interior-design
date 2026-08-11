"""The end-to-end entry point: a photographed room in, a renovated one out."""

from __future__ import annotations

import torch
from jaxtyping import Float
from torch import Tensor

from roomgen_geometry.depth.base import DepthEstimator
from roomgen_geometry.segmentation.base import Segmenter
from roomgen_guide.guidance.schedule import GuidanceSchedule
from roomgen_guide.sampling.guided import Img2ImgPipeline


def renovate(
    photo: Float[Tensor, "batch 3 height width"],
    prompt: str,
    *,
    pipeline: Img2ImgPipeline,
    depth_estimator: DepthEstimator,
    segmenter: Segmenter,
    schedule: GuidanceSchedule,
    erosion_radius: int,
    steps: int,
    noise_strength: float,
    text_guidance_scale: float,
    generator: torch.Generator,
) -> Float[Tensor, "batch 3 height width"]:
    """Restyle the room in `photo` under `prompt` while holding its geometry.

    Args:
        photo: NCHW, floating point, values in ``[0, 1]``; the image returned
            follows the same convention.

    Freezes the photo's geometry into a
    :class:`~roomgen_geometry.reference.GeometricReference`, then samples
    against it — the same `depth_estimator` on both sides, since the reference
    is only meaningful to the backend that produced it.
    """
    raise NotImplementedError
