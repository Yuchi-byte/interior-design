"""Gate (a): how far a finished render's geometry has drifted from the photo's."""

from __future__ import annotations

from typing import Final

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from roomgen_geometry.alignment import align_scale_shift, apply_scale_shift
from roomgen_geometry.depth.base import DepthEstimator
from roomgen_geometry.reference import GeometricReference
from roomgen_geometry.reward import masked_depth_error

# `masked_depth_error` computes in float32 by contract whatever dtype the maps
# arrive in, so the residual map behind it must too: fitted and differenced at a
# half-precision backend's own dtype, the map would round away from the score it
# is supposed to square to.
_RESIDUAL_DTYPE: Final = torch.float32


def depth_error(
    image: Float[Tensor, "batch 3 height width"],
    *,
    reference: GeometricReference,
    depth_estimator: DepthEstimator,
    mask: Bool[Tensor, "batch height width"] | None = None,
) -> float:
    """Masked, scale-and-shift-invariant depth error of `image` against `reference`.

    Args:
        image: NCHW, floating point, values in ``[0, 1]``.
        depth_estimator: the backend `reference` was built with — a reference is
            only meaningful to the backend that produced it.
        mask: the pixels to score, defaulting to the reference's whole shell.
            A render is judged on its *own* evaluation mask — that shell narrowed
            by a segmentation of this very image — so that furniture the render
            legitimately put in front of a wall is not charged as moved
            architecture; the caller builds it, because only the caller knows
            which polarity the guidance was steered under. Always read
            :func:`shell_coverage` beside the score: a mask that exempts the
            frame scores a perfect zero.

    The score is a plain number, taken under ``no_grad``: nothing downstream of
    a finished render is differentiated.
    """
    scored = reference.mask if mask is None else mask
    with torch.no_grad():
        disparity = depth_estimator(image)
        error = masked_depth_error(disparity, reference.disparity, scored)
    return float(error.mean())


def shell_coverage(
    mask: Bool[Tensor, "batch height width"], *, reference: GeometricReference
) -> float:
    """The share of the reference's shell that an arm's evaluation mask still scores.

    1 is the whole photographed shell; 0 is a render whose segmentation excused
    all of it. The degenerate way to win gate (a) is to cover the room in
    furniture and be scored on nothing, which costs no depth error at all — this
    is the number that shows it happened.
    """
    shell = int(reference.mask.sum())
    if shell == 0:
        return 0.0
    return int((mask & reference.mask).sum()) / shell


def signed_depth_residual_map(
    image: Float[Tensor, "batch 3 height width"],
    *,
    reference: GeometricReference,
    depth_estimator: DepthEstimator,
    mask: Bool[Tensor, "batch height width"] | None = None,
) -> Float[Tensor, "batch height width"]:
    """Which way `image` bent the room, pixel by pixel: aligned disparity minus reference.

    Args:
        image: NCHW, floating point, values in ``[0, 1]``.
        depth_estimator: the backend `reference` was built with.
        mask: the pixels to score, as :func:`depth_error` takes it and with the
            same default; the map is zero outside it, and the alignment is fitted
            over it, so a map and a score only agree when both were given it.

    Positive where the render's inverse depth sits above the photo's — the
    surface reads closer than it was — negative where it reads further away,
    and zero outside the mask. Returned in float32, in the aligned inverse-depth
    units of `depth_error`, whose score is the mean of this map's square over
    the mask. Taken under ``no_grad``, like every finished-render measurement.
    """
    scored = reference.mask if mask is None else mask
    with torch.no_grad():
        disparity = depth_estimator(image).to(_RESIDUAL_DTYPE)
        target = reference.disparity.to(_RESIDUAL_DTYPE)
        # Affine-invariant inverse depth shares no frame with the reference
        # until it is fitted onto it, exactly as `masked_depth_error` does.
        aligned = apply_scale_shift(
            disparity, align_scale_shift(disparity, target, scored)
        )
        # Replaced rather than weighted, since `0 * NaN` is NaN: a non-finite
        # pixel the mask excludes must not decide the map the score is read from.
        zero = torch.zeros((), dtype=_RESIDUAL_DTYPE, device=aligned.device)
        return torch.where(scored, aligned - target, zero)
