"""Closed-form least-squares removal of the affine freedom in inverse depth."""

from typing import Final, NamedTuple

import torch
from jaxtyping import Bool, Float
from torch import Tensor

# The normal equations of a scale/shift fit are ill-conditioned whenever the
# masked depth range is narrow, and float32 loses several digits there for no
# gain: the solve is a handful of reductions, so it always runs in float64 and
# the result is cast back to the caller's dtype.
_SOLVE_DTYPE: Final = torch.float64

# The 2x2 system's determinant is non-negative by Cauchy-Schwarz and vanishes
# exactly when the masked depths carry no variance (empty mask, one pixel, a
# flat region). Comparing it against this fraction of its own scale makes the
# test dimensionless, so it fires the same way for disparity in [0, 1] and for
# disparity in [0, 1000].
_DEGENERACY_TOLERANCE: Final = 1e-12


class ScaleShift(NamedTuple):
    scale: Float[Tensor, "batch"]
    shift: Float[Tensor, "batch"]


def align_scale_shift(
    prediction: Float[Tensor, "batch height width"],
    target: Float[Tensor, "batch height width"],
    mask: Bool[Tensor, "batch height width"],
) -> ScaleShift:
    """Per-batch-element least-squares fit of `prediction` onto `target`.

    Minimises ``sum_i mask_i * (scale * prediction_i + shift - target_i)^2``.
    Differentiable in both maps, and finite on a degenerate mask, where it
    returns the identity fit ``(1, 0)`` rather than a division by zero.
    """
    _check_shapes(prediction, target, mask)

    weight = mask.to(_SOLVE_DTYPE)
    predicted = prediction.to(_SOLVE_DTYPE)
    reference = target.to(_SOLVE_DTYPE)

    count = weight.flatten(1).sum(-1)
    sum_prediction = (weight * predicted).flatten(1).sum(-1)
    sum_target = (weight * reference).flatten(1).sum(-1)
    sum_prediction_squared = (weight * predicted * predicted).flatten(1).sum(-1)
    sum_cross = (weight * predicted * reference).flatten(1).sum(-1)

    determinant = count * sum_prediction_squared - sum_prediction * sum_prediction
    degenerate = determinant <= _DEGENERACY_TOLERANCE * count * sum_prediction_squared

    # Double guard. `torch.where` alone still evaluates the untaken branch and
    # backpropagates its NaN, so the denominator is sanitised *before* the
    # division rather than the quotient afterwards; and the selection is a
    # `where` rather than a Python `if` because the test is per batch element,
    # and branching on a tensor value would both break batching and drop the
    # gradient.
    safe_determinant = torch.where(
        degenerate, torch.ones_like(determinant), determinant
    )
    identity_scale = torch.ones_like(determinant)
    identity_shift = torch.zeros_like(determinant)

    # Cramer's rule on [[sum_dd, sum_d], [sum_d, count]] @ [scale, shift] =
    # [sum_cross, sum_target]. Solving the shift from its own cofactor instead
    # of back-substituting `(sum_target - scale * sum_prediction) / count`
    # keeps the whole solve behind one guarded denominator.
    scale = torch.where(
        degenerate,
        identity_scale,
        (count * sum_cross - sum_prediction * sum_target) / safe_determinant,
    )
    shift = torch.where(
        degenerate,
        identity_shift,
        (sum_prediction_squared * sum_target - sum_prediction * sum_cross)
        / safe_determinant,
    )
    return ScaleShift(scale.to(prediction.dtype), shift.to(prediction.dtype))


def apply_scale_shift(
    prediction: Float[Tensor, "batch height width"], fit: ScaleShift
) -> Float[Tensor, "batch height width"]:
    """Map `prediction` into the frame the fit was solved against."""
    return fit.scale[:, None, None] * prediction + fit.shift[:, None, None]


def _check_shapes(
    prediction: Float[Tensor, "batch height width"],
    target: Float[Tensor, "batch height width"],
    mask: Bool[Tensor, "batch height width"],
) -> None:
    if prediction.ndim != 3:
        raise ValueError(
            f"prediction must be (batch, height, width), got shape {tuple(prediction.shape)}"
        )
    if prediction.shape != target.shape:
        raise ValueError(
            f"target shape {tuple(target.shape)} does not match prediction shape "
            f"{tuple(prediction.shape)}"
        )
    if prediction.shape != mask.shape:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} does not match prediction shape "
            f"{tuple(prediction.shape)}"
        )
