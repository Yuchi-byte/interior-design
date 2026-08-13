"""Closed-form least-squares removal of the positive-affine freedom in inverse depth."""

from typing import Final, Literal, NamedTuple

import torch
from jaxtyping import Bool, Float
from torch import Tensor

# How much of the affine freedom a comparison hands back to the prediction.
#
# `scale_shift` is the default and the only mode an affine-invariant backend
# (Depth Anything V2) may be scored under — see CLAUDE.md's "never compare raw
# depth maps". `shift` exists for *metric* backends, whose disparity is
# determined absolutely rather than up to a scale and shift: for those, fitting
# a scale is handing back the very degree of freedom the network resolved, and
# it is the freedom an over- or under-scaled room escapes through — guided arms
# remove ~90% of the depth energy under a scale-and-shift fit while the room's
# measured size ratio stays where the unguided control put it, or worse.
# Choosing `shift` for an affine-invariant backend measures affine mismatch and
# is wrong however good the number looks.
#
# Fitting *nothing* was measured too, and is not offered: raw metric residuals
# score best of any arm from a renoised photo and collapse from pure noise,
# where a global offset the walk cannot remove dominates a normalised step
# budget. `shift` is the mode that holds at both.
type AlignmentMode = Literal["scale_shift", "shift"]

# The normal equations of a scale/shift fit are ill-conditioned whenever the
# masked depth range is narrow, and float32 loses several digits there for no
# gain: the solve is a handful of reductions, so it always runs in float64 and
# the result is cast back to the caller's dtype.
_SOLVE_DTYPE: Final = torch.float64

# The 2x2 system's determinant is non-negative by Cauchy-Schwarz and vanishes
# exactly when the masked depths carry no variance (empty mask, one pixel, a
# flat region), so the solve needs a ridge rather than a threshold: a threshold
# leaves the fit jumping between the two regimes it separates, while damping the
# determinant by this fraction of its own scale drives the scale smoothly to
# zero as the variance vanishes. Being a fraction makes it dimensionless, so it
# bites the same way for disparity in [0, 1] and for disparity in [0, 1000], and
# it is 4000x above float64's rounding on the determinant — small enough that
# the fit it returns on any conditioned mask is the undamped one to eleven
# figures.
_DEGENERACY_DAMPING: Final = 1e-12

# The least-squares scale is `correlation * target_std / prediction_std`, so it
# only reaches this bound once the prediction's masked spread has fallen to a
# thousandth of the reference's — a prediction with no structure left to align.
# Capping it there bounds `d(aligned prediction)/d(prediction)` by the same
# factor, instead of letting the fit and its gradient grow without limit on the
# way down to the damped solve's zero.
_MAX_SCALE: Final = 1e3


class ScaleShift(NamedTuple):
    """An affine fit per batch element, and what constraining its sign cost.

    `refused_variance` is the share of the target's masked variance that only a
    *reflected* fit could have explained — zero wherever the unconstrained
    optimum was already positive, and otherwise exactly the residual reduction a
    negative scale would have bought. It is in the squared units of the target,
    so a caller scoring a mean squared residual charges it by adding it on.
    """

    scale: Float[Tensor, "batch"]
    shift: Float[Tensor, "batch"]
    refused_variance: Float[Tensor, "batch"]


def align_scale_shift(
    prediction: Float[Tensor, "batch height width"],
    target: Float[Tensor, "batch height width"],
    mask: Bool[Tensor, "batch height width"],
    *,
    mode: AlignmentMode = "scale_shift",
) -> ScaleShift:
    """Per-batch-element least-squares fit of `prediction` onto `target`.

    Args:
        mode: which parameters the fit is free to move. The default
            `scale_shift` is the full affine solve described below and is the
            only correct mode for an affine-invariant backend; `shift` solves
            for the offset alone at a fixed unit scale. See
            :data:`AlignmentMode` for when the looser one is legitimate.

    Minimises ``sum_i mask_i * (scale * prediction_i + shift - target_i)^2``
    **subject to a non-negative scale**, lightly ridge-damped and with `scale`
    capped at 1000. The damping and the cap are slack by many orders on any pair
    of depth maps that share structure — the fit returned there is the plain
    least-squares one to eleven figures — and both exist for the ill-conditioned
    end, where an undamped fit blows up like ``1 / prediction_std``. Damped and
    capped, it instead falls continuously to the mean-matching ``(0, masked mean
    of target)``: whatever the masked pixels determine is determined, and a mask
    over which the prediction carries no variance (one pixel, a flat region)
    determines the level but not the slope.

    The scale floor is not a numerical guard but the geometry: near/far ordering
    is a fact about the depth maps, not part of the affine freedom the backend
    leaves undetermined, so a prediction the fit could only match by reflecting
    it is one that has the room's depth ordering backwards. The floor is a
    projection — ``scale = max(0, unconstrained scale)``, which is the
    constrained optimum — and the reduction the reflection would have bought is
    reported as `refused_variance` rather than discarded, since a caller that
    only saw the projected fit would be looking at a flat one with no gradient.

    Only an *empty* mask, which determines nothing at all, falls back to the
    identity ``(1, 0)``. Differentiable in both maps throughout, with the
    gradient bounded by the cap.
    """
    _check_shapes(prediction, target, mask)

    # Excluded pixels are *replaced*, not weighted: `0 * NaN` is NaN, so a
    # single non-finite pixel anywhere in either map — including the region the
    # mask exists to exclude — would otherwise poison every sum below and come
    # back as a NaN gradient over the whole image.
    zero = torch.zeros((), dtype=_SOLVE_DTYPE, device=prediction.device)
    weight = mask.to(_SOLVE_DTYPE)
    predicted = torch.where(mask, prediction.to(_SOLVE_DTYPE), zero)
    reference = torch.where(mask, target.to(_SOLVE_DTYPE), zero)

    count = weight.flatten(1).sum(-1)
    sum_prediction = predicted.flatten(1).sum(-1)
    sum_target = reference.flatten(1).sum(-1)
    safe_count = count.clamp(min=1.0)
    # An empty mask determines nothing at all, and unlike a flat one it has no
    # mean to match; every mode leaves its shift at zero so the maps are scored
    # as they stand by the caller's own empty-mask handling.
    empty = count == 0
    unit = torch.ones_like(count)
    nothing = torch.zeros_like(count)

    match mode:
        case "shift":
            # The offset that matches the two masked means, which is the whole
            # of the least-squares solve once the scale is held at one. Nothing
            # can be refused at a fixed *positive* scale, so `refused_variance`
            # — the charge for a scale the floor rejected — is identically zero.
            shift = torch.where(
                empty, nothing, (sum_target - sum_prediction) / safe_count
            )
            return _cast(ScaleShift(unit, shift, nothing), prediction.dtype)
        case "scale_shift":
            pass

    sum_prediction_squared = (predicted * predicted).flatten(1).sum(-1)
    sum_cross = (predicted * reference).flatten(1).sum(-1)

    determinant = count * sum_prediction_squared - sum_prediction * sum_prediction
    damped = determinant + _DEGENERACY_DAMPING * count * sum_prediction_squared
    # The damping is itself zero when there is nothing to damp — an empty mask,
    # or one over which the prediction is identically zero — and `torch.where`
    # still evaluates the untaken branch and backpropagates its NaN, so the
    # denominator is sanitised *before* the division rather than the quotient
    # afterwards. A `where` rather than a Python `if` because the test is per
    # batch element, and branching on a tensor value would both break batching
    # and drop the gradient.
    safe_damped = torch.where(damped > 0.0, damped, torch.ones_like(damped))

    # Cramer's rule on [[sum_dd, sum_d], [sum_d, count]] @ [scale, shift] =
    # [sum_cross, sum_target], for the scale only. Its numerator carries the
    # masked covariance, which vanishes with the variance the damping is holding
    # the denominator up by, so the slope tends to zero rather than diverging.
    covariance = count * sum_cross - sum_prediction * sum_target
    slope = covariance / safe_damped
    scale = slope.clamp(0.0, _MAX_SCALE)
    # What the floor threw away: `min(slope, 0)^2 * masked variance of the
    # prediction`, written over the numerator so a slope past the cap is charged
    # at its uncapped size. Algebraically it is `covariance^2 / variance`, the
    # target variance a reflected fit would have explained, which is bounded by
    # the target's own variance however anti-correlated the prediction is — so
    # charging it doubles the flat fit's residual at worst and cannot swamp the
    # energy. It leaves the floor continuously: quadratic in a slope that is
    # itself continuous, so it and its gradient reach zero together as the
    # unconstrained optimum crosses into positive territory.
    refused_variance = covariance.clamp(max=0.0) ** 2 / (
        safe_count * safe_count * safe_damped
    )
    # The shift is back-substituted from the *capped* scale rather than solved
    # from its own cofactor: it is what makes the capped fit still match the two
    # masked means, and so what makes the fit continuous in the prediction where
    # the cap binds instead of jumping there.
    shift = (sum_target - scale * sum_prediction) / safe_count

    # An empty mask determines neither. It refuses nothing either — every sum
    # above is zero over it, so the penalty already is.
    scale = torch.where(empty, unit, scale)
    shift = torch.where(empty, nothing, shift)
    return _cast(ScaleShift(scale, shift, refused_variance), prediction.dtype)


def _cast(fit: ScaleShift, dtype: torch.dtype) -> ScaleShift:
    """Bring a fit solved in float64 back to the caller's precision."""
    return ScaleShift(
        fit.scale.to(dtype), fit.shift.to(dtype), fit.refused_variance.to(dtype)
    )


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
