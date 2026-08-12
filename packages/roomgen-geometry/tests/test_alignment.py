from itertools import pairwise

import pytest
import torch
from jaxtyping import TypeCheckError
from torch import Generator, Tensor

from roomgen_fakes import assert_gradient_is_finite_and_nonzero
from roomgen_geometry.alignment import (
    _MAX_SCALE,
    ScaleShift,
    align_scale_shift,
    apply_scale_shift,
)


def _masked_mean_squared_error(
    prediction: Tensor, target: Tensor, mask: Tensor
) -> Tensor:
    # `masked_depth_error` in one line, including the variance the fit refused
    # to explain by reflecting: a fit floored at zero scores flat and gradient-
    # free without it, so an alignment test that dropped it would be scoring a
    # different objective than the reward does.
    fit = align_scale_shift(prediction, target, mask)
    aligned = apply_scale_shift(prediction, fit)
    weight = mask.to(prediction.dtype)
    return (
        weight * (aligned - target) ** 2
    ).sum() / weight.sum() + fit.refused_variance.sum()


def _assert_is_identity_fit(scale: Tensor, shift: Tensor) -> None:
    torch.testing.assert_close(scale, torch.ones_like(scale))
    torch.testing.assert_close(shift, torch.zeros_like(shift))


def test_align_scale_shift_recovers_a_per_element_affine_map(
    generator: Generator,
) -> None:
    prediction = torch.rand((2, 16, 16), generator=generator)
    scale = torch.tensor([2.5, 0.4])
    shift = torch.tensor([-1.0, 3.0])
    target = scale[:, None, None] * prediction + shift[:, None, None]
    mask = torch.ones_like(prediction, dtype=torch.bool)

    fit = align_scale_shift(prediction, target, mask)

    torch.testing.assert_close(fit.scale, scale)
    torch.testing.assert_close(fit.shift, shift)


def test_align_scale_shift_ignores_pixels_outside_the_mask(
    generator: Generator,
) -> None:
    prediction = torch.rand((1, 16, 16), generator=generator)
    target = 3.0 * prediction - 2.0
    mask = torch.zeros_like(prediction, dtype=torch.bool)
    mask[:, :8] = True

    fit = align_scale_shift(
        torch.where(mask, prediction, torch.full_like(prediction, 9.0)),
        torch.where(mask, target, torch.full_like(target, -40.0)),
        mask,
    )

    torch.testing.assert_close(fit.scale, torch.tensor([3.0]))
    torch.testing.assert_close(fit.shift, torch.tensor([-2.0]))


def test_masked_error_is_invariant_to_an_affine_rescale_of_the_prediction(
    generator: Generator,
) -> None:
    prediction = torch.rand((1, 16, 16), generator=generator)
    target = torch.rand((1, 16, 16), generator=generator)
    mask = torch.ones_like(prediction, dtype=torch.bool)

    error = _masked_mean_squared_error(prediction, target, mask)
    rescaled_error = _masked_mean_squared_error(7.0 * prediction - 5.0, target, mask)

    assert float(error) > 0.0
    torch.testing.assert_close(error, rescaled_error)


def test_an_empty_mask_determines_nothing_and_gives_the_identity_fit(
    generator: Generator,
) -> None:
    prediction = torch.rand((1, 8, 8), generator=generator, requires_grad=True)
    target = torch.rand((1, 8, 8), generator=generator)
    mask = torch.zeros((1, 8, 8), dtype=torch.bool)

    fit = align_scale_shift(prediction, target, mask)

    _assert_is_identity_fit(fit.scale, fit.shift)
    (gradient,) = torch.autograd.grad(fit.scale.sum() + fit.shift.sum(), prediction)
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("masked_pixels", [1, 64], ids=["one pixel", "flat region"])
def test_a_mask_with_no_variance_matches_the_means_rather_than_guessing_a_slope(
    masked_pixels: int, generator: Generator
) -> None:
    # No slope is determined over either mask, but the level is: the fit that
    # says so is the one the capped fit tends to as the prediction flattens,
    # which is what makes the two regimes meet.
    prediction = torch.full((1, 8, 8), 0.25, requires_grad=True)
    target = torch.rand((1, 8, 8), generator=generator)
    mask = torch.zeros((1, 8, 8), dtype=torch.bool)
    mask.view(-1)[:masked_pixels] = True

    fit = align_scale_shift(prediction, target, mask)

    torch.testing.assert_close(fit.scale, torch.zeros(1))
    torch.testing.assert_close(fit.shift, target.flatten()[:masked_pixels].mean()[None])
    (gradient,) = torch.autograd.grad(fit.scale.sum() + fit.shift.sum(), prediction)
    assert torch.isfinite(gradient).all()


def test_the_fitted_scale_is_capped_rather_than_growing_as_the_prediction_flattens(
    generator: Generator,
) -> None:
    structure = torch.rand((1, 16, 16), generator=generator)
    mask = torch.ones_like(structure, dtype=torch.bool)

    # A prediction 1e-2 of the target's spread needs a scale of ~100 and gets
    # it; one 1e-6 of it would need ~1e6 and is held at the documented cap.
    conditioned = align_scale_shift(1e-2 * structure + 5.0, structure, mask)
    capped = align_scale_shift(1e-6 * structure + 5.0, structure, mask)

    assert float(conditioned.scale) == pytest.approx(100.0, rel=1e-3)
    assert float(capped.scale) == pytest.approx(_MAX_SCALE, rel=1e-9)


def test_the_fit_stays_continuous_across_the_flattening_it_cannot_follow(
    generator: Generator,
) -> None:
    # Either side of the point where the damped solve gives up on a slope, and
    # far past it. An undamped solve jumps here — the aligned map goes from
    # tracking the target to not being aligned at all — and the whole point of
    # the damping is that this sweep has no step in it.
    target = torch.rand((1, 32, 32), generator=generator)
    mask = torch.ones_like(target, dtype=torch.bool)
    flatnesses = [10 ** (-4.0 - tenth / 10) for tenth in range(51)]

    errors = []
    gradients = []
    for flatness in flatnesses:
        prediction = (flatness * target + 5.0).requires_grad_(True)
        error = _masked_mean_squared_error(prediction, target, mask)
        (gradient,) = torch.autograd.grad(error, prediction)
        errors.append(float(error.detach()))
        gradients.append(float(gradient.abs().max()))

    # No tenth of a decade of flattening moves the error by more than a twentieth
    # of itself, and the far end lands on the variance of the target, which is
    # exactly what a fit that has given up on the slope and matched the means
    # scores. Undamped, this sweep steps from ~0 to that variance in one jump.
    for previous, current in pairwise(errors):
        assert abs(current - previous) < 0.05 * max(previous, current)
    assert errors[-1] == pytest.approx(float(target.var(unbiased=False)), rel=1e-4)
    # The cap is what bounds the gradient: uncapped it grows like 1/flatness,
    # so the far end of this sweep would be eight orders above the near end.
    assert max(gradients) < 2.0
    assert min(gradients) > 0.0


def test_the_fit_rescales_the_prediction_but_never_reflects_it(
    generator: Generator,
) -> None:
    # A reflected prediction is an affine image of the target, so an
    # unconstrained fit lands on it exactly and scores a room turned inside out
    # as a perfect match. Near/far ordering is geometry, not part of the affine
    # freedom the backend leaves undetermined: the fit floors at zero, which is
    # the mean-matching fit, and reports the variance it declined to explain.
    target = torch.rand((1, 16, 16), generator=generator)
    mask = torch.ones_like(target, dtype=torch.bool)

    fit = align_scale_shift(-2.0 * target + 7.0, target, mask)

    torch.testing.assert_close(fit.scale, torch.zeros(1))
    torch.testing.assert_close(fit.shift, target.mean()[None])
    # Reflected, the prediction explains all of the target's variance, so all of
    # it is refused — and the energy that charges it scores twice the flat fit.
    torch.testing.assert_close(fit.refused_variance, target.var(unbiased=False)[None])


def test_the_energy_meets_the_scale_floor_with_no_step_across_it(
    generator: Generator,
) -> None:
    # The counterpart of the flattening sweep above, for the other end the fit
    # is constrained at. The prediction is an independent map with its target
    # component projected out plus `tilt` of the target, so the unconstrained
    # scale is proportional to `tilt` and crosses zero in the middle of the
    # sweep rather than somewhere the test would have to hunt for.
    target = torch.rand((1, 32, 32), generator=generator)
    mask = torch.ones_like(target, dtype=torch.bool)
    centred = target - target.mean()
    noise = torch.rand((1, 32, 32), generator=generator)
    orthogonal = noise - (noise * centred).sum() / (centred * centred).sum() * centred
    tilts = [(step - 25) / 250 for step in range(51)]

    errors = []
    gradients = []
    for tilt in tilts:
        prediction = (orthogonal + tilt * centred).requires_grad_(True)
        error = _masked_mean_squared_error(prediction, target, mask)
        (gradient,) = torch.autograd.grad(error, prediction)
        errors.append(float(error.detach()))
        gradients.append(float(gradient.abs().max()))

    flat_fit = float(target.var(unbiased=False))
    for previous, current in pairwise(errors):
        assert abs(current - previous) < 0.05 * flat_fit
    for previous, current in pairwise(gradients):
        assert abs(current - previous) < 0.05 * max(gradients)
    # At the crossing the fit is the flat one and scores its residual exactly;
    # either side of it the two branches are mirror images about that value, so
    # the penalty joins the fit with the same slope rather than at a kink.
    assert errors[25] == pytest.approx(flat_fit, rel=1e-6)
    for below, above in zip(errors[:25], reversed(errors[26:]), strict=True):
        assert below + above == pytest.approx(2.0 * flat_fit, rel=1e-6)
    # Monotone the whole way: more of the target in the prediction always scores
    # better, and reversing it always scores worse than carrying none of it.
    for previous, current in pairwise(errors):
        assert current < previous


def test_the_aligned_error_gradient_reaches_every_masked_pixel(
    generator: Generator,
) -> None:
    prediction = torch.rand((1, 8, 8), generator=generator, requires_grad=True)
    target = torch.rand((1, 8, 8), generator=generator)
    mask = torch.ones_like(target, dtype=torch.bool)

    error = _masked_mean_squared_error(prediction, target, mask)

    gradient = assert_gradient_is_finite_and_nonzero(error, prediction)
    assert (gradient.abs() > 0.0).all()


@pytest.mark.parametrize(
    ("prediction_shape", "target_shape", "mask_shape", "offending_tensor"),
    [
        ((8, 8), (8, 8), (8, 8), "prediction"),
        ((1, 8, 8), (1, 8, 4), (1, 8, 8), "target"),
        ((1, 8, 8), (1, 8, 8), (1, 4, 8), "mask"),
    ],
    ids=["not batched", "target mismatch", "mask mismatch"],
)
def test_align_scale_shift_names_the_tensor_whose_shape_does_not_fit(
    prediction_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    mask_shape: tuple[int, ...],
    offending_tensor: str,
) -> None:
    # Two guards fire on this, and which one gets there first depends on where
    # the code is running: the suite installs jaxtyping's runtime hook over the
    # package (see the root conftest), so the annotation rejects the call and
    # raises `TypeCheckError`; unhooked, the function's own check raises
    # `ValueError`. Both name the tensor at fault, which is the contract.
    with pytest.raises((TypeCheckError, ValueError), match=offending_tensor):
        align_scale_shift(
            torch.zeros(prediction_shape),
            torch.zeros(target_shape),
            torch.ones(mask_shape, dtype=torch.bool),
        )


def test_a_shape_annotation_with_no_hand_written_guard_is_still_enforced() -> None:
    # `apply_scale_shift` checks nothing itself: an unbatched map would silently
    # broadcast against the fit. This is the whole suite's coverage of the
    # jaxtyping import hook — if it stops being installed, this is what says so.
    with pytest.raises(TypeCheckError):
        apply_scale_shift(
            torch.zeros((8, 8)),
            ScaleShift(torch.ones(1), torch.zeros(1), torch.zeros(1)),
        )


def _room(generator: Generator) -> tuple[Tensor, Tensor]:
    """A disparity map and a mask over most of it."""
    disparity = torch.rand((2, 16, 16), generator=generator) + 0.5
    mask = torch.ones((2, 16, 16), dtype=torch.bool)
    mask[:, :2] = False
    return disparity, mask


def test_no_alignment_leaves_both_maps_exactly_where_they_stand(
    generator: Generator,
) -> None:
    prediction, mask = _room(generator)
    target = 3.0 * prediction + 1.0

    fit = align_scale_shift(prediction, target, mask, mode="none")

    assert torch.equal(fit.scale, torch.ones(2))
    assert torch.equal(fit.shift, torch.zeros(2))
    assert torch.equal(fit.refused_variance, torch.zeros(2))
    assert torch.equal(apply_scale_shift(prediction, fit), prediction)


def test_shift_only_matches_the_masked_means_and_nothing_else(
    generator: Generator,
) -> None:
    prediction, mask = _room(generator)
    offset = torch.tensor([0.25, -0.75])
    target = prediction + offset[:, None, None]

    fit = align_scale_shift(prediction, target, mask, mode="shift")

    assert torch.equal(fit.scale, torch.ones(2))
    assert torch.allclose(fit.shift, offset)
    # A pure offset is all a metric backend should ever have to give up, and
    # under this mode it is removed exactly.
    assert torch.allclose(apply_scale_shift(prediction, fit), target)


def test_shift_only_refuses_to_absorb_a_scale_the_full_fit_would_swallow(
    generator: Generator,
) -> None:
    # The experiment this mode exists for, in one assertion: a room rendered at
    # the wrong absolute size is free under `scale_shift` and charged under
    # `shift`. A metric backend is the only kind that can tell the difference.
    prediction, mask = _room(generator)
    target = 2.0 * prediction

    full = _masked_mean_squared_error(
        prediction, target, mask
    )  # scored through the affine fit
    shifted = align_scale_shift(prediction, target, mask, mode="shift")
    residual = apply_scale_shift(prediction, shifted) - target
    charged = (residual**2)[mask].mean()

    assert float(full) == pytest.approx(0.0, abs=1e-6)
    assert float(charged) > 0.05


@pytest.mark.parametrize("mode", ["scale_shift", "shift", "none"])
def test_every_mode_leaves_an_empty_mask_alone(mode: str) -> None:
    prediction = torch.full((1, 4, 4), 3.0)
    target = torch.zeros((1, 4, 4))
    empty = torch.zeros((1, 4, 4), dtype=torch.bool)

    fit = align_scale_shift(prediction, target, empty, mode=mode)  # type: ignore[arg-type]

    assert torch.equal(apply_scale_shift(prediction, fit), prediction)
    assert torch.equal(fit.refused_variance, torch.zeros(1))


@pytest.mark.parametrize("mode", ["scale_shift", "shift", "none"])
def test_every_mode_keeps_the_prediction_on_the_graph(
    mode: str, generator: Generator
) -> None:
    prediction, mask = _room(generator)
    prediction = prediction.requires_grad_(True)
    target = 0.7 * prediction.detach() + 0.2

    fit = align_scale_shift(prediction, target, mask, mode=mode)  # type: ignore[arg-type]
    aligned = apply_scale_shift(prediction, fit)

    assert_gradient_is_finite_and_nonzero((aligned**2).sum(), prediction)
