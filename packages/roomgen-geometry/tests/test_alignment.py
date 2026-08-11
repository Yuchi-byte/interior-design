import pytest
import torch
from roomgen_fakes import assert_gradient_is_finite_and_nonzero
from torch import Generator, Tensor

from roomgen_geometry.alignment import align_scale_shift, apply_scale_shift


def _masked_mean_squared_error(
    prediction: Tensor, target: Tensor, mask: Tensor
) -> Tensor:
    aligned = apply_scale_shift(prediction, align_scale_shift(prediction, target, mask))
    weight = mask.to(prediction.dtype)
    return (weight * (aligned - target) ** 2).sum() / weight.sum()


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


@pytest.mark.parametrize("masked_pixels", [0, 1], ids=["empty mask", "one pixel"])
def test_an_undetermined_fit_is_the_identity_and_keeps_the_gradient_finite(
    masked_pixels: int, generator: Generator
) -> None:
    prediction = torch.rand((1, 8, 8), generator=generator, requires_grad=True)
    target = torch.rand((1, 8, 8), generator=generator)
    mask = torch.zeros((1, 8, 8), dtype=torch.bool)
    mask.view(-1)[:masked_pixels] = True

    fit = align_scale_shift(prediction, target, mask)

    _assert_is_identity_fit(fit.scale, fit.shift)
    (gradient,) = torch.autograd.grad(fit.scale.sum() + fit.shift.sum(), prediction)
    assert torch.isfinite(gradient).all()


def test_a_flat_masked_region_gives_the_identity_fit_rather_than_a_huge_scale() -> None:
    prediction = torch.full((1, 8, 8), 0.25)
    target = torch.full((1, 8, 8), 4.0)
    mask = torch.ones_like(prediction, dtype=torch.bool)

    fit = align_scale_shift(prediction, target, mask)

    _assert_is_identity_fit(fit.scale, fit.shift)


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
    ("prediction_shape", "target_shape", "mask_shape"),
    [
        ((8, 8), (8, 8), (8, 8)),
        ((1, 8, 8), (1, 8, 4), (1, 8, 8)),
        ((1, 8, 8), (1, 8, 8), (1, 4, 8)),
    ],
    ids=["not batched", "target mismatch", "mask mismatch"],
)
def test_align_scale_shift_rejects_mismatched_shapes(
    prediction_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    mask_shape: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="shape"):
        align_scale_shift(
            torch.zeros(prediction_shape),
            torch.zeros(target_shape),
            torch.ones(mask_shape, dtype=torch.bool),
        )
