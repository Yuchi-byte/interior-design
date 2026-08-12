import pytest
import torch
from torch import Generator, Tensor

from roomgen_fakes import assert_gradient_is_finite_and_nonzero
from roomgen_geometry.reward import gradient_matching_error, masked_depth_error


def _disparity(generator: Generator) -> Tensor:
    return torch.rand((2, 8, 8), generator=generator)


def test_error_vanishes_for_an_affine_rescaling_of_the_target(
    generator: Generator,
) -> None:
    target = _disparity(generator)
    mask = torch.ones_like(target, dtype=torch.bool)

    error = masked_depth_error(3.0 * target - 2.0, target, mask)

    torch.testing.assert_close(error, torch.zeros(2), atol=1e-6, rtol=0.0)


def test_error_grows_with_the_size_of_the_drift(generator: Generator) -> None:
    target = _disparity(generator)
    mask = torch.ones_like(target, dtype=torch.bool)
    drift = _disparity(generator)

    small = masked_depth_error(target + 0.1 * drift, target, mask)
    large = masked_depth_error(target + 0.5 * drift, target, mask)

    assert bool((small < large).all())
    assert bool((small > 0.0).all())


def test_pixels_outside_the_mask_do_not_count(generator: Generator) -> None:
    target = _disparity(generator)
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[:, 4:] = False
    prediction = target.clone()
    prediction[:, 4:] += 10.0

    error = masked_depth_error(prediction, target, mask)

    torch.testing.assert_close(error, torch.zeros(2), atol=1e-6, rtol=0.0)


def test_an_empty_mask_scores_zero_rather_than_a_nan(generator: Generator) -> None:
    target = _disparity(generator)
    prediction = target + 1.0

    error = masked_depth_error(
        prediction, target, torch.zeros_like(target, dtype=torch.bool)
    )

    assert torch.equal(error, torch.zeros(2))


def test_the_error_gradient_reaches_the_prediction(generator: Generator) -> None:
    target = _disparity(generator)
    prediction = (target + 0.3 * _disparity(generator)).requires_grad_(True)
    mask = torch.ones_like(target, dtype=torch.bool)

    gradient = assert_gradient_is_finite_and_nonzero(
        masked_depth_error(prediction, target, mask).sum(), prediction
    )

    assert gradient.shape == prediction.shape


@pytest.mark.parametrize(
    "poison", [float("nan"), float("inf"), -float("inf")], ids=["nan", "inf", "-inf"]
)
def test_a_non_finite_pixel_outside_the_mask_changes_nothing(
    poison: float, generator: Generator
) -> None:
    # The mask exists to say which pixels count, so a garbage pixel it excludes
    # must not be able to decide the score — and `0 * NaN` is NaN, so weighting
    # such a pixel to zero does not exclude it. Backwards it is worse than the
    # score: the fitted scale's gradient sums over every pixel of the map, so
    # one excluded NaN returns a NaN gradient over the whole image.
    target = _disparity(generator)
    clean = target + 0.3 * _disparity(generator)
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[:, 6:] = False
    poisoned_prediction = clean.clone()
    poisoned_prediction[:, 7, 3] = poison
    poisoned_target = target.clone()
    poisoned_target[:, 6, 1] = poison

    clean_source = clean.clone().requires_grad_(True)
    clean_error = masked_depth_error(clean_source, target, mask)
    clean_gradient = assert_gradient_is_finite_and_nonzero(
        clean_error.sum(), clean_source
    )
    poisoned_source = poisoned_prediction.requires_grad_(True)
    poisoned_error = masked_depth_error(poisoned_source, poisoned_target, mask)
    poisoned_gradient = assert_gradient_is_finite_and_nonzero(
        poisoned_error.sum(), poisoned_source
    )

    assert torch.equal(poisoned_error.detach(), clean_error.detach())
    assert torch.equal(poisoned_gradient, clean_gradient)


def test_a_non_finite_pixel_inside_the_mask_is_left_loud(generator: Generator) -> None:
    # The other side of the contract above: the scored region is not sanitised,
    # so a depth backend that starts emitting NaN is a visible failure.
    target = _disparity(generator)
    prediction = target + 0.3 * _disparity(generator)
    prediction[:, 1, 1] = float("nan")

    error = masked_depth_error(
        prediction, target, torch.ones_like(target, dtype=torch.bool)
    )

    assert bool(torch.isnan(error).all())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_a_half_precision_map_is_scored_by_a_reduction_that_cannot_overflow(
    dtype: torch.dtype, generator: Generator
) -> None:
    # The shape a real run has: a 512x512 disparity field of Depth Anything's
    # magnitudes, a quarter of a million masked pixels. Summing their squared
    # residuals in float16 saturates past 65504 and comes back inf, then NaN
    # once divided — and a NaN energy backpropagates as an all-zero gradient,
    # which is guidance silently switched off while sampling looks healthy.
    reference = 20.0 * torch.rand((1, 512, 512), generator=generator)
    prediction = (reference + torch.randn((1, 512, 512), generator=generator)).to(dtype)
    target = reference.to(dtype)
    mask = torch.rand((1, 512, 512), generator=generator) > 0.4
    prediction.requires_grad_(True)

    error = masked_depth_error(prediction, target, mask)

    assert error.dtype == torch.float32
    gradient = assert_gradient_is_finite_and_nonzero(error.sum(), prediction)
    assert gradient.dtype == dtype
    # Not merely finite: the same direction the full-precision energy gives.
    full_precision = (reference + 0.0).requires_grad_(True)
    expected = assert_gradient_is_finite_and_nonzero(
        masked_depth_error(
            full_precision + (prediction.detach().float() - reference), target, mask
        ).sum(),
        full_precision,
    )
    cosine = torch.nn.functional.cosine_similarity(
        gradient.float().flatten(), expected.flatten(), dim=0
    )
    assert float(cosine) > 0.99


def test_the_energy_is_float32_whatever_the_maps_arrive_in(
    generator: Generator,
) -> None:
    target = _disparity(generator)
    mask = torch.ones_like(target, dtype=torch.bool)

    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        error = masked_depth_error((target + 1.0).to(dtype), target.to(dtype), mask)
        assert error.dtype == torch.float32, dtype


def test_the_gradient_matching_term_takes_two_disparity_maps_and_a_mask(
    generator: Generator,
) -> None:
    target = _disparity(generator)
    mask = torch.ones_like(target, dtype=torch.bool)

    with pytest.raises(NotImplementedError):
        gradient_matching_error(_disparity(generator), target, mask, levels=4)
