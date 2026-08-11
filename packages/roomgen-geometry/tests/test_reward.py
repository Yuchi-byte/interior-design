import torch
from torch import Generator, Tensor

from roomgen_fakes import assert_gradient_is_finite_and_nonzero
from roomgen_geometry.reward import masked_depth_error


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
