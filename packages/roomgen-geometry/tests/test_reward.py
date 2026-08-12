import pytest
import torch
from torch import Generator, Tensor

from roomgen_fakes import assert_gradient_is_finite_and_nonzero
from roomgen_geometry.reward import masked_depth_error


def _disparity(generator: Generator) -> Tensor:
    return torch.rand((2, 8, 8), generator=generator)


def _room_disparity(size: int = 32) -> Tensor:
    """A room-shaped reference: a dominant floor ramp with wall structure on it."""
    rows, columns = torch.meshgrid(
        torch.linspace(0.0, 1.0, size), torch.linspace(0.0, 1.0, size), indexing="ij"
    )
    return (1.4 - rows + 0.15 * torch.sin(7.0 * columns) * (rows < 0.5))[None]


def _featureless_ramp(size: int = 32) -> Tensor:
    """The degenerate render: a smooth ramp running the other way, no structure."""
    rows, _ = torch.meshgrid(
        torch.linspace(0.0, 1.0, size), torch.linspace(0.0, 1.0, size), indexing="ij"
    )
    return (0.9 * rows + 0.05)[None]


def _flat_fit_residual(target: Tensor) -> float:
    """What a fit that has given up on a slope and matched the means scores."""
    return float(target.var(unbiased=False))


def _correlation(prediction: Tensor, target: Tensor) -> float:
    pair = torch.stack([prediction.flatten(), target.flatten()])
    return float(torch.corrcoef(pair)[0, 1])


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


def test_a_reversed_depth_ordering_scores_above_the_flat_fit() -> None:
    # The measured failure, in miniature. On MFM v2's kitchen cells the fit came
    # back with scales of -0.2 to -0.9 on renders whose depth read inside out or
    # was a featureless ramp, and the sign flip let them score ~95% below the
    # control — the reduction was the alignment's, not the render's. A ramp
    # running the wrong way scored 5.5% of the flat fit here before the scale
    # was floored; every anti-correlated prediction now scores above it.
    target = _room_disparity()
    mask = torch.ones_like(target, dtype=torch.bool)
    flat_fit = _flat_fit_residual(target)
    ramp = _featureless_ramp()

    reversed_error = float(masked_depth_error(ramp, target, mask))
    aligned_error = float(masked_depth_error(-ramp, target, mask))

    assert _correlation(ramp, target) < -0.9
    assert reversed_error > flat_fit
    # The same ramp the right way up is what the objective is for: it carries
    # the room's near/far ordering, and keeps every bit of credit it did before.
    assert aligned_error == pytest.approx(0.055 * flat_fit, rel=0.1)


@pytest.mark.parametrize(
    "reflection", [-1.0, -0.2, -7.0], ids=["mirror", "shallow", "steep"]
)
def test_the_energy_is_invariant_to_a_positive_rescale_but_not_to_a_reflection(
    reflection: float, generator: Generator
) -> None:
    # The affine freedom the depth backend leaves undetermined is a positive
    # scale and a shift. A negative scale is not a change of frame but a
    # different room, and the energy has to be able to say so.
    target = _room_disparity()
    mask = torch.ones_like(target, dtype=torch.bool)
    prediction = target + 0.05 * torch.rand(target.shape, generator=generator)

    error = masked_depth_error(prediction, target, mask)

    torch.testing.assert_close(
        masked_depth_error(7.0 * prediction - 5.0, target, mask), error
    )
    assert float(masked_depth_error(reflection * prediction, target, mask)) > float(
        error
    ) + _flat_fit_residual(target)


def test_an_inverted_prediction_is_pushed_back_toward_the_rooms_own_ordering(
    generator: Generator,
) -> None:
    # A floored scale on its own would leave the energy flat in the prediction —
    # correctly high, and with no direction out of the state that earned it.
    # What the refused variance buys is this: the gradient at an inverted
    # prediction points at un-inverting it, and following it does.
    target = _room_disparity()
    mask = torch.ones_like(target, dtype=torch.bool)
    inverted = -target + 0.1 * torch.rand(target.shape, generator=generator)
    assert _correlation(inverted, target) < -0.99

    source = inverted.clone().requires_grad_(True)
    error = masked_depth_error(source, target, mask)
    gradient = assert_gradient_is_finite_and_nonzero(error.sum(), source)

    stepped = inverted - 0.1 * gradient / gradient.square().mean().sqrt()
    assert float(masked_depth_error(stepped, target, mask)) < float(error.detach())
    assert _correlation(stepped, target) > _correlation(inverted, target)

    # And it is an escape, not a nudge: a plain normalised descent on the energy
    # walks the prediction from inside out to aligned within a few dozen steps.
    state = inverted.clone()
    for _ in range(60):
        source = state.clone().requires_grad_(True)
        (step,) = torch.autograd.grad(
            masked_depth_error(source, target, mask).sum(), source
        )
        state = state - 0.02 * step / step.square().mean().sqrt()
    assert _correlation(state, target) > 0.9
    assert float(masked_depth_error(state, target, mask)) < 0.1 * _flat_fit_residual(
        target
    )


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
