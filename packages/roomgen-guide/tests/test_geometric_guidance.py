import torch
from torch import Generator, Tensor

from roomgen_fakes import FakeDepthEstimator, assert_gradient_is_finite_and_nonzero
from roomgen_guide.guidance.base import Guidance
from roomgen_guide.guidance.geometric import GeometricGuidance
from roomgen_guide.sampling.guided import decode_latent
from roomgen_guide_fakes import LATENT_CHANNELS, FakePipeline


def _brightened(image: Tensor, amount: float, region: tuple[slice, slice]) -> Tensor:
    changed = image.clone()
    changed[:, :, region[0], region[1]] += amount
    return changed


def test_the_energy_vanishes_on_the_photo_its_reference_was_frozen_from(
    geometric_guidance: GeometricGuidance, room_photo: Tensor
) -> None:
    assert isinstance(geometric_guidance, Guidance)
    assert float(geometric_guidance(room_photo)) < 1e-10


def test_the_energy_grows_with_the_drift_of_the_sample_from_the_photo(
    geometric_guidance: GeometricGuidance, room_photo: Tensor
) -> None:
    shell = (slice(0, 32), slice(0, 32))

    small = float(geometric_guidance(_brightened(room_photo, 0.05, shell)))
    large = float(geometric_guidance(_brightened(room_photo, 0.2, shell)))

    assert 0.0 < small < large


def test_the_energy_ignores_the_affine_freedom_the_depth_backend_leaves(
    geometric_guidance: GeometricGuidance, room_photo: Tensor
) -> None:
    rescaled_backend = GeometricGuidance(
        depth_estimator=FakeDepthEstimator(scale=3.0, shift=-2.0),
        reference=geometric_guidance.reference,
    )
    drifted = _brightened(room_photo, 0.2, (slice(0, 32), slice(0, 32)))

    torch.testing.assert_close(rescaled_backend(drifted), geometric_guidance(drifted))


def test_the_energy_scores_only_the_masked_shell(
    geometric_guidance: GeometricGuidance, room_photo: Tensor
) -> None:
    # The fixture's mask excludes the bottom-right 8x8 corner.
    outside = _brightened(room_photo, 0.5, (slice(-8, None), slice(-8, None)))
    inside = _brightened(room_photo, 0.5, (slice(0, 8), slice(0, 8)))

    assert float(geometric_guidance(outside)) < 1e-10
    assert float(geometric_guidance(inside)) > 0.0


def test_the_energy_gradient_reaches_the_latent_through_the_decoder(
    geometric_guidance: GeometricGuidance,
    fake_pipeline: FakePipeline,
    generator: Generator,
) -> None:
    latent = torch.randn(
        (1, LATENT_CHANNELS, 8, 8), generator=generator, requires_grad=True
    )

    energy = geometric_guidance(decode_latent(fake_pipeline.vae, latent))

    gradient = assert_gradient_is_finite_and_nonzero(energy, latent)
    assert gradient.shape == latent.shape
