import torch
from torch import Generator, Tensor
from torch.nn.functional import interpolate

from roomgen_fakes import FakeDepthEstimator, assert_gradient_is_finite_and_nonzero
from roomgen_geometry.alignment import align_scale_shift, apply_scale_shift


def _decode(latent: Tensor, size: tuple[int, int]) -> Tensor:
    # Stands in for the VAE: upsample to image resolution and squash into the
    # [0, 1] range a DepthEstimator expects, in ops that keep the latent on the
    # graph exactly as the real decoder must.
    return interpolate(latent, size=size, mode="bilinear").sigmoid()


def test_a_guidance_shaped_energy_steers_the_latent_it_was_computed_from(
    fake_depth_estimator: FakeDepthEstimator, room_photo: Tensor, generator: Generator
) -> None:
    reference_disparity = fake_depth_estimator(room_photo).detach()
    shell_mask = torch.ones_like(reference_disparity, dtype=torch.bool)
    latent = torch.rand((1, 3, 16, 16), generator=generator, requires_grad=True)

    predicted_disparity = fake_depth_estimator(_decode(latent, room_photo.shape[-2:]))
    fit = align_scale_shift(predicted_disparity, reference_disparity, shell_mask)
    aligned = apply_scale_shift(predicted_disparity, fit)
    weight = shell_mask.to(aligned.dtype)
    energy = (weight * (aligned - reference_disparity) ** 2).sum() / weight.sum()

    gradient = assert_gradient_is_finite_and_nonzero(energy, latent)
    assert gradient.shape == latent.shape
