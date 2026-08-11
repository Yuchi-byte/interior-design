"""The structural view of a diffusers img2img pipeline that the guided loop drives.

Every member here is named for the diffusers attribute it stands for, so a real
`StableDiffusionImg2ImgPipeline` satisfies `Img2ImgPipeline` as it comes; the
Protocols exist because the loop steps the components itself, which a fake with
no weights can therefore stand in for.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from jaxtyping import Float
from torch import Tensor


class SampleOutput(Protocol):
    """What a UNet call and a VAE decode both return: the tensor under `.sample`."""

    sample: Tensor


class LatentDistribution(Protocol):
    """The posterior a VAE encoder returns, sampled from an explicit generator."""

    def sample(self, generator: torch.Generator | None = None) -> Tensor: ...


class EncoderOutput(Protocol):
    latent_dist: LatentDistribution


class SchedulerStepOutput(Protocol):
    prev_sample: Tensor


class AutoencoderConfig(Protocol):
    scaling_factor: float


class SchedulerConfig(Protocol):
    prediction_type: str


class NoisePredictor(Protocol):
    """The UNet: noisy latent and conditioning in, ε under `.sample` out."""

    def __call__(
        self,
        sample: Float[Tensor, "batch channels height width"],
        timestep: Tensor,
        encoder_hidden_states: Float[Tensor, "batch sequence embedding"],
    ) -> SampleOutput: ...


class LatentAutoencoder(Protocol):
    """The VAE, in the ``[-1, 1]`` image convention diffusers encodes and decodes in."""

    config: AutoencoderConfig

    def encode(self, x: Float[Tensor, "batch 3 height width"]) -> EncoderOutput: ...

    def decode(
        self, z: Float[Tensor, "batch channels height width"]
    ) -> SampleOutput: ...


class NoiseScheduler(Protocol):
    """The scheduler, in the epsilon parameterisation the guided loop's x̂₀ assumes.

    `alphas_cumprod` is indexed by *training* timestep, as the schedulers that
    subsample a trajectory (DDIM, PNDM) keep it.
    """

    config: SchedulerConfig
    timesteps: Tensor
    alphas_cumprod: Tensor

    def set_timesteps(
        self, num_inference_steps: int, device: torch.device | None = None
    ) -> None: ...

    def scale_model_input(self, sample: Tensor, timestep: Tensor) -> Tensor: ...

    def add_noise(
        self, original_samples: Tensor, noise: Tensor, timesteps: Tensor
    ) -> Tensor: ...

    def step(
        self, model_output: Tensor, timestep: Tensor, sample: Tensor
    ) -> SchedulerStepOutput: ...


@runtime_checkable
class Img2ImgPipeline(Protocol):
    """The part of a diffusers img2img pipeline the guided loop drives directly.

    The loop steps the scheduler itself rather than calling the pipeline, whose
    `__call__` runs under `no_grad` and would strand the guidance gradient.
    """

    unet: NoisePredictor
    vae: LatentAutoencoder
    scheduler: NoiseScheduler

    def encode_prompt(
        self,
        prompt: str,
        device: torch.device,
        num_images_per_prompt: int,
        do_classifier_free_guidance: bool,
    ) -> tuple[
        Float[Tensor, "batch sequence embedding"],
        Float[Tensor, "batch sequence embedding"] | None,
    ]: ...
