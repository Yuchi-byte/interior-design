"""A weightless stand-in for the diffusers img2img pipeline the guided loop drives."""

from dataclasses import dataclass, field
from typing import Final

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Generator, Tensor
from torch.nn.functional import avg_pool2d, interpolate

# The Stable Diffusion 1.5 geometry the loop is written against: a 4-channel
# latent at an eighth of the image resolution, over a 1000-step trajectory.
LATENT_CHANNELS: Final = 4
DOWNSCALE_FACTOR: Final = 8
TRAIN_TIMESTEPS: Final = 1000
_EMBEDDING_SEQUENCE: Final = 7
_EMBEDDING_WIDTH: Final = 5


@dataclass(frozen=True)
class _Sampled:
    sample: Tensor


@dataclass(frozen=True)
class _PreviousSample:
    prev_sample: Tensor


@dataclass(frozen=True)
class _LatentDistribution:
    mean: Tensor
    spread: float

    def sample(self, generator: Generator | None = None) -> Tensor:
        noise = torch.randn(
            self.mean.shape,
            generator=generator,
            dtype=self.mean.dtype,
            device=self.mean.device,
        )
        return self.mean + self.spread * noise


@dataclass(frozen=True)
class _Encoded:
    latent_dist: _LatentDistribution


@dataclass(frozen=True)
class _AutoencoderConfig:
    scaling_factor: float = 0.18215


@dataclass(frozen=True)
class FakeVae:
    """A `LatentAutoencoder` that needs no checkpoint: pooling down, a smooth tanh back up.

    Args:
        posterior_spread: the width of the encoder posterior the loop draws its
            starting latent from, and the only randomness the VAE contributes.
    """

    posterior_spread: float = 0.05
    config: _AutoencoderConfig = field(default_factory=_AutoencoderConfig)

    # Both directions carry the scaling factor, as the real VAE does: the raw
    # latent it emits is a factor of `1 / scaling_factor` larger than the
    # unit-variance one the sampler works in, and the loop's scaling is what
    # converts between them.
    @jaxtyped(typechecker=beartype)
    def encode(self, x: Float[Tensor, "batch 3 height width"]) -> _Encoded:
        pooled = avg_pool2d(x, DOWNSCALE_FACTOR)
        mean = torch.cat([pooled, pooled.mean(dim=1, keepdim=True)], dim=1)
        return _Encoded(
            _LatentDistribution(
                mean / self.config.scaling_factor, self.posterior_spread
            )
        )

    @jaxtyped(typechecker=beartype)
    def decode(self, z: Float[Tensor, "batch 4 height width"]) -> _Sampled:
        # Every latent channel reaches the image, so a gradient test cannot pass
        # on a decoder that only ever moves three of the four.
        coloured = 0.5 * self.config.scaling_factor * (z[:, :3] + z[:, 3:])
        upsampled = interpolate(
            coloured, scale_factor=DOWNSCALE_FACTOR, mode="bilinear"
        )
        # tanh, not a clamp: the decode stays strictly inside the [-1, 1] the
        # loop maps to [0, 1], so no pixel saturates and silently drops the
        # guidance gradient the tests are asserting on.
        return _Sampled(torch.tanh(upsampled))


@dataclass(frozen=True)
class FakeUNet:
    """A `NoisePredictor` that needs no checkpoint: a smooth function of latent, timestep and conditioning."""

    conditioning_weight: float = 0.1

    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        sample: Float[Tensor, "batch 4 height width"],
        timestep: Tensor,
        encoder_hidden_states: Float[Tensor, "batch sequence embedding"],
    ) -> _Sampled:
        conditioning = encoder_hidden_states.mean(dim=(1, 2))[:, None, None, None]
        noisiness = float(timestep) / TRAIN_TIMESTEPS
        # Bounded like the unit-variance ε a trained UNet predicts, so that the
        # trajectory this drives stays in the range the decoder is smooth over
        # even once classifier-free guidance has amplified it.
        return _Sampled(
            torch.tanh(
                noisiness * sample + self.conditioning_weight * conditioning
            ).expand_as(sample)
        )


@dataclass(frozen=True)
class _SchedulerConfig:
    prediction_type: str = "epsilon"


class FakeScheduler:
    """A `NoiseScheduler` that needs no checkpoint: a deterministic DDIM over a linear ᾱ.

    Args:
        prediction_type: what the paired noise predictor is taken to emit; the
            guided loop only drives ``"epsilon"``.
    """

    def __init__(self, *, prediction_type: str = "epsilon") -> None:
        self.config = _SchedulerConfig(prediction_type)
        self.alphas_cumprod = torch.linspace(0.9999, 0.02, TRAIN_TIMESTEPS)
        self.timesteps = torch.empty(0, dtype=torch.int64)

    def set_timesteps(
        self, num_inference_steps: int, device: torch.device | None = None
    ) -> None:
        spacing = TRAIN_TIMESTEPS // num_inference_steps
        self.timesteps = torch.arange(
            TRAIN_TIMESTEPS - 1, -1, -spacing, device=device, dtype=torch.int64
        )[:num_inference_steps]

    def scale_model_input(self, sample: Tensor, timestep: Tensor) -> Tensor:
        return sample

    def add_noise(
        self, original_samples: Tensor, noise: Tensor, timesteps: Tensor
    ) -> Tensor:
        alpha = self.alphas_cumprod[timesteps][:, None, None, None]
        return alpha.sqrt() * original_samples + (1.0 - alpha).sqrt() * noise

    def step(
        self, model_output: Tensor, timestep: Tensor, sample: Tensor
    ) -> _PreviousSample:
        alpha = self.alphas_cumprod[timestep]
        predicted_original = (
            sample - (1.0 - alpha).sqrt() * model_output
        ) / alpha.sqrt()
        alpha_previous = self._next_alpha_cumprod(timestep)
        return _PreviousSample(
            alpha_previous.sqrt() * predicted_original
            + (1.0 - alpha_previous).sqrt() * model_output
        )

    def _next_alpha_cumprod(self, timestep: Tensor) -> Tensor:
        index = int((self.timesteps == timestep).nonzero()[0])
        if index + 1 == len(self.timesteps):
            return torch.ones((), dtype=self.alphas_cumprod.dtype)
        return self.alphas_cumprod[self.timesteps[index + 1]]


@dataclass(frozen=True)
class FakePipeline:
    """An `Img2ImgPipeline` standing in for Stable Diffusion img2img, weights and all.

    Prompt embeddings are a deterministic function of the prompt string, so two
    runs of the same prompt condition identically and two different prompts do
    not.
    """

    vae: FakeVae = field(default_factory=FakeVae)
    unet: FakeUNet = field(default_factory=FakeUNet)
    scheduler: FakeScheduler = field(default_factory=FakeScheduler)

    def encode_prompt(
        self,
        prompt: str,
        device: torch.device,
        num_images_per_prompt: int,
        do_classifier_free_guidance: bool,
    ) -> tuple[Tensor, Tensor | None]:
        shape = (num_images_per_prompt, _EMBEDDING_SEQUENCE, _EMBEDDING_WIDTH)
        # A stable digest of the prompt: `hash` is salted per process, which
        # would make a fixed-seed run reproducible only within one interpreter.
        value = sum(ord(character) for character in prompt) % 97 / 97.0
        conditional = torch.full(shape, value, device=device)
        if not do_classifier_free_guidance:
            return conditional, None
        return conditional, torch.zeros(shape, device=device)
