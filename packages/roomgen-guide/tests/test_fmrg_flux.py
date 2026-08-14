"""`FluxFlowMapBackend`'s three conversions, against stubs rather than FLUX.

The adapter is four small calls and every one of them is a unit conversion the
rest of the stack trusts silently: a sigma grid that decides which noise levels
the walk visits at all, a ``[-1, 1] -> [0, 1]`` remap the reward is scored
through, and FLUX's shift-*and*-scale latent normalisation. None of the three
fails loudly when it is wrong — the walk runs, the images look like images, and
the numbers are wrong — so each is pinned here. No FLUX weights are involved:
the schedule goes through a real `FlowMatchEulerDiscreteScheduler` configured as
FLUX.1-dev's is, and everything else through stubs.
"""

from dataclasses import dataclass, field
from itertools import pairwise
from typing import Final

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from jaxtyping import Float
from torch import Generator, Tensor

from roomgen_guide.adapters.fmrg_flux import (
    DEFAULT_TEXT_GUIDANCE_SCALE,
    FluxFlowMapBackend,
    register_image_context_prompt,
    select_cached_prompt,
)

# FLUX.1-dev's VAE normalisation, which is a shift *and* a scale where Stable
# Diffusion's is a scale alone — the shift is the term an adapter written from
# SD habits drops, and dropping it biases every latent by 0.116 without ever
# leaving the range the decoder accepts.
_SCALING_FACTOR: Final = 0.3611
_SHIFT_FACTOR: Final = 0.1159

# A 1024x1024 image: 128x128 latents, packed two-by-two into 4096 tokens, which
# is `max_image_seq_len` and so the top of FLUX's resolution-dependent shift.
_LATENT_SIDE: Final = 128

# The sigmas FLUX.1-dev's own scheduler produces for the pre-shift grid this
# adapter builds — `linspace(start_sigma, start_sigma / steps, steps)` — at
# 1024x1024, verified against `FlowMatchEulerDiscreteScheduler.sigmas` to 2.9e-8
# and hardcoded because the schedule is the one part of the walk that has no
# runtime check at all: a grid that started somewhere else, or stepped somewhere
# else, produces a perfectly healthy sample of a different trajectory.
#
# The closed form behind them, for a reader checking the numbers rather than
# re-running diffusers: at 4096 tokens the shift is `max_shift` = 1.15, and each
# pre-shift sigma is warped by `exp(mu) / (exp(mu) + 1 / sigma - 1)`. That is
# why `start_sigma` 0.6 begins the walk at 0.826 and not at 0.6.
_FLUX_SIGMAS: Final = {
    (4, 0.6): (0.82570166, 0.720980286, 0.575102905, 0.357874573, 0.0),
    (4, 1.0): (1.0, 0.904530762, 0.759510925, 0.512844116, 0.0),
    (8, 0.6): (
        0.82570166,
        0.777314331,
        0.720980286,
        0.654566772,
        0.575102905,
        0.478322968,
        0.357874573,
        0.203865845,
        0.0,
    ),
    (8, 1.0): (
        1.0,
        0.956723694,
        0.904530762,
        0.840348755,
        0.759510925,
        0.654566772,
        0.512844116,
        0.310901062,
        0.0,
    ),
}


@dataclass
class _StubLatentDistribution:
    encoded: Tensor
    generators: list[Generator | None]

    def sample(self, generator: Generator | None = None) -> Tensor:
        self.generators.append(generator)
        return self.encoded


@dataclass
class _StubEncoderOutput:
    latent_dist: _StubLatentDistribution


@dataclass
class _StubVaeConfig:
    scaling_factor: float = _SCALING_FACTOR
    shift_factor: float = _SHIFT_FACTOR


@dataclass
class _StubVae:
    """A VAE whose posterior mean is its input, so the remaps are all that shows."""

    config: _StubVaeConfig = field(default_factory=_StubVaeConfig)
    seen: list[Tensor] = field(default_factory=list)
    generators: list[Generator | None] = field(default_factory=list)

    def encode(self, x: Float[Tensor, "batch 3 height width"]) -> _StubEncoderOutput:
        self.seen.append(x)
        return _StubEncoderOutput(_StubLatentDistribution(x, self.generators))


@dataclass
class _StubPipeline:
    vae: _StubVae
    scheduler: FlowMatchEulerDiscreteScheduler
    transformer: object = None


@dataclass
class _StubSampler:
    """FMRG's `FluxFlowMapSampler` reduced to what the adapter calls on it."""

    pipeline: _StubPipeline
    dtype: torch.dtype = torch.float32
    cached_prompt_embeds: Tensor = field(default_factory=lambda: torch.zeros(1, 2, 4))
    cached_pooled_prompt_embeds: Tensor = field(
        default_factory=lambda: torch.zeros(1, 4)
    )
    prompt_cache: dict[str, tuple[Tensor, Tensor]] = field(default_factory=dict)
    decoded: Tensor | None = None
    vector_calls: list[dict[str, object]] = field(default_factory=list)

    def decode(self, z: Tensor) -> Tensor:
        return z if self.decoded is None else self.decoded

    def predict_vector(
        self,
        z: Tensor,
        t_cur: float,
        prompt_embeds: Tensor | None = None,
        pooled_prompt_embeds: Tensor | None = None,
        guidance_scale: float = DEFAULT_TEXT_GUIDANCE_SCALE,
        t_next: float | None = None,
    ) -> Tensor:
        self.vector_calls.append(
            {
                "t_cur": t_cur,
                "t_next": t_next,
                "guidance_scale": guidance_scale,
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
            }
        )
        return torch.zeros_like(z)


def _flux_scheduler() -> FlowMatchEulerDiscreteScheduler:
    """FLUX.1-dev's scheduler config, which is what makes the shift what it is."""
    return FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=3.0,
        use_dynamic_shifting=True,
        base_shift=0.5,
        max_shift=1.15,
        base_image_seq_len=256,
        max_image_seq_len=4096,
    )


@pytest.fixture
def sampler() -> _StubSampler:
    return _StubSampler(_StubPipeline(_StubVae(), _flux_scheduler()))


@pytest.fixture
def backend(sampler: _StubSampler) -> FluxFlowMapBackend:
    return FluxFlowMapBackend(sampler=sampler)


@pytest.mark.parametrize(("steps", "start_sigma"), sorted(_FLUX_SIGMAS))
def test_the_schedule_is_fluxs_own_shifted_grid(
    backend: FluxFlowMapBackend, steps: int, start_sigma: float
) -> None:
    latent = torch.zeros(1, 16, _LATENT_SIDE, _LATENT_SIDE)

    sigmas = backend.schedule(steps=steps, start_sigma=start_sigma, latent=latent)

    assert len(sigmas) == steps + 1
    assert sigmas == pytest.approx(_FLUX_SIGMAS[steps, start_sigma], abs=1e-6)


def test_the_schedule_descends_to_the_data_end(backend: FluxFlowMapBackend) -> None:
    latent = torch.zeros(1, 16, _LATENT_SIDE, _LATENT_SIDE)

    sigmas = backend.schedule(steps=8, start_sigma=0.6, latent=latent)

    assert all(a > b for a, b in pairwise(sigmas))
    assert sigmas[-1] == 0.0


def test_the_shift_is_read_off_the_latent_it_was_handed(
    backend: FluxFlowMapBackend,
) -> None:
    # FLUX warps its schedule by sequence length, so the same `start_sigma`
    # begins two resolutions at two different noise levels. The adapter takes
    # the latent as an argument for exactly this, and a 512x512 walk that
    # borrowed a 1024x1024 grid would be renoised further than it asked for.
    small = backend.schedule(
        steps=4, start_sigma=0.6, latent=torch.zeros(1, 16, 64, 64)
    )
    large = backend.schedule(
        steps=4, start_sigma=0.6, latent=torch.zeros(1, 16, _LATENT_SIDE, _LATENT_SIDE)
    )

    assert small[0] != large[0]
    assert small[0] < large[0]


def test_the_decode_maps_the_vaes_range_onto_the_rewards(
    sampler: _StubSampler, backend: FluxFlowMapBackend
) -> None:
    # FLUX's VAE speaks [-1, 1]; every reward in this repo — and the depth
    # backend under it — is documented in [0, 1]. Skipping the remap leaves
    # images that still decode, still display, and score against a reference
    # built on the other convention.
    sampler.decoded = torch.tensor([[-1.0, -0.5, 0.0, 0.5, 1.0, 1.4, -1.4]])

    image = backend.decode(torch.zeros(1, 16, 2, 2))

    torch.testing.assert_close(
        image,
        torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 0.0]]),
        rtol=0.0,
        atol=0.0,
    )


def test_the_decode_hands_back_float32_whatever_the_transformer_runs_in(
    sampler: _StubSampler, backend: FluxFlowMapBackend
) -> None:
    sampler.decoded = torch.zeros(1, 3, 4, 4, dtype=torch.bfloat16)

    assert backend.decode(torch.zeros(1, 16, 2, 2)).dtype == torch.float32


def test_the_encode_shifts_before_it_scales(
    sampler: _StubSampler, backend: FluxFlowMapBackend
) -> None:
    image = torch.rand(1, 3, 8, 8)
    generator = torch.Generator().manual_seed(0)

    latent = backend.encode(image, generator=generator)

    # Both conversions in one line, and the shift is the one that goes missing:
    # `latent * scaling_factor` is Stable Diffusion's normalisation and is what
    # FLUX's looks like with the shift dropped — a constant 0.116 offset on
    # every latent, which the decoder happily renders and no test notices.
    expected = ((image * 2.0 - 1.0).to(sampler.dtype) - _SHIFT_FACTOR) * _SCALING_FACTOR
    torch.testing.assert_close(latent, expected, rtol=0.0, atol=0.0)
    # And the VAE was handed the photo in its own range, not ours.
    torch.testing.assert_close(
        sampler.pipeline.vae.seen[0], image * 2.0 - 1.0, rtol=0.0, atol=0.0
    )


def test_the_encode_hands_the_vae_the_dtype_the_transformer_runs_in(
    sampler: _StubSampler, backend: FluxFlowMapBackend
) -> None:
    sampler.dtype = torch.bfloat16

    latent = backend.encode(torch.rand(1, 3, 8, 8), generator=torch.Generator())

    assert sampler.pipeline.vae.seen[0].dtype == torch.bfloat16
    assert latent.dtype == torch.bfloat16


def test_the_encode_samples_the_posterior_from_the_walks_own_generator(
    sampler: _StubSampler, backend: FluxFlowMapBackend
) -> None:
    # The patch that exists for determinism: upstream samples the posterior off
    # the global RNG, so a seeded run would not reproduce its own first latent.
    generator = torch.Generator().manual_seed(0)

    backend.encode(torch.rand(1, 3, 8, 8), generator=generator)

    assert sampler.pipeline.vae.generators == [generator]


def test_the_velocity_asks_the_two_time_map_for_the_jump_it_was_given(
    sampler: _StubSampler, backend: FluxFlowMapBackend
) -> None:
    latent = torch.zeros(1, 16, 4, 4)

    backend.velocity(latent, sigma=0.6, target_sigma=0.0)

    (call,) = sampler.vector_calls
    assert call["t_cur"] == 0.6
    assert call["t_next"] == 0.0
    assert call["guidance_scale"] == DEFAULT_TEXT_GUIDANCE_SCALE
    assert call["prompt_embeds"] is sampler.cached_prompt_embeds
    assert call["pooled_prompt_embeds"] is sampler.cached_pooled_prompt_embeds


def test_the_text_guidance_scale_is_the_backends_and_not_the_rewards(
    sampler: _StubSampler,
) -> None:
    backend = FluxFlowMapBackend(sampler=sampler, text_guidance_scale=1.0)

    backend.velocity(torch.zeros(1, 16, 4, 4), sigma=0.6, target_sigma=0.0)

    assert sampler.vector_calls[0]["guidance_scale"] == 1.0


def test_selecting_a_cached_prompt_swaps_the_conditioning_every_call_reads(
    sampler: _StubSampler,
) -> None:
    """One transformer load serves many prompts: the cache entry becomes the conditioning."""
    embeds, pooled = torch.full((1, 2, 4), 3.0), torch.full((1, 4), 5.0)
    sampler.prompt_cache["a scandinavian living room"] = (embeds, pooled)
    backend = FluxFlowMapBackend(sampler=sampler)

    select_cached_prompt(sampler, "a scandinavian living room")
    backend.velocity(torch.zeros(1, 16, 4, 4), sigma=0.6, target_sigma=0.0)

    call = sampler.vector_calls[0]
    assert torch.equal(call["prompt_embeds"], embeds)  # type: ignore[arg-type]
    assert torch.equal(call["pooled_prompt_embeds"], pooled)  # type: ignore[arg-type]


def test_a_prompt_that_was_never_encoded_is_refused_by_name(
    sampler: _StubSampler,
) -> None:
    """The text encoders are gone by now, so a cache miss must fail loudly, not re-encode."""
    sampler.prompt_cache["the one that was encoded"] = (
        torch.zeros(1, 2, 4),
        torch.zeros(1, 4),
    )

    with pytest.raises(KeyError, match="never encoded") as raised:
        select_cached_prompt(sampler, "the one that was not")

    assert "the one that was encoded" in str(raised.value)


def test_an_image_context_entry_appends_scaled_tokens_after_the_text(
    sampler: _StubSampler,
) -> None:
    """Redux tokens ride behind the T5 embeds, and the scale touches them alone."""
    text = torch.full((1, 2, 4), 3.0)
    pooled = torch.full((1, 4), 5.0)
    sampler.prompt_cache["renovate this room"] = (text, pooled)
    tokens = torch.full((1, 3, 4), 2.0, dtype=torch.bfloat16)

    register_image_context_prompt(
        sampler,
        name="renovate this room + photo",
        prompt="renovate this room",
        image_tokens=tokens,
        scale=0.5,
    )

    combined, combined_pooled = sampler.prompt_cache["renovate this room + photo"]
    assert combined.shape == (1, 5, 4)
    assert combined.dtype == text.dtype
    # The text half is untouched — the scale is not diffusers'
    # `prompt_embeds_scale`, which would multiply this half too.
    torch.testing.assert_close(combined[:, :2], text, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        combined[:, 2:], torch.full((1, 3, 4), 1.0), rtol=0.0, atol=0.0
    )
    # Redux has no pooled embedding: the text CLIP pooled is passed through.
    assert combined_pooled is pooled


def test_an_image_context_entry_conditions_velocity_calls_like_any_prompt(
    sampler: _StubSampler,
) -> None:
    """The longer context is selected and served exactly as a text-only entry is."""
    sampler.prompt_cache["a room"] = (torch.ones(1, 2, 4), torch.ones(1, 4))
    register_image_context_prompt(
        sampler,
        name="a room + photo",
        prompt="a room",
        image_tokens=torch.ones(1, 3, 4),
        scale=1.0,
    )
    backend = FluxFlowMapBackend(sampler=sampler)

    select_cached_prompt(sampler, "a room + photo")
    backend.velocity(torch.zeros(1, 16, 4, 4), sigma=0.6, target_sigma=0.0)

    call = sampler.vector_calls[0]
    embeds = call["prompt_embeds"]
    assert isinstance(embeds, Tensor)
    assert embeds.shape == (1, 5, 4)


def test_an_image_context_entry_needs_its_text_half_already_encoded(
    sampler: _StubSampler,
) -> None:
    sampler.prompt_cache["the one that was encoded"] = (
        torch.zeros(1, 2, 4),
        torch.zeros(1, 4),
    )

    with pytest.raises(KeyError, match="never encoded"):
        register_image_context_prompt(
            sampler,
            name="anything",
            prompt="the one that was not",
            image_tokens=torch.zeros(1, 3, 4),
            scale=1.0,
        )


def test_a_negative_image_token_scale_is_refused(sampler: _StubSampler) -> None:
    sampler.prompt_cache["a room"] = (torch.zeros(1, 2, 4), torch.zeros(1, 4))

    with pytest.raises(ValueError, match="non-negative"):
        register_image_context_prompt(
            sampler,
            name="a room + photo",
            prompt="a room",
            image_tokens=torch.zeros(1, 3, 4),
            scale=-0.5,
        )
