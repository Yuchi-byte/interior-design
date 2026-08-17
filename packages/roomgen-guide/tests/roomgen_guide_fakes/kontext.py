"""Weightless doubles for the Kontext backend: FLUX's shapes with no FLUX in the room.

The fake transformer is a velocity field over *packed* tokens whose model of
data is a blur pulled toward the reference tokens — enough for the real
`KontextFluxBackend` to walk, for the reference conditioning to be observable,
and for a detached graph to show up as the zero gradient it is. The fake VAE is
a pooling encode and a differentiable upsampling decode in FLUX's scaled and
shifted latent convention. The scheduler is diffusers' real
`FlowMatchEulerDiscreteScheduler` in FLUX's dynamic-shifting configuration:
it is weightless arithmetic, and the resolution-dependent shift is exactly
what the schedule test is about.
"""

from dataclasses import dataclass, field
from typing import Final, cast

import torch
from jaxtyping import Float
from torch import Tensor

from roomgen_guide.adapters.fmrg_flux import SigmaScheduler
from roomgen_guide.adapters.kontext_flux import (
    KontextStack,
    KontextTransformer,
)

# FLUX's latent geometry, scaled down: 16 latent channels packed 2x2 into
# 64-wide tokens, over a VAE that downsamples by 8.
KONTEXT_LATENT_CHANNELS: Final = 16
_VAE_FACTOR: Final = 8
_MINIMUM_SIGMA: Final = 1e-3

# The prompt every fake stack encodes; the fake transformer never reads it.
FAKE_KONTEXT_PROMPT: Final = "a renovated room"


@dataclass(frozen=True)
class _FakeVaeConfig:
    # Deliberately not 1 and 0, so a backend that skips either convention
    # produces a visibly wrong roundtrip rather than a lucky identity.
    scaling_factor: float = 0.5
    shift_factor: float = 0.1


@dataclass(frozen=True)
class _FakeLatentDistribution:
    latent: Float[Tensor, "batch channels height width"]

    def mode(self) -> Tensor:
        return self.latent


@dataclass(frozen=True)
class _FakeEncoderOutput:
    latent_dist: _FakeLatentDistribution


class FakeKontextVae:
    """Pooling in, differentiable upsampling out, in the VAE's ``[-1, 1]`` speech."""

    config = _FakeVaeConfig()

    def encode(self, x: Float[Tensor, "batch 3 height width"]) -> _FakeEncoderOutput:
        pooled = torch.nn.functional.avg_pool2d(x, _VAE_FACTOR)
        latent = torch.cat([pooled] * 6, dim=1)[:, :KONTEXT_LATENT_CHANNELS]
        return _FakeEncoderOutput(_FakeLatentDistribution(latent))

    def decode(
        self,
        z: Float[Tensor, "batch channels height width"],
        return_dict: bool = True,
    ) -> tuple[Float[Tensor, "batch 3 height width"], ...]:
        del return_dict
        upsampled = torch.nn.functional.interpolate(
            z[:, :3], scale_factor=_VAE_FACTOR, mode="nearest"
        )
        # tanh rather than a clamp, as every fake decoder here: no pixel
        # saturates, so a zero gradient is a real finding rather than a rail.
        return (torch.tanh(upsampled),)


@dataclass
class FakeKontextTransformer:
    """A packed-token velocity field whose model of data leans on the reference.

    Records every call's full sequence length, so a test can assert the
    reference tokens actually rode along; the endpoint the velocity points at
    is a blur of the noisy tokens pulled toward the reference tokens, which
    makes the conditioning observable in the output.
    """

    sequence_lengths: list[int] = field(default_factory=list)

    def __call__(
        self,
        *,
        hidden_states: Float[Tensor, "batch sequence channels"],
        timestep: Float[Tensor, "batch"],
        guidance: Float[Tensor, "batch"] | None,
        pooled_projections: Float[Tensor, "batch embedding"],
        encoder_hidden_states: Float[Tensor, "batch text embedding"],
        txt_ids: Float[Tensor, "text 3"],
        img_ids: Float[Tensor, "sequence 3"],
        return_dict: bool,
    ) -> tuple[Float[Tensor, "batch sequence channels"], ...]:
        del guidance, pooled_projections, encoder_hidden_states, txt_ids
        del return_dict
        self.sequence_lengths.append(int(hidden_states.shape[1]))
        noisy_tokens = int((img_ids[:, 0] == 0.0).sum())
        grid_height = int(img_ids[:noisy_tokens, 1].max()) + 1
        grid_width = int(img_ids[:noisy_tokens, 2].max()) + 1
        noisy = hidden_states[:, :noisy_tokens]
        reference = hidden_states[:, noisy_tokens:]
        spatial = _tokens_to_grid(noisy, grid_height, grid_width)
        smoothed = torch.nn.functional.avg_pool2d(spatial, 3, stride=1, padding=1)
        endpoint = 0.6 * _grid_to_tokens(smoothed) + 0.4 * reference
        sigma = max(float(timestep[0]), _MINIMUM_SIGMA)
        velocity = (noisy - endpoint) / sigma
        return (torch.cat([velocity, torch.zeros_like(reference)], dim=1),)


def fake_kontext_stack(
    *,
    prompt: str = FAKE_KONTEXT_PROMPT,
    dtype: torch.dtype = torch.float32,
) -> KontextStack:
    """A stack the real `build_kontext_backend` accepts, with nothing loaded."""
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import (  # noqa: PLC0415
        FlowMatchEulerDiscreteScheduler,
    )

    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        use_dynamic_shifting=True,
        base_shift=0.5,
        max_shift=1.15,
        base_image_seq_len=256,
        max_image_seq_len=4096,
    )
    return KontextStack(
        transformer=cast("KontextTransformer", FakeKontextTransformer()),
        vae=FakeKontextVae(),
        scheduler=cast("SigmaScheduler", scheduler),
        device=torch.device("cpu"),
        dtype=dtype,
        prompt_cache={prompt: (torch.zeros(1, 4, 8), torch.zeros(1, 8))},
    )


def _tokens_to_grid(
    tokens: Float[Tensor, "batch tokens packed"], height: int, width: int
) -> Float[Tensor, "batch channels grid_height grid_width"]:
    batch, _, packed = tokens.shape
    return tokens.permute(0, 2, 1).reshape(batch, packed, height, width)


def _grid_to_tokens(
    grid: Float[Tensor, "batch channels grid_height grid_width"],
) -> Float[Tensor, "batch tokens packed"]:
    batch, packed, height, width = grid.shape
    return grid.reshape(batch, packed, height * width).permute(0, 2, 1)
