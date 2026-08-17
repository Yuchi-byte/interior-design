"""FLUX.1-Kontext-dev as a `FlowMapBackend`: instruction editing under the reward driver.

Kontext is a plain rectified-flow velocity model whose editing behaviour is
conditioning, not initialisation: the reference photo is VAE-encoded, packed
into FLUX's 2x2 token grid and *sequence-concatenated* with the noisy latent
tokens under a RoPE layer offset, and every velocity call attends over both.
The one-step endpoint ``z - sigma * v(z, sigma)`` and the Euler step share the
single velocity the model predicts — exactly `FlowMapProcess`'s
``endpoint_reuse`` stepping — so Kontext plugs into the existing process with
no new walk: build one backend per (prompt, reference photo) and run
`FlowMapProcess` at ``start_sigma`` 1, because Kontext samples from pure noise
and the photo enters through attention, never through the init.

The load is staged as the FMRG one is: the text encoders exist only long
enough to fill a CPU-side prompt cache, and the ~24GB transformer is lifted
once they are gone. No gradient is ever taken through the transformer — the
reward's pullback runs through the endpoint arithmetic and the VAE decoder
alone — so it is loaded frozen and without gradient checkpointing.
"""

import gc
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import torch
from jaxtyping import Float
from torch import Tensor

from roomgen_guide.adapters.fmrg_flux import FluxAutoencoderConfig, SigmaScheduler

DEFAULT_KONTEXT_CHECKPOINT = "black-forest-labs/FLUX.1-Kontext-dev"

# Black Forest Labs' recommended embedded guidance for Kontext-dev editing.
# diffusers' pipeline signature defaults to 3.5 (inherited from FLUX.1-dev);
# every published Kontext example runs 2.5, and that is the default here.
DEFAULT_KONTEXT_GUIDANCE_SCALE = 2.5

# T5 context length the prompts are encoded at, the pipeline's own default.
_MAX_SEQUENCE_LENGTH = 512


class KontextTransformer(Protocol):
    """The velocity network, as the backend calls it (diffusers' FluxTransformer2DModel).

    ``hidden_states`` is the noisy tokens with the reference tokens appended on
    the sequence axis; the prediction covers the whole sequence and only its
    noisy head is the velocity.
    """

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
    ) -> tuple[Float[Tensor, "batch sequence channels"], ...]: ...


class KontextLatentDistribution(Protocol):
    def mode(self) -> Tensor: ...


class KontextEncoderOutput(Protocol):
    latent_dist: KontextLatentDistribution


class KontextAutoencoder(Protocol):
    """FLUX's VAE, both directions: the decode stays on the autograd graph."""

    config: FluxAutoencoderConfig

    def encode(
        self, x: Float[Tensor, "batch 3 height width"]
    ) -> KontextEncoderOutput: ...

    def decode(
        self, z: Float[Tensor, "batch channels height width"], return_dict: bool
    ) -> tuple[Float[Tensor, "batch 3 height width"], ...]: ...


@dataclass(frozen=True)
class KontextStack:
    """One transformer load serving every (prompt, reference) backend a run builds.

    Args:
        prompt_cache: T5 and pooled-CLIP embeddings per prompt, on the CPU —
            the text encoders are freed at load, so every prompt a run will
            ever condition on must be named to :func:`load_kontext_stack`.
    """

    transformer: KontextTransformer
    vae: KontextAutoencoder
    scheduler: SigmaScheduler
    device: torch.device
    dtype: torch.dtype
    prompt_cache: dict[
        str,
        tuple[
            Float[Tensor, "1 text embedding"],
            Float[Tensor, "1 embedding"],
        ],
    ]


@dataclass(frozen=True)
class KontextFluxBackend:
    """Kontext's sampler surface in roomgen's conventions, one instance per sample.

    Images cross in ``[0, 1]`` where the VAE speaks ``[-1, 1]``; latents cross
    *packed*, ``[batch, tokens, 64]``, because that is the shape the
    transformer and the RoPE ids are built over. `target_sigma` on the velocity
    call is accepted and ignored: a plain rectified-flow field is the diagonal
    of a two-time flow map, so the velocity toward data is the same object
    whatever the target — the Euler step and the endpoint jump share it, which
    is exactly `FlowMapProcess`'s ``endpoint_reuse`` arithmetic.

    Args:
        text_guidance_scale: Kontext's embedded guidance — conditioning, not
            reward steering.
    """

    transformer: KontextTransformer
    vae: KontextAutoencoder
    scheduler: SigmaScheduler
    prompt_embeds: Float[Tensor, "1 text embedding"]
    pooled_prompt_embeds: Float[Tensor, "1 embedding"]
    txt_ids: Float[Tensor, "text 3"]
    reference_tokens: Float[Tensor, "1 tokens channels"]
    image_ids: Float[Tensor, "sequence 3"]
    latent_height: int
    latent_width: int
    text_guidance_scale: float = DEFAULT_KONTEXT_GUIDANCE_SCALE

    def schedule(
        self,
        *,
        steps: int,
        start_sigma: float,
        latent: Float[Tensor, "batch tokens channels"],
    ) -> Sequence[float]:
        """FLUX's shifted sigmas from `start_sigma` down to 0.

        At ``start_sigma`` 1 — Kontext's own operating point — this is exactly
        the pipeline's default grid: ``linspace(1, 1/steps, steps)`` warped by
        the sequence-length shift, which reads the *noisy* tokens alone, as the
        pipeline's does — the reference tokens never enter the schedule.
        """
        from diffusers.pipelines.flux.pipeline_flux import (  # noqa: PLC0415
            calculate_shift,
            retrieve_timesteps,
        )

        shift = calculate_shift(
            latent.shape[1],
            int(self.scheduler.config.get("base_image_seq_len", 256)),
            int(self.scheduler.config.get("max_image_seq_len", 4096)),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        sigmas: list[float] = torch.linspace(
            start_sigma, start_sigma / steps, steps, dtype=torch.float64
        ).tolist()
        timesteps, _ = retrieve_timesteps(
            self.scheduler, steps, latent.device, sigmas=sigmas, mu=shift
        )
        return [*(float(timestep) / 1000.0 for timestep in timesteps), 0.0]

    def encode(
        self,
        image: Float[Tensor, "batch 3 height width"],
        *,
        generator: torch.Generator,
    ) -> Float[Tensor, "batch tokens channels"]:
        # The mode, not a posterior draw: Kontext's own reference path encodes
        # at the mode (diffusers' ``sample_mode="argmax"``), and the walk at
        # `start_sigma` 1 discards this latent anyway — only its shape and the
        # schedule read it. The generator is the protocol's, unused here.
        del generator
        return _pack(self._scaled_latent(image))

    def decode(
        self, latent: Float[Tensor, "batch tokens channels"]
    ) -> Float[Tensor, "batch 3 height width"]:
        unpacked = _unpack(latent, self.latent_height, self.latent_width)
        z = unpacked / self.vae.config.scaling_factor + self.vae.config.shift_factor
        image = self.vae.decode(z, return_dict=False)[0]
        return ((image.float() + 1.0) / 2.0).clamp(0.0, 1.0)

    def velocity(
        self,
        latent: Float[Tensor, "batch tokens channels"],
        *,
        sigma: float,
        target_sigma: float,
    ) -> Float[Tensor, "batch tokens channels"]:
        del target_sigma
        batch = latent.shape[0]
        hidden = torch.cat([latent, self.reference_tokens.expand(batch, -1, -1)], dim=1)
        prediction = self.transformer(
            hidden_states=hidden,
            # The pipeline hands the transformer `t / 1000` in the latent's
            # dtype, and its `t` is the shifted sigma on the training scale —
            # so what the network sees is the sigma itself.
            timestep=torch.full(
                (batch,), sigma, device=latent.device, dtype=latent.dtype
            ),
            guidance=torch.full(
                (batch,),
                self.text_guidance_scale,
                device=latent.device,
                dtype=torch.float32,
            ),
            pooled_projections=self.pooled_prompt_embeds.expand(batch, -1),
            encoder_hidden_states=self.prompt_embeds.expand(batch, -1, -1),
            txt_ids=self.txt_ids,
            img_ids=self.image_ids,
            return_dict=False,
        )[0]
        # The tail of the prediction covers the reference tokens, which never
        # move: only the noisy head is the velocity.
        return prediction[:, : latent.shape[1]]

    def _scaled_latent(
        self, image: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, "batch channels latent_height latent_width"]:
        return _scaled_latent(self.vae, image, dtype=self.reference_tokens.dtype)


def build_kontext_backend(
    stack: KontextStack,
    *,
    prompt: str,
    reference_image: Float[Tensor, "1 3 height width"],
    text_guidance_scale: float = DEFAULT_KONTEXT_GUIDANCE_SCALE,
) -> KontextFluxBackend:
    """One sample's backend: `prompt` from the stack's cache, the photo as context tokens.

    The reference is encoded at the VAE posterior's mode, packed, and given the
    RoPE ids of a second image layer (``ids[..., 0] = 1``) over the same
    spatial grid — diffusers' own composition, verified against
    `FluxKontextPipeline.prepare_latents`. It is a constant of the walk and is
    encoded under `no_grad`: the gradient path is the *noisy* tokens'.

    Args:
        reference_image: NCHW in ``[0, 1]``, sides multiples of 16 (the VAE
            downsamples by 8 and the packing takes 2x2 blocks of the grid).
    """
    if prompt not in stack.prompt_cache:
        raise KeyError(
            f"prompt {prompt!r} was never encoded: the stack's text encoders "
            f"are freed at load, so every prompt must be named to "
            f"`load_kontext_stack`. Cached: {sorted(stack.prompt_cache)}"
        )
    height, width = reference_image.shape[-2:]
    if height % 16 or width % 16:
        raise ValueError(
            f"reference_image sides must be multiples of 16 to survive the VAE "
            f"and the 2x2 packing unrounded, got {height}x{width}"
        )
    embeds, pooled = stack.prompt_cache[prompt]
    embeds = embeds.to(device=stack.device, dtype=stack.dtype)
    pooled = pooled.to(device=stack.device, dtype=stack.dtype)
    with torch.no_grad():
        reference_latent = _scaled_latent(
            stack.vae, reference_image.to(stack.device), dtype=stack.dtype
        )
    latent_height, latent_width = reference_latent.shape[-2:]
    grid = (latent_height // 2, latent_width // 2)
    ids = torch.cat(
        [
            _token_ids(*grid, layer=0.0, device=stack.device, dtype=stack.dtype),
            _token_ids(*grid, layer=1.0, device=stack.device, dtype=stack.dtype),
        ],
        dim=0,
    )
    return KontextFluxBackend(
        transformer=stack.transformer,
        vae=stack.vae,
        scheduler=stack.scheduler,
        prompt_embeds=embeds,
        pooled_prompt_embeds=pooled,
        txt_ids=torch.zeros(embeds.shape[1], 3, device=stack.device, dtype=stack.dtype),
        reference_tokens=_pack(reference_latent),
        image_ids=ids,
        latent_height=latent_height,
        latent_width=latent_width,
        text_guidance_scale=text_guidance_scale,
    )


def load_kontext_stack(
    *,
    checkpoint: str | Path = DEFAULT_KONTEXT_CHECKPOINT,
    prompts: Sequence[str],
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    cache_dir: Path | None = None,
) -> KontextStack:
    """Load FLUX.1-Kontext-dev staged to fit a 32GB card, prompts encoded up front.

    The pipeline is first loaded *without* its transformer — T5, CLIP, the VAE
    and the scheduler together are ~10GB in bf16 — every prompt is encoded into
    a CPU-side cache, the text encoders are freed, and only then is the ~24GB
    transformer lifted onto the device. It comes back frozen and without
    gradient checkpointing: the only gradient anywhere is with respect to the
    state, and it never traverses the transformer (the cheap pullback runs
    through the endpoint arithmetic and the VAE decoder alone).

    Args:
        checkpoint: Hub id or local snapshot path; the checkpoint is gated like
            FLUX.1-dev and resolves with the same token.
        prompts: every prompt any backend built over this stack will ever
            condition on — a second prompt cannot be encoded later.
    """
    from diffusers.pipelines.flux.pipeline_flux_kontext import (  # noqa: PLC0415
        FluxKontextPipeline,
    )

    pipeline = FluxKontextPipeline.from_pretrained(
        str(checkpoint),
        transformer=None,
        torch_dtype=dtype,
        cache_dir=None if cache_dir is None else str(cache_dir),
    )
    pipeline.to(device)
    prompt_cache: dict[str, tuple[Tensor, Tensor]] = {}
    with torch.no_grad():
        for prompt in dict.fromkeys(prompts):
            embeds, pooled, _text_ids = pipeline.encode_prompt(
                prompt=prompt,
                prompt_2=None,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=_MAX_SEQUENCE_LENGTH,
            )
            prompt_cache[prompt] = (embeds.to("cpu"), pooled.to("cpu"))
    vae = cast("KontextAutoencoder", pipeline.vae)
    scheduler = cast("SigmaScheduler", pipeline.scheduler)
    # The pipeline object holds the encoders through its component registry;
    # nulling the attributes before dropping it is what actually frees them.
    pipeline.text_encoder = None
    pipeline.text_encoder_2 = None
    del pipeline
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    from diffusers.models.transformers.transformer_flux import (  # noqa: PLC0415
        FluxTransformer2DModel,
    )

    transformer = FluxTransformer2DModel.from_pretrained(
        str(checkpoint),
        subfolder="transformer",
        torch_dtype=dtype,
        cache_dir=None if cache_dir is None else str(cache_dir),
    )
    transformer.to(device)
    transformer.requires_grad_(False)
    return KontextStack(
        transformer=cast("KontextTransformer", transformer),
        vae=vae,
        scheduler=scheduler,
        device=device,
        dtype=dtype,
        prompt_cache=prompt_cache,
    )


def _pack(
    latent: Float[Tensor, "batch channels height width"],
) -> Float[Tensor, "batch tokens packed"]:
    """FLUX's 2x2 token packing, differentiably (a view/permute/reshape chain)."""
    batch, channels, height, width = latent.shape
    packed = latent.view(batch, channels, height // 2, 2, width // 2, 2)
    packed = packed.permute(0, 2, 4, 1, 3, 5)
    return packed.reshape(batch, (height // 2) * (width // 2), channels * 4)


def _unpack(
    latent: Float[Tensor, "batch tokens packed"],
    height: int,
    width: int,
) -> Float[Tensor, "batch channels height width"]:
    batch, _, packed_channels = latent.shape
    channels = packed_channels // 4
    unpacked = latent.view(batch, height // 2, width // 2, channels, 2, 2)
    unpacked = unpacked.permute(0, 3, 1, 4, 2, 5)
    return unpacked.reshape(batch, channels, height, width)


def _token_ids(
    height: int,
    width: int,
    *,
    layer: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Float[Tensor, "tokens 3"]:
    """RoPE ids over a token grid: (layer, row, column) per token, row-major.

    Layer 0 is the noisy image, layer 1 the reference — the index offset that
    lets the model tell the two same-shaped grids apart.
    """
    ids = torch.zeros(height, width, 3)
    ids[..., 0] = layer
    ids[..., 1] += torch.arange(height)[:, None]
    ids[..., 2] += torch.arange(width)[None, :]
    return ids.reshape(height * width, 3).to(device=device, dtype=dtype)


def _scaled_latent(
    vae: KontextAutoencoder,
    image: Float[Tensor, "batch 3 height width"],
    *,
    dtype: torch.dtype,
) -> Float[Tensor, "batch channels latent_height latent_width"]:
    posterior = vae.encode(image.to(dtype) * 2.0 - 1.0).latent_dist
    return (posterior.mode() - vae.config.shift_factor) * vae.config.scaling_factor
