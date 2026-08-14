"""FMRG's FLUX flow-map sampler behind roomgen's `FlowMapBackend`.

The vendored tree (`packages/fmrg`) is third-party source that is never edited,
never imported by anything under test, and pinned to a `diffusers` release the
rest of the repo does not hold — so every import of it here is deferred to the
call that needs it, and the sampler itself is named only by the Protocol below.
What crosses the boundary is four calls and the two patches upstream cannot make
for itself: the staged load that keeps a 32GB card alive, and a VAE encode that
takes the walk's generator instead of the global RNG.
"""

import contextlib
import gc
import importlib
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import torch
from jaxtyping import Float
from torch import Tensor

# FLUX.1-dev's own embedded guidance scale, as both FMRG and the diffusers
# pipeline default it. Unrelated to the reward's `strength`.
DEFAULT_TEXT_GUIDANCE_SCALE = 3.5

# FLUX.1-Redux-dev: the SigLIP image encoder plus the Redux adapter that
# projects its patch grid into T5 embedding space, gated behind the same HF
# licence as FLUX.1-dev itself.
DEFAULT_REDUX_CHECKPOINT = "black-forest-labs/FLUX.1-Redux-dev"


class SampleOutput(Protocol):
    sample: Tensor


class LatentDistribution(Protocol):
    def sample(self, generator: torch.Generator | None = None) -> Tensor: ...


class EncoderOutput(Protocol):
    latent_dist: LatentDistribution


class FluxAutoencoderConfig(Protocol):
    """FLUX's VAE is scaled *and* shifted, unlike Stable Diffusion's."""

    scaling_factor: float
    shift_factor: float


class FluxAutoencoder(Protocol):
    config: FluxAutoencoderConfig

    def encode(self, x: Float[Tensor, "batch 3 height width"]) -> EncoderOutput: ...


class SigmaScheduler(Protocol):
    """Read only for the resolution-dependent shift its config parameterises."""

    config: Mapping[str, float]


class Transformer(Protocol):
    """The flow map's own network, as the loader has to configure it."""

    def to(self, device: torch.device) -> object: ...

    def requires_grad_(self, requires_grad: bool) -> object: ...

    def enable_gradient_checkpointing(self) -> None: ...


class TwoTimestepPipeline(Protocol):
    vae: FluxAutoencoder
    scheduler: SigmaScheduler
    transformer: Transformer


class TwoTimeFlowMapSampler(Protocol):
    """The surface of FMRG's `FluxFlowMapSampler` this adapter drives.

    Duck-typed on purpose, and matched against the vendored implementation only
    at runtime: naming the class here would put a `diffusers`-0.34-pinned import
    inside a package that has to keep importing without it.
    """

    dtype: torch.dtype
    pipeline: TwoTimestepPipeline
    cached_prompt_embeds: Float[Tensor, "batch sequence embedding"]
    cached_pooled_prompt_embeds: Float[Tensor, "batch embedding"]
    prompt_cache: dict[
        str,
        tuple[
            Float[Tensor, "batch sequence embedding"], Float[Tensor, "batch embedding"]
        ],
    ]

    def decode(
        self, z: Float[Tensor, "batch channels height width"]
    ) -> Float[Tensor, "batch 3 height width"]:
        """Latent to image in ``[-1, 1]``, differentiable by contract."""
        ...

    def predict_vector(
        self,
        z: Float[Tensor, "batch channels height width"],
        t_cur: float,
        prompt_embeds: Float[Tensor, "batch sequence embedding"] | None = None,
        pooled_prompt_embeds: Float[Tensor, "batch embedding"] | None = None,
        guidance_scale: float = DEFAULT_TEXT_GUIDANCE_SCALE,
        t_next: float | None = None,
    ) -> Float[Tensor, "batch channels height width"]:
        """The two-time flow map's velocity from `t_cur` towards `t_next`."""
        ...


@dataclass(frozen=True)
class FluxFlowMapBackend:
    """FMRG's sampler as a `FlowMapBackend`, in roomgen's conventions.

    Two conversions and nothing else: images cross in ``[0, 1]`` where FLUX's
    VAE speaks ``[-1, 1]``, and the endpoint is handed back in float32 whatever
    the transformer runs in, because the depth backend the reward is built on is
    a float32 network and a bf16 energy would quantise the gradient that steers
    the walk.

    Args:
        text_guidance_scale: FLUX's embedded guidance, which is conditioning and
            not steering — the reward's pull is the tilt policy's `strength`.
    """

    sampler: TwoTimeFlowMapSampler
    text_guidance_scale: float = DEFAULT_TEXT_GUIDANCE_SCALE

    def schedule(
        self,
        *,
        steps: int,
        start_sigma: float,
        latent: Float[Tensor, "batch channels height width"],
    ) -> Sequence[float]:
        """FLUX's shifted sigmas from `start_sigma` down to 0.

        The shift is resolution-dependent — a 512² walk and a 1024² walk visit
        different noise levels for the same step index — so the schedule is
        built through the pipeline's own scheduler rather than reimplemented,
        and `start_sigma` scales the pre-shift grid exactly as FMRG's own
        partial-trajectory decode does.
        """
        from diffusers.pipelines.flux.pipeline_flux import (  # noqa: PLC0415
            calculate_shift,
            retrieve_timesteps,
        )

        scheduler = self.sampler.pipeline.scheduler
        # Packed FLUX tokens: each covers a 2x2 block of the latent grid.
        image_sequence_length = (latent.shape[-2] // 2) * (latent.shape[-1] // 2)
        shift = calculate_shift(
            image_sequence_length,
            int(scheduler.config.get("base_image_seq_len", 256)),
            int(scheduler.config.get("max_image_seq_len", 4096)),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.15),
        )
        # The same pre-shift grid FMRG samples on, scaled by `start_sigma`: the
        # walk keeps its step count and gives up only the noisiest part of the
        # trajectory, which is what makes an img2img arm comparable to a
        # text-to-image one at equal NFE.
        sigmas: list[float] = torch.linspace(
            start_sigma, start_sigma / steps, steps, dtype=torch.float64
        ).tolist()
        timesteps, _ = retrieve_timesteps(
            scheduler, steps, latent.device, sigmas=sigmas, mu=shift
        )
        # The scheduler reports the shifted sigmas as timesteps on its training
        # scale; the walk is in sigma units, and it ends at data.
        return [*(float(timestep) / 1000.0 for timestep in timesteps), 0.0]

    def encode(
        self,
        image: Float[Tensor, "batch 3 height width"],
        *,
        generator: torch.Generator,
    ) -> Float[Tensor, "batch channels height width"]:
        vae = self.sampler.pipeline.vae
        posterior = vae.encode(image.to(self.sampler.dtype) * 2.0 - 1.0).latent_dist
        # The posterior is sampled, not taken at its mode, because that is what
        # the pipeline this checkpoint was trained under does — but from the
        # walk's own generator, so the encode is part of what a seed reproduces.
        latent = posterior.sample(generator=generator)
        return (latent - vae.config.shift_factor) * vae.config.scaling_factor

    def decode(
        self, latent: Float[Tensor, "batch channels height width"]
    ) -> Float[Tensor, "batch 3 height width"]:
        return ((self.sampler.decode(latent).float() + 1.0) / 2.0).clamp(0.0, 1.0)

    def velocity(
        self,
        latent: Float[Tensor, "batch channels height width"],
        *,
        sigma: float,
        target_sigma: float,
    ) -> Float[Tensor, "batch channels height width"]:
        return self.sampler.predict_vector(
            latent,
            sigma,
            prompt_embeds=self.sampler.cached_prompt_embeds,
            pooled_prompt_embeds=self.sampler.cached_pooled_prompt_embeds,
            guidance_scale=self.text_guidance_scale,
            t_next=target_sigma,
        )


def select_cached_prompt(sampler: TwoTimeFlowMapSampler, prompt: str) -> None:
    """Make `prompt` the conditioning of every velocity call the sampler serves next.

    The text encoders are freed the moment the sampler finishes loading, so a
    prompt is only reachable here if it was named to `load_flow_map_sampler` up
    front; the cache holds its embeddings on the CPU and this call is what lifts
    the chosen pair back beside the transformer.
    """
    if prompt not in sampler.prompt_cache:
        raise KeyError(
            f"prompt {prompt!r} was never encoded: the sampler's text encoders "
            f"are freed at load, so every prompt must be named to "
            f"`load_flow_map_sampler` (prompt= or additional_prompts=). Cached: "
            f"{sorted(sampler.prompt_cache)}"
        )
    embeds, pooled = sampler.prompt_cache[prompt]
    device = sampler.cached_prompt_embeds.device
    sampler.cached_prompt_embeds = embeds.to(device)
    sampler.cached_pooled_prompt_embeds = pooled.to(device)


def encode_redux_image_tokens(
    image: Float[Tensor, "1 3 height width"],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    cache_dir: Path | None = None,
    checkpoint: str = DEFAULT_REDUX_CHECKPOINT,
) -> Float[Tensor, "1 tokens embedding"]:
    """The photo as FLUX context tokens, via FLUX.1-Redux-dev's SigLIP + adapter.

    The tokens live in the same 4096-wide embedding space as the T5 prompt
    embeds and are meant to be *appended* to them — image-as-context, exactly
    analogous to a prompt — which is what diffusers' own
    `FluxPriorReduxPipeline` does (`torch.cat([prompt_embeds, image_embeds],
    dim=1)`). The encoders are loaded for this call and freed before it
    returns: the tokens are a constant of the reference photo, computed once,
    and nothing else here ever needs SigLIP resident.

    Off the gradient path by construction — the reference image is a constant,
    so the PIL-free but non-differentiable uint8 round trip into SigLIP's
    processor is fine here and would not be inside a reward.

    Args:
        image: NCHW in ``[0, 1]``, the same convention every reward speaks.
        cache_dir: the Hub snapshot cache (`RoomgenSettings.hub_cache`); the
            checkpoint is gated like FLUX.1-dev and resolves with the same token.

    Returns:
        The Redux tokens on the CPU in `dtype`, ready for
        :func:`register_image_context_prompt`.
    """
    from diffusers.pipelines.flux.pipeline_flux_prior_redux import (  # noqa: PLC0415
        FluxPriorReduxPipeline,
    )

    pipeline = FluxPriorReduxPipeline.from_pretrained(
        checkpoint,
        torch_dtype=dtype,
        cache_dir=None if cache_dir is None else str(cache_dir),
    ).to(device)
    # The pipeline's methods carry no annotations upstream; the shapes here are
    # SigLIP's patch grid and the adapter's projection of it into T5 width.
    encode_image = cast(
        "Callable[[object, torch.device, int], Tensor]", pipeline.encode_image
    )
    with torch.no_grad():
        # SigLIP's processor takes HWC uint8; the quantisation is invisible at
        # its 384-square resample and keeps the preprocessing on its own path.
        eight_bit = (
            (image[0].detach().float().clamp(0.0, 1.0) * 255.0)
            .round()
            .to(torch.uint8)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        hidden = encode_image(eight_bit, device, 1)
        tokens = pipeline.image_embedder(hidden).image_embeds
    result: Float[Tensor, "1 tokens embedding"] = tokens.detach().to(
        device="cpu", dtype=dtype
    )
    del pipeline, hidden, tokens
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def register_image_context_prompt(
    sampler: TwoTimeFlowMapSampler,
    *,
    name: str,
    prompt: str,
    image_tokens: Float[Tensor, "1 tokens embedding"],
    scale: float,
) -> None:
    """Cache `name` as `prompt`'s T5 embeds with `image_tokens` appended, scaled.

    The combined context conditions velocity calls exactly as a text-only entry
    does — `select_cached_prompt(sampler, name)` lifts it, and the vendored
    sampler sizes its `text_ids` off `prompt_embeds.shape[1]` at call time, so
    the longer context flows through with no vendored change.

    Two deliberate conventions, both verified against diffusers'
    `FluxPriorReduxPipeline`:

    - the pooled embedding stays the *text* CLIP pooled — Redux produces no
      pooled embedding, and diffusers keeps the text one (or zeros) untouched.
    - `scale` multiplies the image tokens **alone**. Redux tokens at full
      strength are known to drown a 512-token prompt under 729 tokens of image;
      diffusers' own `prompt_embeds_scale` is no help, since it scales the
      whole concatenated row (text included) and exists to weight several
      images against each other, not the image against the text.

    Args:
        name: the cache key the walk will select; any string not already
            naming a text prompt the run still needs.
        prompt: the already-encoded text prompt the tokens ride behind.
        scale: multiplier on the image tokens, non-negative; 1.0 is diffusers'
            composition, ~0.3 is the community operating point for keeping the
            text audible.
    """
    if scale < 0.0:
        raise ValueError(f"scale must be non-negative, got {scale}")
    if prompt not in sampler.prompt_cache:
        raise KeyError(
            f"prompt {prompt!r} was never encoded: the sampler's text encoders "
            f"are freed at load, so the text half of an image-context entry "
            f"must be named to `load_flow_map_sampler` (prompt= or "
            f"additional_prompts=). Cached: {sorted(sampler.prompt_cache)}"
        )
    embeds, pooled = sampler.prompt_cache[prompt]
    tokens = image_tokens.to(device=embeds.device, dtype=embeds.dtype) * scale
    sampler.prompt_cache[name] = (torch.cat([embeds, tokens], dim=1), pooled)


def load_flow_map_sampler(
    *,
    fmrg_path: Path,
    lora_path: Path,
    prompt: str,
    device: torch.device,
    model_id: str = "black-forest-labs/FLUX.1-dev",
    dtype: torch.dtype = torch.bfloat16,
    additional_prompts: Sequence[str] = (),
) -> TwoTimeFlowMapSampler:
    """Load FMRG's FLUX flow-map sampler with its LoRA, staged to fit a 32GB card.

    Args:
        fmrg_path: the vendored `packages/fmrg` checkout, which ships no
            installable package and is therefore reached through `sys.path`.
        lora_path: directory holding `pytorch_lora_weights.safetensors` for the
            flow-map LoRA trained at the resolution being sampled.
        prompt: precomputed here because the text encoders are freed immediately
            afterwards — the sampler cannot encode a second prompt later.
        additional_prompts: every other prompt this sampler will ever be asked
            to condition on, encoded into its CPU-side cache alongside `prompt`
            and switched between with `select_cached_prompt` — which is how one
            ~24GB transformer load serves a whole multi-prompt benchmark.

    Upstream's loader moves the transformer, T5, CLIP and the VAE onto the GPU
    together (~34GB in bf16) and only then frees the text encoders, which does
    not fit. The transformer is held back for the length of that call and lifted
    once the encoders are gone, which halves the transient; the vendored tree
    stays byte-identical to the commit its NOTICE.md records.

    The transformer comes back frozen and checkpointed, which is what the
    Jacobian path needs to fit at all: no weight gradient is ever used — the only
    gradient taken anywhere is with respect to the state — so letting a backward
    populate `.grad` on 12B parameters spends VRAM on nothing, and a 512² FLUX
    backward without activation checkpointing does not fit on a 32GB card.
    Neither changes a number: both are memory, not arithmetic.
    """
    with _fmrg_on_path(fmrg_path):
        try:
            two_timestep = importlib.import_module("flux_two_timestep")
            reward_module = importlib.import_module("fluxfm_sampler_reward")
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                f"{error.name!r} is not importable with fmrg_path {fmrg_path} on "
                f"sys.path. Point fmrg_path at the vendored FMRG checkout — the "
                f"directory holding flux_two_timestep/ and "
                f"fluxfm_sampler_reward.py, `packages/fmrg` in this repo — and "
                f"if the missing module is one of its own dependencies, install "
                f"the diffusers release its NOTICE.md pins."
            ) from error
        with _transformer_held_on_cpu(two_timestep.FluxPipelineTwoTimestep):
            sampler: TwoTimeFlowMapSampler = reward_module.FluxFlowMapSampler(
                model_id=model_id,
                lora_path=str(lora_path),
                device=device,
                dtype=dtype,
                prompt=prompt,
                additional_prompts=(
                    list(additional_prompts) if additional_prompts else None
                ),
            )
    transformer = sampler.pipeline.transformer
    transformer.to(device)
    transformer.requires_grad_(False)
    transformer.enable_gradient_checkpointing()
    return sampler


@contextlib.contextmanager
def _fmrg_on_path(fmrg_path: Path) -> Iterator[None]:
    """`sys.path` carrying the vendored checkout, rolled back if the load fails.

    The entry outlives a *successful* load deliberately: FMRG's modules import
    their own siblings lazily, inside the calls the sampler makes later, so
    removing it here would break the object this function returns. What must not
    survive is a failure — a mistyped `fmrg_path` that stayed on `sys.path`
    would shadow imports for the rest of the process, and the next call with the
    right path would find `flux_two_timestep` already half-imported from the
    wrong one.
    """
    entry = str(fmrg_path)
    if entry in sys.path:
        yield
        return
    sys.path.insert(0, entry)
    try:
        yield
    except BaseException:
        # `remove` and not `pop(0)`: another thread may have inserted since.
        with contextlib.suppress(ValueError):
            sys.path.remove(entry)
        raise


@contextlib.contextmanager
def _transformer_held_on_cpu(pipeline_class: type[object]) -> Iterator[None]:
    original_to = pipeline_class.to  # type: ignore[attr-defined]

    def staged_to(self: object, *args: object, **kwargs: object) -> object:
        device = args[0] if args else kwargs["device"]
        for name in ("vae", "text_encoder", "text_encoder_2"):
            module = getattr(self, name, None)
            if module is not None:
                module.to(device)
        return self

    setattr(pipeline_class, "to", staged_to)  # noqa: B010
    try:
        yield
    finally:
        setattr(pipeline_class, "to", original_to)  # noqa: B010
