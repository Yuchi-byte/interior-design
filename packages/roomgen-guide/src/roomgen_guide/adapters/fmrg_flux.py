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
import importlib
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from jaxtyping import Float
from torch import Tensor

# FLUX.1-dev's own embedded guidance scale, as both FMRG and the diffusers
# pipeline default it. Unrelated to the reward's `strength`.
DEFAULT_TEXT_GUIDANCE_SCALE = 3.5


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


class DeviceMovable(Protocol):
    def to(self, device: torch.device) -> object: ...


class TwoTimestepPipeline(Protocol):
    vae: FluxAutoencoder
    scheduler: SigmaScheduler
    transformer: DeviceMovable


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


def load_flow_map_sampler(
    *,
    fmrg_path: Path,
    lora_path: Path,
    prompt: str,
    device: torch.device,
    model_id: str = "black-forest-labs/FLUX.1-dev",
    dtype: torch.dtype = torch.bfloat16,
) -> TwoTimeFlowMapSampler:
    """Load FMRG's FLUX flow-map sampler with its LoRA, staged to fit a 32GB card.

    Args:
        fmrg_path: the vendored `packages/fmrg` checkout, which ships no
            installable package and is therefore reached through `sys.path`.
        lora_path: directory holding `pytorch_lora_weights.safetensors` for the
            flow-map LoRA trained at the resolution being sampled.
        prompt: precomputed here because the text encoders are freed immediately
            afterwards — the sampler cannot encode a second prompt later.

    Upstream's loader moves the transformer, T5, CLIP and the VAE onto the GPU
    together (~34GB in bf16) and only then frees the text encoders, which does
    not fit. The transformer is held back for the length of that call and lifted
    once the encoders are gone, which halves the transient; the vendored tree
    stays byte-identical to the commit its NOTICE.md records.
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
            )
    sampler.pipeline.transformer.to(device)
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
