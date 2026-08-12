"""Stable Diffusion img2img as an endpoint process, and the clipped step that steers it."""

from dataclasses import dataclass
from typing import Final

import torch
from jaxtyping import Float, Int
from torch import Tensor

from roomgen_guide.guidance.base import Guidance
from roomgen_guide.process.driver import sample_endpoint_process
from roomgen_guide.process.endpoint import Endpoint, StepContext
from roomgen_guide.process.tilt import EnergyGradient
from roomgen_guide.sampling.pipeline import Img2ImgPipeline, LatentAutoencoder

# Small enough to be invisible against any gradient worth stepping along, and
# there only so that an exactly zero gradient rescales to zero rather than NaN.
_RMS_EPSILON: Final = 1e-12

# The step is sized in float32 whatever the latent's dtype. A guidance gradient
# runs around 1e-5 per element, and squaring that in float16 lands under the
# 6e-8 subnormal floor: the RMS underflows to exactly zero, which turns
# `clip_step_rms` into a cap that never binds. Only the sizing arithmetic is
# lifted; the update is handed back in the latent's own dtype.
_UPDATE_DTYPE: Final = torch.float32


def sample_guided(
    pipeline: Img2ImgPipeline,
    photo: Float[Tensor, "batch 3 height width"],
    prompt: str,
    *,
    guidance: Guidance,
    strength: float,
    steps: int,
    noise_strength: float,
    text_guidance_scale: float,
    generator: torch.Generator,
    clip_step_rms: float | None = None,
    differentiate_through_unet: bool = False,
) -> Float[Tensor, "batch 3 height width"]:
    """Renoise `photo` and denoise it back under `prompt`, steered by `guidance`.

    Args:
        photo: NCHW, floating point, values in ``[0, 1]``; the image returned
            follows the same convention.
        strength: what the energy gradient is multiplied by before it is applied
            to the latent, at every step of the walk; 0 leaves the trajectory
            unguided.
        noise_strength: how far up the trajectory the photo is renoised, in
            ``(0, 1]``; 1 discards it entirely.
        text_guidance_scale: classifier-free guidance on the prompt, unrelated
            to `guidance`; 1 turns it off.
        generator: the only source of randomness in the loop, so a run is
            reproducible from its seed alone.
        clip_step_rms: ceiling on the RMS of the raw step, in latent-RMS units,
            sized per batch element. Below it the step is the raw one, which
            shrinks as the energy is satisfied; above it the step is exactly
            this size.
        differentiate_through_unet: keep the UNet forward on the gradient path,
            so ∂energy/∂latent carries the denoiser's Jacobian and a push the
            denoiser would undo is damped at the source (DPS, arXiv:2209.14687),
            at the cost of a UNet backward per guided step. Left off, ε̂ enters
            x̂₀ as a constant and only the decoder and the energy are traversed.

    A guided step forms the predicted clean image from the current latent,
    decodes it, evaluates `guidance` on it, and shifts the latent along the
    gradient of that energy — scaled by `strength` — before the scheduler takes
    its step. No weights are updated.
    """
    return sample_endpoint_process(
        LatentDiffusionProcess(
            pipeline,
            photo,
            prompt,
            steps=steps,
            noise_strength=noise_strength,
            text_guidance_scale=text_guidance_scale,
            generator=generator,
            differentiate_through_unet=differentiate_through_unet,
        ),
        reward=guidance,
        tilt=ClippedGradientTilt(strength=strength, clip_step_rms=clip_step_rms),
    )


class LatentDiffusionProcess:
    """Stable Diffusion img2img as an `EndpointProcess`, over one photo and one prompt.

    The state is the sampler's latent; the endpoint estimate is Tweedie's x̂₀
    decoded through the VAE, so the energy is scored in pixel space and its
    gradient reaches the latent through the decoder; the step is the scheduler's
    own. One instance drives one sample: it holds that sample's schedule,
    prompt embeddings and generator.

    Args:
        photo: NCHW, floating point, values in ``[0, 1]``; renoised to the
            noisiest walked timestep to start the walk (SDEdit).
        noise_strength: how far up the trajectory the photo is renoised, in
            ``(0, 1]``; 1 discards it entirely.
        text_guidance_scale: classifier-free guidance on the prompt; 1 turns it
            off.
        generator: the only source of randomness in the walk.
        differentiate_through_unet: keep the UNet forward on the gradient path,
            so the endpoint carries the denoiser's Jacobian (DPS,
            arXiv:2209.14687) for one UNet backward more per evaluated step.
    """

    def __init__(
        self,
        pipeline: Img2ImgPipeline,
        photo: Float[Tensor, "batch 3 height width"],
        prompt: str,
        *,
        steps: int,
        noise_strength: float,
        text_guidance_scale: float,
        generator: torch.Generator,
        differentiate_through_unet: bool = False,
    ) -> None:
        if steps < 1:
            raise ValueError(f"steps must be at least 1, got {steps}")
        if not 0.0 < noise_strength <= 1.0:
            raise ValueError(f"noise_strength must lie in (0, 1], got {noise_strength}")
        scheduler = pipeline.scheduler
        prediction_type: str = scheduler.config.prediction_type
        if prediction_type != "epsilon":
            raise ValueError(
                "the guided loop recovers x̂₀ from an epsilon noise prediction, so it "
                f"cannot drive a scheduler with prediction_type {prediction_type!r}"
            )
        scheduler.set_timesteps(steps, device=photo.device)
        self._pipeline = pipeline
        self._photo = photo
        self._generator = generator
        self._differentiate_through_unet = differentiate_through_unet
        self._timesteps = _renoised_timesteps(
            scheduler.timesteps, noise_strength=noise_strength
        )
        self._alphas_cumprod = scheduler.alphas_cumprod.to(
            device=photo.device, dtype=photo.dtype
        )
        self._text_guidance_scale = text_guidance_scale
        self._apply_text_guidance = text_guidance_scale > 1.0
        self._prompt_embeddings = _prompt_embeddings(
            pipeline,
            prompt,
            device=photo.device,
            images_per_prompt=photo.shape[0],
            apply_text_guidance=self._apply_text_guidance,
        )

    @property
    def steps(self) -> int:
        return len(self._timesteps)

    def initial_state(self) -> Float[Tensor, "batch channels height width"]:
        latent = _encode_photo(
            self._pipeline.vae, self._photo, generator=self._generator
        )
        noise = torch.randn(
            latent.shape,
            generator=self._generator,
            device=latent.device,
            dtype=latent.dtype,
        )
        return self._pipeline.scheduler.add_noise(latent, noise, self._timesteps[:1])

    def endpoint(
        self,
        state: Float[Tensor, "batch channels height width"],
        step: StepContext,
    ) -> Endpoint[Float[Tensor, "batch channels height width"]]:
        timestep = self._timesteps[step.index]
        alpha_cumprod = self._alphas_cumprod[timestep]
        if self._differentiate_through_unet:
            return _through_unet_endpoint(
                self._pipeline,
                latent=state,
                timestep=timestep,
                prompt_embeddings=self._prompt_embeddings,
                text_guidance_scale=self._text_guidance_scale,
                apply_text_guidance=self._apply_text_guidance,
                alpha_cumprod=alpha_cumprod,
            )
        noise_prediction = self._noise_prediction(state, timestep)
        differentiable_latent = state.detach().requires_grad_(True)
        return Endpoint(
            differentiable_latent,
            _predicted_original_image(
                self._pipeline.vae,
                differentiable_latent,
                noise_prediction,
                alpha_cumprod,
            ),
            noise_prediction,
        )

    def advance(
        self,
        state: Float[Tensor, "batch channels height width"],
        step: StepContext,
        *,
        prediction: Float[Tensor, "batch channels height width"] | None,
        steering: Float[Tensor, "batch channels height width"] | None,
    ) -> Float[Tensor, "batch channels height width"]:
        timestep = self._timesteps[step.index]
        noise_prediction = (
            self._noise_prediction(state, timestep)
            if prediction is None
            else prediction
        )
        if steering is not None:
            state = state + steering
        return self._pipeline.scheduler.step(
            noise_prediction, timestep, state, generator=self._generator
        ).prev_sample

    def to_image(
        self, state: Float[Tensor, "batch channels height width"]
    ) -> Float[Tensor, "batch 3 height width"]:
        return decode_latent(self._pipeline.vae, state)

    def _noise_prediction(
        self,
        latent: Float[Tensor, "batch channels height width"],
        timestep: Int[Tensor, ""],
    ) -> Float[Tensor, "batch channels height width"]:
        return _predict_noise(
            self._pipeline,
            latent,
            timestep,
            self._prompt_embeddings,
            text_guidance_scale=self._text_guidance_scale,
            apply_text_guidance=self._apply_text_guidance,
        )


@dataclass(frozen=True)
class ClippedGradientTilt:
    """The raw energy gradient as the step, capped at a fixed RMS.

    The raw gradient scales with the energy, so the step decays by itself on a
    sample the energy is already satisfied on; the cap is what stops a large
    energy running away. One evaluation per step, and none at all at strength 0.

    Args:
        strength: what the energy gradient is multiplied by; 0 leaves every step
            unsteered.
        clip_step_rms: ceiling on the RMS of the step, in the state's own units,
            sized per batch element. Below it the step passes through
            bit-for-bit; above it the step is exactly this size and points where
            the raw one pointed.
    """

    strength: float
    clip_step_rms: float | None = None

    def __post_init__(self) -> None:
        if self.strength < 0.0:
            raise ValueError(f"strength must be non-negative, got {self.strength}")
        if self.clip_step_rms is not None and self.clip_step_rms <= 0.0:
            raise ValueError(
                f"clip_step_rms must be positive, got {self.clip_step_rms}"
            )

    def __call__(
        self,
        energy_gradient: EnergyGradient,
        *,
        state: Float[Tensor, "batch *state"],
        step: StepContext,
    ) -> Float[Tensor, "batch *state"] | None:
        if self.strength == 0.0:
            return None
        # Negated because the energy is descended: the steering is the
        # displacement the state takes, not the direction the energy rises in.
        return -_guided_update(
            energy_gradient(state),
            strength=self.strength,
            clip_step_rms=self.clip_step_rms,
        )


def guidance_gradient(
    vae: LatentAutoencoder,
    guidance: Guidance,
    *,
    latent: Float[Tensor, "batch channels height width"],
    noise_prediction: Float[Tensor, "batch channels height width"],
    alpha_cumprod: Float[Tensor, ""],
) -> Float[Tensor, "batch channels height width"]:
    """The steering direction of one guided step: ∂energy/∂latent at `latent`.

    Args:
        latent: the current step's latent; needs no gradient of its own, since
            the energy is differentiated against a leaf made from it here.
        noise_prediction: this step's ε, held constant — the gradient runs
            through Tweedie's x̂₀, the decoder and `guidance`, not the UNet.
        alpha_cumprod: the scheduler's ᾱ at this step's timestep.
    """
    # `enable_grad` rather than an implicit ambient mode: the caller may be
    # inside the `no_grad` a diffusers pipeline runs under, and this region is
    # the one place in the loop where that would silently zero the guidance.
    with torch.enable_grad():
        differentiable_latent = latent.detach().requires_grad_(True)
        # Reduced here and not by the reward: `Guidance` hands back one energy
        # per batch element, and this is a single backward over the batch.
        energy = guidance(
            _predicted_original_image(
                vae, differentiable_latent, noise_prediction, alpha_cumprod
            )
        ).mean()
        (gradient,) = torch.autograd.grad(energy, differentiable_latent)
    return gradient


def guidance_gradient_through_unet(
    pipeline: Img2ImgPipeline,
    guidance: Guidance,
    *,
    latent: Float[Tensor, "batch channels height width"],
    timestep: Tensor,
    prompt_embeddings: Float[Tensor, "batch sequence embedding"],
    text_guidance_scale: float,
    apply_text_guidance: bool,
    alpha_cumprod: Float[Tensor, ""],
) -> tuple[
    Float[Tensor, "batch channels height width"],
    Float[Tensor, "batch channels height width"],
]:
    """One guided step's ε̂ and the ∂energy/∂latent taken through the UNet behind it.

    Args:
        latent: the current step's latent; needs no gradient of its own, since
            the energy is differentiated against a leaf made from it here.
        alpha_cumprod: the scheduler's ᾱ at this step's timestep.

    The returned ε̂ is the one this gradient was taken through, detached and ready
    for the scheduler, so the step pays for a single UNet forward rather than one
    per purpose. No weights are updated: the gradient is taken with respect to
    the latent alone, so no parameter accumulates a `.grad`.
    """
    with torch.enable_grad():
        endpoint = _through_unet_endpoint(
            pipeline,
            latent=latent,
            timestep=timestep,
            prompt_embeddings=prompt_embeddings,
            text_guidance_scale=text_guidance_scale,
            apply_text_guidance=apply_text_guidance,
            alpha_cumprod=alpha_cumprod,
        )
        energy = guidance(endpoint.image).mean()
        (gradient,) = torch.autograd.grad(energy, endpoint.state)
    return endpoint.prediction, gradient


def decode_latent(
    vae: LatentAutoencoder, latent: Float[Tensor, "batch channels height width"]
) -> Float[Tensor, "batch 3 height width"]:
    """Decode a latent to an image in ``[0, 1]``, keeping it on the autograd graph."""
    decoded: Tensor = vae.decode(latent / vae.config.scaling_factor).sample
    # The clamp is the ``[0, 1]`` convention, and passes the gradient wherever
    # the decode is in range; a decode pinned at the rails would zero the
    # guidance instead, which is what the loop's gradient tests are watching for.
    return ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)


def _through_unet_endpoint(
    pipeline: Img2ImgPipeline,
    *,
    latent: Float[Tensor, "batch channels height width"],
    timestep: Tensor,
    prompt_embeddings: Float[Tensor, "batch sequence embedding"],
    text_guidance_scale: float,
    apply_text_guidance: bool,
    alpha_cumprod: Float[Tensor, ""],
) -> Endpoint[Float[Tensor, "batch channels height width"]]:
    differentiable_latent = latent.detach().requires_grad_(True)
    noise_prediction = _predict_noise(
        pipeline,
        differentiable_latent,
        timestep,
        prompt_embeddings,
        text_guidance_scale=text_guidance_scale,
        apply_text_guidance=apply_text_guidance,
        differentiable=True,
    )
    # ε̂ stays attached, which is the whole difference from the constant-ε̂
    # step: x̂₀ then carries dε̂/dx_t and the denoiser damps its own undoing.
    return Endpoint(
        differentiable_latent,
        _predicted_original_image(
            pipeline.vae, differentiable_latent, noise_prediction, alpha_cumprod
        ),
        noise_prediction.detach(),
    )


def _predicted_original_image(
    vae: LatentAutoencoder,
    latent: Float[Tensor, "batch channels height width"],
    noise_prediction: Float[Tensor, "batch channels height width"],
    alpha_cumprod: Float[Tensor, ""],
) -> Float[Tensor, "batch 3 height width"]:
    predicted_original = (
        latent - (1.0 - alpha_cumprod).sqrt() * noise_prediction
    ) / alpha_cumprod.sqrt()
    return decode_latent(vae, predicted_original)


def _guided_update(
    gradient: Float[Tensor, "batch channels height width"],
    *,
    strength: float,
    clip_step_rms: float | None,
) -> Float[Tensor, "batch channels height width"]:
    update = strength * gradient.to(_UPDATE_DTYPE)
    if clip_step_rms is not None:
        # min(1, cap/RMS): a step under the cap is multiplied by exactly 1.0 and
        # so passes through bit-for-bit, and one over it lands exactly on the cap.
        scale = clip_step_rms / (_root_mean_square(update) + _RMS_EPSILON)
        update = update * scale.clamp(max=1.0)
    return update.to(gradient.dtype)


def _root_mean_square(
    tensor: Float[Tensor, "batch channels height width"],
) -> Float[Tensor, "batch 1 1 1"]:
    # Per batch element, keeping the reduction broadcastable back over it: the
    # cap is stated per image, so one loud element in a batch must not shrink
    # every other element's step, nor a quiet one let a loud element through
    # above the cap.
    return tensor.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()


def _encode_photo(
    vae: LatentAutoencoder,
    photo: Float[Tensor, "batch 3 height width"],
    *,
    generator: torch.Generator,
) -> Float[Tensor, "batch channels height width"]:
    with torch.no_grad():
        distribution = vae.encode(photo * 2.0 - 1.0).latent_dist
        latent: Tensor = distribution.sample(generator=generator)
    return latent * vae.config.scaling_factor


def _renoised_timesteps(
    timesteps: Int[Tensor, "steps"], *, noise_strength: float
) -> Int[Tensor, "renoised_steps"]:
    # SDEdit: `noise_strength` decides how much of the trajectory is redone, so
    # the head of the schedule — the noisiest steps — is the part dropped.
    denoised = min(max(int(len(timesteps) * noise_strength), 1), len(timesteps))
    return timesteps[len(timesteps) - denoised :]


def _prompt_embeddings(
    pipeline: Img2ImgPipeline,
    prompt: str,
    *,
    device: torch.device,
    images_per_prompt: int,
    apply_text_guidance: bool,
) -> Float[Tensor, "batch sequence embedding"]:
    with torch.no_grad():
        conditional, unconditional = pipeline.encode_prompt(
            prompt, device, images_per_prompt, apply_text_guidance
        )
    if not apply_text_guidance:
        return conditional
    if unconditional is None:
        raise ValueError(
            "encode_prompt returned no unconditional embedding, so classifier-free "
            "guidance has nothing to interpolate away from"
        )
    return torch.cat([unconditional, conditional])


def _predict_noise(
    pipeline: Img2ImgPipeline,
    latent: Float[Tensor, "batch channels height width"],
    timestep: Int[Tensor, ""],
    prompt_embeddings: Float[Tensor, "batch sequence embedding"],
    *,
    text_guidance_scale: float,
    apply_text_guidance: bool,
    differentiable: bool = False,
) -> Float[Tensor, "batch channels height width"]:
    batched = torch.cat([latent, latent]) if apply_text_guidance else latent
    model_input: Tensor = pipeline.scheduler.scale_model_input(batched, timestep)
    # The UNet is off the gradient path unless a caller is deliberately
    # differentiating through it: guidance otherwise differentiates the energy
    # through x̂₀'s closed form only, which costs one decoder backward per step
    # instead of a UNet one (FreeDoM, arXiv:2303.09833).
    with torch.set_grad_enabled(differentiable):
        prediction: Tensor = pipeline.unet(
            model_input, timestep, encoder_hidden_states=prompt_embeddings
        ).sample
        if not apply_text_guidance:
            return prediction
        unconditional, conditional = prediction.chunk(2)
        return unconditional + text_guidance_scale * (conditional - unconditional)
