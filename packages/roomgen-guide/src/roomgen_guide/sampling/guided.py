"""The denoising loop that steers each step by an energy gradient."""

from typing import Final

import torch
from jaxtyping import Float, Int
from torch import Tensor

from roomgen_guide.guidance.base import Guidance
from roomgen_guide.guidance.schedule import GuidanceSchedule
from roomgen_guide.sampling.pipeline import Img2ImgPipeline, LatentAutoencoder

# Small enough to be invisible against any gradient worth stepping along, and
# there only so that an exactly zero gradient rescales to zero rather than NaN.
_RMS_EPSILON: Final = 1e-12


def sample_guided(
    pipeline: Img2ImgPipeline,
    photo: Float[Tensor, "batch 3 height width"],
    prompt: str,
    *,
    guidance: Guidance,
    schedule: GuidanceSchedule,
    steps: int,
    noise_strength: float,
    text_guidance_scale: float,
    generator: torch.Generator,
    normalize_gradient: bool = False,
    clip_step_rms: float | None = None,
    differentiate_through_unet: bool = False,
) -> Float[Tensor, "batch 3 height width"]:
    """Renoise `photo` and denoise it back under `prompt`, steered by `guidance`.

    Args:
        photo: NCHW, floating point, values in ``[0, 1]``; the image returned
            follows the same convention.
        noise_strength: how far up the trajectory the photo is renoised, in
            ``(0, 1]``; 1 discards it entirely.
        text_guidance_scale: classifier-free guidance on the prompt, unrelated
            to `guidance`; 1 turns it off.
        generator: the only source of randomness in the loop, so a run is
            reproducible from its seed alone.
        normalize_gradient: step along the gradient divided by its own RMS, which
            makes the strength `schedule` returns a step size in latent-RMS units
            — bounded per step and invariant to the absolute scale of the energy
            — instead of a multiplier on a gradient whose size the energy sets.
        clip_step_rms: ceiling on the RMS of the raw step, in latent-RMS units.
            Below it the step is the raw one, which shrinks as the energy is
            satisfied; above it the step is exactly this size. Mutually
            exclusive with `normalize_gradient`, whose step is already fixed-RMS.
        differentiate_through_unet: keep the UNet forward on the gradient path,
            so ∂energy/∂latent carries the denoiser's Jacobian and a push the
            denoiser would undo is damped at the source (DPS, arXiv:2209.14687),
            at the cost of a UNet backward per guided step. Left off, ε̂ enters
            x̂₀ as a constant and only the decoder and the energy are traversed.

    A guided step forms the predicted clean image from the current latent,
    decodes it, evaluates `guidance` on it, and shifts the latent along the
    gradient of that energy — scaled by `schedule` — before the scheduler takes
    its step. No weights are updated.
    """
    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")
    if not 0.0 < noise_strength <= 1.0:
        raise ValueError(f"noise_strength must lie in (0, 1], got {noise_strength}")
    if clip_step_rms is not None:
        if normalize_gradient:
            raise ValueError(
                "clip_step_rms caps the raw step, but normalize_gradient already "
                "fixes every step at the schedule's strength; pass one or the other"
            )
        if clip_step_rms <= 0.0:
            raise ValueError(f"clip_step_rms must be positive, got {clip_step_rms}")

    scheduler = pipeline.scheduler
    prediction_type: str = scheduler.config.prediction_type
    if prediction_type != "epsilon":
        raise ValueError(
            "the guided loop recovers x̂₀ from an epsilon noise prediction, so it "
            f"cannot drive a scheduler with prediction_type {prediction_type!r}"
        )

    scheduler.set_timesteps(steps, device=photo.device)
    timesteps = _renoised_timesteps(scheduler.timesteps, noise_strength=noise_strength)
    alphas_cumprod = scheduler.alphas_cumprod.to(device=photo.device, dtype=photo.dtype)

    latent = _encode_photo(pipeline.vae, photo, generator=generator)
    noise = torch.randn(
        latent.shape, generator=generator, device=latent.device, dtype=latent.dtype
    )
    latent = scheduler.add_noise(latent, noise, timesteps[:1])

    apply_text_guidance = text_guidance_scale > 1.0
    prompt_embeddings = _prompt_embeddings(
        pipeline,
        prompt,
        device=photo.device,
        images_per_prompt=photo.shape[0],
        apply_text_guidance=apply_text_guidance,
    )

    total_steps = len(timesteps)
    for step, timestep in enumerate(timesteps):
        strength = schedule(step, total_steps)
        steering: Tensor | None = None
        if strength != 0.0 and differentiate_through_unet:
            noise_prediction, steering = guidance_gradient_through_unet(
                pipeline,
                guidance,
                latent=latent,
                timestep=timestep,
                prompt_embeddings=prompt_embeddings,
                text_guidance_scale=text_guidance_scale,
                apply_text_guidance=apply_text_guidance,
                alpha_cumprod=alphas_cumprod[timestep],
            )
        else:
            noise_prediction = _predict_noise(
                pipeline,
                latent,
                timestep,
                prompt_embeddings,
                text_guidance_scale=text_guidance_scale,
                apply_text_guidance=apply_text_guidance,
            )
            if strength != 0.0:
                steering = guidance_gradient(
                    pipeline.vae,
                    guidance,
                    latent=latent,
                    noise_prediction=noise_prediction,
                    alpha_cumprod=alphas_cumprod[timestep],
                )
        if steering is not None:
            latent = latent - _guided_update(
                steering,
                strength=strength,
                normalize_gradient=normalize_gradient,
                clip_step_rms=clip_step_rms,
            )
        latent = scheduler.step(noise_prediction, timestep, latent).prev_sample

    return decode_latent(pipeline.vae, latent)


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
        predicted_original = (
            differentiable_latent - (1.0 - alpha_cumprod).sqrt() * noise_prediction
        ) / alpha_cumprod.sqrt()
        energy = guidance(decode_latent(vae, predicted_original))
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
        predicted_original = (
            differentiable_latent - (1.0 - alpha_cumprod).sqrt() * noise_prediction
        ) / alpha_cumprod.sqrt()
        energy = guidance(decode_latent(pipeline.vae, predicted_original))
        (gradient,) = torch.autograd.grad(energy, differentiable_latent)
    return noise_prediction.detach(), gradient


def decode_latent(
    vae: LatentAutoencoder, latent: Float[Tensor, "batch channels height width"]
) -> Float[Tensor, "batch 3 height width"]:
    """Decode a latent to an image in ``[0, 1]``, keeping it on the autograd graph."""
    decoded: Tensor = vae.decode(latent / vae.config.scaling_factor).sample
    # The clamp is the ``[0, 1]`` convention, and passes the gradient wherever
    # the decode is in range; a decode pinned at the rails would zero the
    # guidance instead, which is what the loop's gradient tests are watching for.
    return ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)


def _guided_update(
    gradient: Float[Tensor, "batch channels height width"],
    *,
    strength: float,
    normalize_gradient: bool,
    clip_step_rms: float | None,
) -> Float[Tensor, "batch channels height width"]:
    if normalize_gradient:
        # Unit RMS whatever the energy's units are, so `strength` becomes the
        # size of the step rather than a multiplier on it.
        gradient = gradient / (_root_mean_square(gradient) + _RMS_EPSILON)
    update = strength * gradient
    if clip_step_rms is None:
        return update
    # min(1, cap/RMS): a step under the cap is multiplied by exactly 1.0 and so
    # passes through bit-for-bit, and one over it lands exactly on the cap.
    scale = clip_step_rms / (_root_mean_square(update) + _RMS_EPSILON)
    return update * scale.clamp(max=1.0)


def _root_mean_square(
    tensor: Float[Tensor, "batch channels height width"],
) -> Float[Tensor, ""]:
    return tensor.square().mean().sqrt()


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
