"""The denoising loop that steers each step by an energy gradient."""

from __future__ import annotations

import torch
from jaxtyping import Float
from torch import Tensor

from roomgen_guide.guidance.base import Guidance
from roomgen_guide.guidance.schedule import GuidanceSchedule
from roomgen_guide.sampling.pipeline import Img2ImgPipeline, LatentAutoencoder


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

    A guided step forms the predicted clean image from the current latent,
    decodes it, evaluates `guidance` on it, and shifts the latent along the
    gradient of that energy — scaled by `schedule` — before the scheduler takes
    its step. No weights are updated.
    """
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
    timesteps = _renoised_timesteps(scheduler.timesteps, noise_strength=noise_strength)
    alphas_cumprod: Tensor = scheduler.alphas_cumprod.to(
        device=photo.device, dtype=photo.dtype
    )

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
        noise_prediction = _predict_noise(
            pipeline,
            latent,
            timestep,
            prompt_embeddings,
            text_guidance_scale=text_guidance_scale,
            apply_text_guidance=apply_text_guidance,
        )
        strength = schedule(step, total_steps)
        if strength != 0.0:
            latent = latent - strength * guidance_gradient(
                pipeline.vae,
                guidance,
                latent=latent,
                noise_prediction=noise_prediction,
                alpha_cumprod=alphas_cumprod[timestep],
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


def decode_latent(
    vae: LatentAutoencoder, latent: Float[Tensor, "batch channels height width"]
) -> Float[Tensor, "batch 3 height width"]:
    """Decode a latent to an image in ``[0, 1]``, keeping it on the autograd graph."""
    decoded: Tensor = vae.decode(latent / vae.config.scaling_factor).sample
    return ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)


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


def _renoised_timesteps(timesteps: Tensor, *, noise_strength: float) -> Tensor:
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
    timestep: Tensor,
    prompt_embeddings: Float[Tensor, "batch sequence embedding"],
    *,
    text_guidance_scale: float,
    apply_text_guidance: bool,
) -> Float[Tensor, "batch channels height width"]:
    batched = torch.cat([latent, latent]) if apply_text_guidance else latent
    model_input: Tensor = pipeline.scheduler.scale_model_input(batched, timestep)
    # The UNet is off the gradient path on purpose: guidance differentiates the
    # energy through x̂₀'s closed form only, which costs one decoder backward
    # per step instead of a UNet one (FreeDoM, arXiv:2303.09833).
    with torch.no_grad():
        prediction: Tensor = pipeline.unet(
            model_input, timestep, encoder_hidden_states=prompt_embeddings
        ).sample
    if not apply_text_guidance:
        return prediction
    unconditional, conditional = prediction.chunk(2)
    return unconditional + text_guidance_scale * (conditional - unconditional)
