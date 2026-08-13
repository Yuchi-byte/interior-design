"""Two-stage renovation on FMRG's FLUX flow map: strip the room, then furnish it.

Stage 1 constructs the room's geometric ground truth — the same shell with the
furniture gone — from pure noise under the depth reward. Stage 2 treats that
render as the reference for a second guided walk that puts furniture back, so
the constraint the renovation is held to is a shell covering nearly the whole
frame rather than the 45% the photograph leaves unoccluded.

The sampler config is FMRG's own `fmrgj_nfe100`
(`packages/fmrg/experiments/geneval_nfe_sweep/generate_fmrg.sh`): a 208-point
sigma grid, the walk truncated to its first 51 entries plus one uncontrolled
jump to data, eta 5.0, `n_opt` 1, `use_adam=True`.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
import torchvision.transforms.functional as TF
import transformers.utils as _transformers_utils
from jaxtyping import Float
from PIL import Image
from torch import Tensor

# diffusers 0.34 — the release `packages/fmrg/NOTICE.md` pins — imports three
# weight-filename constants at module scope that transformers 5 dropped. The
# pod needs transformers 5 for the other side of the stack: EoMT's
# `post_process_semantic_segmentation` on 4.57 requires a `patch_offsets`
# argument the repo's segmenter does not pass, so downgrading trades this
# ImportError for a broken segmenter. The constants only ever name files no
# FLUX load reads, so restoring them is inert.
for _name, _value in (
    ("FLAX_WEIGHTS_NAME", "flax_model.msgpack"),
    ("TF2_WEIGHTS_NAME", "tf_model.h5"),
    ("TF_WEIGHTS_NAME", "model.ckpt"),
):
    if not hasattr(_transformers_utils, _name):
        setattr(_transformers_utils, _name, _value)

from roomgen_geometry.depth.depth_anything import DepthAnythingV2
from roomgen_geometry.masking import scorable_mask
from roomgen_geometry.reference import GeometricReference
from roomgen_geometry.segmentation.eomt import EoMT
from roomgen_guide.adapters.flow_map import FlowMapProcess
from roomgen_guide.adapters.fmrg_flux import FluxFlowMapBackend, load_flow_map_sampler
from roomgen_guide.eval.clip import ClipPromptAdherence
from roomgen_guide.eval.depth import depth_error, shell_coverage
from roomgen_guide.guidance.geometric import GeometricGuidance
from roomgen_guide.process.driver import sample_endpoint_process

# FMRG's `fmrgj_nfe100`, read off its own sweep script rather than re-tuned.
GRID_STEPS: Final = 208
"""The dense sigma grid `num_steps` names; the walk never visits most of it."""
WINDOW_SIGMAS: Final = 51
"""`early_stop`: how many of the grid's sigmas the walk actually traverses."""
GUIDED_STEPS: Final = 50
"""The trailing jump to data is uncontrolled, so the walk is 51 steps / 50 rewards."""
ETA: Final = 5.0
"""`step_size`, FMRG's guidance strength lambda."""

RESOLUTION: Final = 512
EROSION_RADIUS: Final = 2
FMRG_PATH: Final = Path("/root/roomgen/packages/fmrg")
LORA_PATH: Final = Path("/root/checkpoints/flux-flowmap-lora-512")
FLUX_IGNORE: Final = (
    "*.bin",
    "flux1-dev.safetensors",
    "ae.safetensors",
    "*.onnx",
    "*.msgpack",
    "*.h5",
)
"""The diffusers-format load reads none of these; the two root files are 24GB."""

# The blur probe: whether a render's depth match survives losing its
# high-frequency texture, which is what separates real geometry from texture
# painted to satisfy the depth network.
BLUR_KERNEL: Final = 9
BLUR_SIGMA: Final = 2.0


@dataclass(frozen=True)
class FmrgWindowBackend:
    """`FluxFlowMapBackend` restricted to FMRG's high-noise window.

    FMRG's `early_stop` builds the dense `num_steps` grid, walks only its first
    `window` sigmas, and closes with a single uncontrolled flow-map jump to
    data. That is the whole "guide inside the high-noise window and stop early"
    prescription, and the plain schedule cannot express it: asking the backend
    for 51 steps would spread them over the entire trajectory instead of
    keeping them bunched at the noisy end.
    """

    inner: FluxFlowMapBackend
    grid_steps: int = GRID_STEPS
    window: int = WINDOW_SIGMAS

    def schedule(
        self,
        *,
        steps: int,
        start_sigma: float,
        latent: Float[Tensor, "batch channels height width"],
    ) -> list[float]:
        full = self.inner.schedule(
            steps=self.grid_steps, start_sigma=start_sigma, latent=latent
        )
        return [*full[: self.window], 0.0]

    def encode(
        self,
        image: Float[Tensor, "batch 3 height width"],
        *,
        generator: torch.Generator,
    ) -> Float[Tensor, "batch channels height width"]:
        return self.inner.encode(image, generator=generator)

    def decode(
        self, latent: Float[Tensor, "batch channels height width"]
    ) -> Float[Tensor, "batch 3 height width"]:
        return self.inner.decode(latent)

    def velocity(
        self,
        latent: Float[Tensor, "batch channels height width"],
        *,
        sigma: float,
        target_sigma: float,
    ) -> Float[Tensor, "batch channels height width"]:
        return self.inner.velocity(latent, sigma=sigma, target_sigma=target_sigma)


@dataclass(frozen=True)
class FmrgJacobianTilt:
    """FMRG-J's guided step: one Adam iteration on the state, scaled by the step's dt.

    Upstream runs `use_adam=True` with `num_optim_iters` 1, so the inner
    optimisation is a single Adam step from zeroed moments — which is exactly
    ``lr * g / (|g| + eps)`` elementwise, an RMS-`lr` step along the gradient's
    sign rather than a scaled raw gradient. The learning rate decays linearly
    across the walk, and the resulting displacement enters the Euler update
    multiplied by that step's own `dt`, because upstream adds it to the
    *velocity* and not to the state. Both matter: without the `dt` factor the
    same eta is a step some fifty times larger.

    `grad_norm_mode` is deliberately absent — upstream's Adam branch ignores the
    velocity-norm rescaling that `--normalize_grad` sets up, so FMRG-J's headline
    config never applies it.
    """

    sigmas: tuple[float, ...]
    eta: float
    guided_steps: int = GUIDED_STEPS
    eps: float = 1e-8

    def __call__(
        self,
        energy_gradient: object,
        *,
        state: Float[Tensor, "batch channels height width"],
        step: object,
    ) -> Float[Tensor, "batch channels height width"] | None:
        index, total = step.index, step.total  # type: ignore[attr-defined]
        if self.eta == 0.0 or index >= self.guided_steps:
            return None
        gradient = energy_gradient(state)  # type: ignore[operator]
        learning_rate = self.eta * (1.0 - index / total)
        dt = self.sigmas[index] - self.sigmas[index + 1]
        # Negated because the energy is descended: this is the displacement.
        return -(dt * learning_rate) * gradient / (gradient.abs() + self.eps)


def load_photo(path: Path, *, device: torch.device) -> Float[Tensor, "1 3 h w"]:
    """The photo as a square `RESOLUTION` tensor in [0, 1], NCHW.

    The frame is squashed to square rather than centre-cropped: the benchmark
    photos are 3:2, and cropping throws away the side walls that carry most of
    the room's perspective — the shell the whole method is about. Aspect
    distortion is uniform across every arm and the reference, so it costs the
    comparison nothing.
    """
    image = Image.open(path).convert("RGB")
    tensor = TF.to_tensor(image).unsqueeze(0)
    return TF.resize(tensor, [RESOLUTION, RESOLUTION], antialias=True).to(device)


def save_image(tensor: Float[Tensor, "1 3 h w"], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    TF.to_pil_image(tensor[0].clamp(0, 1).cpu()).save(path)


def save_depth(
    tensor: Float[Tensor, "1 3 h w"], depth: DepthAnythingV2, path: Path
) -> None:
    with torch.no_grad():
        disparity = depth(tensor)[0]
    lo, hi = float(disparity.min()), float(disparity.max())
    normalised = (disparity - lo) / max(hi - lo, 1e-8)
    TF.to_pil_image(normalised.cpu()).save(path)


def evaluation_mask(
    image: Float[Tensor, "1 3 h w"],
    *,
    reference: GeometricReference,
    segmenter: EoMT,
) -> Tensor:
    """The render's own eval mask: the reference shell narrowed by this render's."""
    with torch.no_grad():
        segmentation = segmenter(image)
    own = scorable_mask(
        segmentation, mode="intersect_shell", erosion_radius=EROSION_RADIUS
    )
    return reference.mask & own


def evaluate(
    image: Float[Tensor, "1 3 h w"],
    *,
    reference: GeometricReference,
    depth: DepthAnythingV2,
    segmenter: EoMT,
) -> dict[str, float]:
    """Depth error against `reference`, the coverage it was read over, and the blur probe."""
    mask = evaluation_mask(image, reference=reference, segmenter=segmenter)
    blurred = TF.gaussian_blur(image, [BLUR_KERNEL, BLUR_KERNEL], [BLUR_SIGMA])
    return {
        "depth_error": round(
            depth_error(
                image, reference=reference, depth_estimator=depth, mask=mask
            ),
            6,
        ),
        "blurred_depth_error": round(
            depth_error(
                blurred, reference=reference, depth_estimator=depth, mask=mask
            ),
            6,
        ),
        "shell_coverage": round(shell_coverage(mask, reference=reference), 4),
    }


def shell_fraction(image: Float[Tensor, "1 3 h w"], *, segmenter: EoMT) -> float:
    """How much of the frame this render's own segmentation calls architectural shell."""
    with torch.no_grad():
        segmentation = segmenter(image)
    own = scorable_mask(
        segmentation, mode="intersect_shell", erosion_radius=EROSION_RADIUS
    )
    return round(float(own.float().mean()), 4)


def run_sample(
    backend: FmrgWindowBackend,
    initial: Float[Tensor, "1 3 h w"],
    *,
    start_sigma: float,
    seed: int,
    reward: GeometricGuidance,
    eta: float,
    device: torch.device,
) -> tuple[Float[Tensor, "1 3 h w"], dict[str, float]]:
    """One walk. `initial` is what sigma < 1 renoises from; at sigma 1 it is discarded."""
    generator = torch.Generator(device=device).manual_seed(seed)
    process = FlowMapProcess(
        backend,
        initial,
        steps=WINDOW_SIGMAS,
        start_sigma=start_sigma,
        generator=generator,
    )
    sigmas = tuple(
        backend.schedule(
            steps=WINDOW_SIGMAS,
            start_sigma=start_sigma,
            latent=torch.zeros(1, 16, RESOLUTION // 8, RESOLUTION // 8, device=device),
        )
    )
    tilt = FmrgJacobianTilt(sigmas=sigmas, eta=eta)
    torch.cuda.reset_peak_memory_stats()
    began = time.time()
    image = sample_endpoint_process(process, reward=reward, tilt=tilt)
    stats = {
        "seconds": round(time.time() - began, 1),
        "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "window_start": round(sigmas[0], 4),
        "window_end": round(sigmas[GUIDED_STEPS], 4),
    }
    return image.float(), stats


def build_sampler(prompt: str, device: torch.device) -> FmrgWindowBackend:
    # The *cached snapshot directory*, not the repo id: handed the id,
    # `from_pretrained` re-resolves the whole repo and pulls the 24GB
    # single-file `flux1-dev.safetensors` and the `.bin` duplicates that the
    # diffusers-format load never reads, which overruns the pod's disk quota
    # mid-run. `local_files_only` turns that into a path lookup.
    from huggingface_hub import snapshot_download

    model_path = snapshot_download(
        "black-forest-labs/FLUX.1-dev",
        local_files_only=True,
        # The same filter the weights were fetched under, so the completeness
        # check does not fail over the two root files deliberately never pulled.
        ignore_patterns=FLUX_IGNORE,
    )
    sampler = load_flow_map_sampler(
        fmrg_path=FMRG_PATH,
        lora_path=LORA_PATH,
        prompt=prompt,
        device=device,
        model_id=model_path,
        dtype=torch.bfloat16,
    )
    return FmrgWindowBackend(FluxFlowMapBackend(sampler=sampler))


def free(backend: object) -> None:
    del backend
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("stage1", "stage2"))
    parser.add_argument("--photo", type=Path, default=Path(
        "/root/photos/scandi_living_room.jpg"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="stage1: one or more empty-room prompts, each its own sampler load.",
    )
    parser.add_argument("--empty", type=Path, help="stage2: the stage-1 render.")
    parser.add_argument("--start-sigma", type=float, default=1.0)
    parser.add_argument("--tag", default="empty")
    parser.add_argument(
        "--furnish-prompt",
        default=(
            "modern and minimalistic living room with walnut, white, and black "
            "tones, plus a pop of colour"
        ),
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    args.out.mkdir(parents=True, exist_ok=True)

    depth = DepthAnythingV2(device=device)
    segmenter = EoMT(device=device)
    photo = load_photo(args.photo, device=device)
    reference = GeometricReference.from_photo(
        photo,
        depth_estimator=depth,
        segmenter=segmenter,
        erosion_radius=EROSION_RADIUS,
    )

    save_image(photo, args.out / "photo.png")
    save_depth(photo, depth, args.out / "depth_photo.png")

    results: dict[str, object] = {
        "photo": {
            **evaluate(photo, reference=reference, depth=depth, segmenter=segmenter),
            "shell_fraction": shell_fraction(photo, segmenter=segmenter),
        },
        "config": {
            "grid_steps": GRID_STEPS,
            "window_sigmas": WINDOW_SIGMAS,
            "guided_steps": GUIDED_STEPS,
            "eta": ETA,
            "seed": args.seed,
            "erosion_radius": EROSION_RADIUS,
            "mask_mode": "intersect_shell",
        },
        "runs": [],
    }
    runs: list[dict[str, object]] = results["runs"]  # type: ignore[assignment]

    def record(name: str, image: Tensor, stats: dict[str, float], **extra: object) -> None:
        save_image(image, args.out / f"{name}.png")
        save_depth(image, depth, args.out / f"depth_{name}.png")
        row: dict[str, object] = {"name": name, **stats, **extra}
        row["shell_fraction"] = shell_fraction(image, segmenter=segmenter)
        row["vs_photo"] = evaluate(
            image, reference=reference, depth=depth, segmenter=segmenter
        )
        runs.append(row)
        print(f"[{name}] {json.dumps(row)}", flush=True)
        # Per-tag in stage 1, where each prompt is its own process writing into
        # the same directory and a shared filename would race.
        stem = f"results_{args.tag}" if args.stage == "stage1" else "results"
        (args.out / f"{stem}.json").write_text(json.dumps(results, indent=2))

    if args.stage == "stage1":
        # One prompt per process: the sampler holds ~24GiB of transformer, and
        # dropping it inside the loop does not reliably give it back, so the
        # second load OOMs. The caller loops over invocations instead.
        (prompt,) = args.prompt
        backend = build_sampler(prompt, device)
        reward = GeometricGuidance(
            depth_estimator=depth,
            reference=reference,
            segmenter=segmenter,
            erosion_radius=EROSION_RADIUS,
        )
        image, stats = run_sample(
            backend,
            photo,
            start_sigma=args.start_sigma,
            seed=args.seed,
            reward=reward,
            eta=ETA,
            device=device,
        )
        record(args.tag, image, stats, prompt=prompt,
               start_sigma=args.start_sigma, arm="guided")
        return

    # stage 2: the empty render is the new reference, and the furnish walk starts
    # from it rather than from the photograph.
    empty = load_photo(args.empty, device=device)
    reference_empty = GeometricReference.from_photo(
        empty,
        depth_estimator=depth,
        segmenter=segmenter,
        erosion_radius=EROSION_RADIUS,
    )
    results["empty"] = {
        **evaluate(empty, reference=reference, depth=depth, segmenter=segmenter),
        "shell_fraction": shell_fraction(empty, segmenter=segmenter),
    }

    backend = build_sampler(args.furnish_prompt, device)
    guided_reward = GeometricGuidance(
        depth_estimator=depth,
        reference=reference_empty,
        segmenter=segmenter,
        erosion_radius=EROSION_RADIUS,
    )
    photo_reward = GeometricGuidance(
        depth_estimator=depth,
        reference=reference,
        segmenter=segmenter,
        erosion_radius=EROSION_RADIUS,
    )

    def record2(name: str, image: Tensor, stats: dict[str, float], **extra: object) -> None:
        # The second hop is scored against the empty render it was held to, as
        # well as end-to-end against the photograph the whole pipeline is about.
        extra["vs_empty"] = evaluate(
            image, reference=reference_empty, depth=depth, segmenter=segmenter
        )
        record(name, image, stats, **extra)

    for start_sigma in (0.8, 0.9):
        image, stats = run_sample(
            backend, empty, start_sigma=start_sigma, seed=args.seed,
            reward=guided_reward, eta=ETA, device=device,
        )
        record2(f"furnished_s{int(start_sigma * 10):02d}", image, stats,
                start_sigma=start_sigma, arm="guided", reward_reference="empty")

    image, stats = run_sample(
        backend, empty, start_sigma=0.9, seed=args.seed,
        reward=guided_reward, eta=0.0, device=device,
    )
    record2("furnished_s09_unguided", image, stats, start_sigma=0.9, arm="unguided",
            reward_reference="none")

    # The one-stage baseline: the same prompt and budget straight from noise,
    # held to the photograph's own 45% shell.
    image, stats = run_sample(
        backend, photo, start_sigma=1.0, seed=args.seed,
        reward=photo_reward, eta=ETA, device=device,
    )
    record("baseline_s10_guided", image, stats, start_sigma=1.0, arm="guided",
           reward_reference="photo")

    image, stats = run_sample(
        backend, photo, start_sigma=1.0, seed=args.seed,
        reward=photo_reward, eta=0.0, device=device,
    )
    record("baseline_s10_unguided", image, stats, start_sigma=1.0, arm="unguided",
           reward_reference="none")

    free(backend)

    # Adherence last, so the sampler's memory is already gone.
    clip = ClipPromptAdherence(device=device)

    def adherence(image: Tensor) -> float:
        return round(
            clip(
                image,
                prompt=args.furnish_prompt,
                photo=photo,
                reference=reference,
            ),
            4,
        )

    for row in runs:
        row["adherence"] = adherence(
            load_photo(args.out / f"{row['name']}.png", device=device)
        )
    results["photo"]["adherence"] = adherence(photo)  # type: ignore[index]
    results["empty"]["adherence"] = adherence(empty)  # type: ignore[index]
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    print("STAGE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
