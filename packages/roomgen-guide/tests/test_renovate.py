import inspect

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from torch import Tensor

from roomgen_fakes import FakeDepthEstimator, FakeSegmenter
from roomgen_geometry.reference import GeometricReference
from roomgen_guide.guidance.geometric import GeometricGuidance, MaskObserver
from roomgen_guide.renovate import renovate
from roomgen_guide.sampling.guided import sample_guided
from roomgen_guide.sampling.pipeline import Img2ImgPipeline
from roomgen_guide_fakes import FakePipeline

_PROMPT = "a mid-century living room"
_STEPS = 4
_NOISE_STRENGTH = 0.6
_TEXT_GUIDANCE_SCALE = 7.5
_EROSION_RADIUS = 2
_STRENGTH = 500.0
# Below the raw step this stack takes at `_STRENGTH`, so the cap binds.
_CLIP_STEP_RMS = 0.01
# `FakeSegmenter` labels by frame position and ignores pixel content, so a
# second layout is how a prediction whose shell no longer matches the photo's is
# staged: handed a reference frozen from the default layout, this one is the
# sample's disagreeing view of it.
_RELAID_OUT_SEGMENTER = FakeSegmenter(floor_fraction=0.5)


def _renovate(
    pipeline: FakePipeline,
    photo: Tensor,
    depth_estimator: FakeDepthEstimator,
    segmenter: FakeSegmenter,
    *,
    strength: float,
    seed: int = 0,
    clip_step_rms: float | None = None,
    reference: GeometricReference | None = None,
    resegment: bool = True,
    on_mask: MaskObserver | None = None,
) -> Tensor:
    return renovate(
        photo,
        _PROMPT,
        pipeline=pipeline,
        depth_estimator=depth_estimator,
        segmenter=segmenter,
        strength=strength,
        erosion_radius=_EROSION_RADIUS,
        steps=_STEPS,
        noise_strength=_NOISE_STRENGTH,
        text_guidance_scale=_TEXT_GUIDANCE_SCALE,
        generator=torch.Generator().manual_seed(seed),
        clip_step_rms=clip_step_rms,
        reference=reference,
        resegment=resegment,
        on_mask=on_mask,
    )


def test_the_same_seed_reproduces_the_same_renovation(
    fake_pipeline: FakePipeline,
    room_photo: Tensor,
    fake_depth_estimator: FakeDepthEstimator,
    fake_segmenter: FakeSegmenter,
) -> None:
    renovated = _renovate(
        fake_pipeline,
        room_photo,
        fake_depth_estimator,
        fake_segmenter,
        strength=_STRENGTH,
    )
    repeated = _renovate(
        fake_pipeline,
        room_photo,
        fake_depth_estimator,
        fake_segmenter,
        strength=_STRENGTH,
    )
    reseeded = _renovate(
        fake_pipeline,
        room_photo,
        fake_depth_estimator,
        fake_segmenter,
        strength=_STRENGTH,
        seed=1,
    )

    assert renovated.shape == room_photo.shape
    assert float(renovated.min()) >= 0.0
    assert float(renovated.max()) <= 1.0
    assert torch.equal(renovated, repeated)
    assert not torch.equal(renovated, reseeded)


def test_the_renovation_is_steered_by_the_photos_own_frozen_geometry(
    fake_pipeline: FakePipeline,
    room_photo: Tensor,
    fake_depth_estimator: FakeDepthEstimator,
    fake_segmenter: FakeSegmenter,
) -> None:
    reference = GeometricReference.from_photo(
        room_photo,
        depth_estimator=fake_depth_estimator,
        segmenter=fake_segmenter,
        erosion_radius=_EROSION_RADIUS,
    )
    sampled = sample_guided(
        fake_pipeline,
        room_photo,
        _PROMPT,
        guidance=GeometricGuidance(
            depth_estimator=fake_depth_estimator, reference=reference
        ),
        strength=_STRENGTH,
        steps=_STEPS,
        noise_strength=_NOISE_STRENGTH,
        text_guidance_scale=_TEXT_GUIDANCE_SCALE,
        generator=torch.Generator().manual_seed(0),
    )

    renovated = _renovate(
        fake_pipeline,
        room_photo,
        fake_depth_estimator,
        fake_segmenter,
        strength=_STRENGTH,
    )

    assert torch.equal(renovated, sampled)


def test_the_clipped_step_reaches_the_sampler(
    fake_pipeline: FakePipeline,
    room_photo: Tensor,
    fake_depth_estimator: FakeDepthEstimator,
    fake_segmenter: FakeSegmenter,
) -> None:
    reference = GeometricReference.from_photo(
        room_photo,
        depth_estimator=fake_depth_estimator,
        segmenter=fake_segmenter,
        erosion_radius=_EROSION_RADIUS,
    )
    sampled = sample_guided(
        fake_pipeline,
        room_photo,
        _PROMPT,
        guidance=GeometricGuidance(
            depth_estimator=fake_depth_estimator, reference=reference
        ),
        strength=_STRENGTH,
        steps=_STEPS,
        noise_strength=_NOISE_STRENGTH,
        text_guidance_scale=_TEXT_GUIDANCE_SCALE,
        generator=torch.Generator().manual_seed(0),
        clip_step_rms=_CLIP_STEP_RMS,
    )

    clipped = _renovate(
        fake_pipeline,
        room_photo,
        fake_depth_estimator,
        fake_segmenter,
        strength=_STRENGTH,
        clip_step_rms=_CLIP_STEP_RMS,
    )
    uncapped = _renovate(
        fake_pipeline,
        room_photo,
        fake_depth_estimator,
        fake_segmenter,
        strength=_STRENGTH,
    )

    assert torch.equal(clipped, sampled)
    assert not torch.equal(clipped, uncapped)


def test_a_prebuilt_reference_renovates_exactly_as_one_built_in_the_call(
    fake_pipeline: FakePipeline,
    room_photo: Tensor,
    fake_depth_estimator: FakeDepthEstimator,
    fake_segmenter: FakeSegmenter,
) -> None:
    reference = GeometricReference.from_photo(
        room_photo,
        depth_estimator=fake_depth_estimator,
        segmenter=fake_segmenter,
        erosion_radius=_EROSION_RADIUS,
    )

    handed = _renovate(
        fake_pipeline,
        room_photo,
        fake_depth_estimator,
        fake_segmenter,
        strength=_STRENGTH,
        reference=reference,
    )
    built = _renovate(
        fake_pipeline,
        room_photo,
        fake_depth_estimator,
        fake_segmenter,
        strength=_STRENGTH,
    )

    assert torch.equal(handed, built)


def test_reference_only_scoring_ignores_the_predictions_own_shell(
    fake_pipeline: FakePipeline,
    room_photo: Tensor,
    fake_depth_estimator: FakeDepthEstimator,
    fake_segmenter: FakeSegmenter,
) -> None:
    reference = GeometricReference.from_photo(
        room_photo,
        depth_estimator=fake_depth_estimator,
        segmenter=fake_segmenter,
        erosion_radius=_EROSION_RADIUS,
    )
    sampled = sample_guided(
        fake_pipeline,
        room_photo,
        _PROMPT,
        guidance=GeometricGuidance(
            depth_estimator=fake_depth_estimator, reference=reference
        ),
        strength=_STRENGTH,
        steps=_STEPS,
        noise_strength=_NOISE_STRENGTH,
        text_guidance_scale=_TEXT_GUIDANCE_SCALE,
        generator=torch.Generator().manual_seed(0),
    )

    reference_only = _renovate(
        fake_pipeline,
        room_photo,
        fake_depth_estimator,
        _RELAID_OUT_SEGMENTER,
        strength=_STRENGTH,
        reference=reference,
        resegment=False,
    )

    assert torch.equal(reference_only, sampled)


def test_resegmenting_scores_a_different_shell_than_the_reference_alone(
    fake_pipeline: FakePipeline,
    room_photo: Tensor,
    fake_depth_estimator: FakeDepthEstimator,
    fake_segmenter: FakeSegmenter,
) -> None:
    reference = GeometricReference.from_photo(
        room_photo,
        depth_estimator=fake_depth_estimator,
        segmenter=fake_segmenter,
        erosion_radius=_EROSION_RADIUS,
    )

    reference_only = _renovate(
        fake_pipeline,
        room_photo,
        fake_depth_estimator,
        _RELAID_OUT_SEGMENTER,
        strength=_STRENGTH,
        reference=reference,
        resegment=False,
    )
    resegmented = _renovate(
        fake_pipeline,
        room_photo,
        fake_depth_estimator,
        _RELAID_OUT_SEGMENTER,
        strength=_STRENGTH,
        reference=reference,
        resegment=True,
    )

    assert not torch.equal(resegmented, reference_only)


def test_the_mask_hook_reports_what_every_guided_step_scored(
    fake_pipeline: FakePipeline,
    room_photo: Tensor,
    fake_depth_estimator: FakeDepthEstimator,
    fake_segmenter: FakeSegmenter,
) -> None:
    reference = GeometricReference.from_photo(
        room_photo,
        depth_estimator=fake_depth_estimator,
        segmenter=fake_segmenter,
        erosion_radius=_EROSION_RADIUS,
    )
    recorded: list[Tensor] = []

    observed = _renovate(
        fake_pipeline,
        room_photo,
        fake_depth_estimator,
        _RELAID_OUT_SEGMENTER,
        strength=_STRENGTH,
        reference=reference,
        on_mask=recorded.append,
    )
    unobserved = _renovate(
        fake_pipeline,
        room_photo,
        fake_depth_estimator,
        _RELAID_OUT_SEGMENTER,
        strength=_STRENGTH,
        reference=reference,
    )

    # One mask per walked step, each the reference shell narrowed by the
    # prediction's disagreeing view of it.
    assert len(recorded) == 2
    assert not any(torch.equal(mask, reference.mask) for mask in recorded)
    # Watching the run must not be part of it.
    assert torch.equal(observed, unobserved)


def test_renovating_lowers_the_masked_depth_error_against_the_photo(
    fake_pipeline: FakePipeline,
    room_photo: Tensor,
    fake_depth_estimator: FakeDepthEstimator,
    fake_segmenter: FakeSegmenter,
) -> None:
    energy = GeometricGuidance(
        depth_estimator=fake_depth_estimator,
        reference=GeometricReference.from_photo(
            room_photo,
            depth_estimator=fake_depth_estimator,
            segmenter=fake_segmenter,
            erosion_radius=_EROSION_RADIUS,
        ),
    )

    unguided = _renovate(
        fake_pipeline, room_photo, fake_depth_estimator, fake_segmenter, strength=0.0
    )
    guided = _renovate(
        fake_pipeline,
        room_photo,
        fake_depth_estimator,
        fake_segmenter,
        strength=_STRENGTH,
    )

    assert float(energy(guided)) < 0.9 * float(energy(unguided))


def test_the_stock_img2img_pipeline_is_callable_the_way_the_loop_calls_it() -> None:
    # `load_pipeline` returns this class, and the loop drives it positionally, so
    # a diffusers release that renames or reorders these parameters breaks the
    # real run while every fake-backed test stays green.
    stock = list(
        inspect.signature(StableDiffusionImg2ImgPipeline.encode_prompt).parameters
    )
    protocol = list(inspect.signature(Img2ImgPipeline.encode_prompt).parameters)

    assert stock[: len(protocol)] == protocol
    # `load_pipeline` refuses the safety checker, whose weights the seeded Hub
    # cache leaves out.
    assert (
        "requires_safety_checker"
        in inspect.signature(StableDiffusionImg2ImgPipeline.__init__).parameters
    )
    assert "safety_checker" in StableDiffusionImg2ImgPipeline._optional_components
