from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch
from jaxtyping import Bool, Float
from torch import Generator, Tensor

from roomgen_fakes import FakeDepthEstimator
from roomgen_geometry.reference import GeometricReference
from roomgen_geometry.reward import masked_depth_error
from roomgen_guide.eval import (
    ClipPromptAdherence,
    depth_error,
    signed_depth_residual_map,
)
from roomgen_guide.eval import clip as clip_module
from roomgen_guide.settings import RoomgenSettings

# What the CLIP ViT-B/32 preprocessor config declares, mirrored here so the
# stubbed tests assert against the same contract the real checkpoint imposes.
_CROP_SIZE = 224
_CLIP_MEAN = (0.481, 0.457, 0.408)
_CLIP_STD = (0.268, 0.261, 0.275)
_PROMPT = "a scandinavian living room"
# Well clear of the fake backend's ~[0, 1.5] disparity resolution, so the sign
# of the residual is the convention under test and not float noise.
_DISPARITY_OFFSET = 0.25


@pytest.fixture
def reference(
    room_photo: Float[Tensor, "1 3 64 64"],
    shell_mask: Bool[Tensor, "1 64 64"],
    fake_depth_estimator: FakeDepthEstimator,
) -> GeometricReference:
    return GeometricReference(
        disparity=fake_depth_estimator(room_photo).detach(), mask=shell_mask
    )


def test_a_photo_scored_against_its_own_geometry_has_no_depth_error(
    room_photo: Float[Tensor, "1 3 64 64"],
    reference: GeometricReference,
    fake_depth_estimator: FakeDepthEstimator,
) -> None:
    score = depth_error(
        room_photo, reference=reference, depth_estimator=fake_depth_estimator
    )

    assert score == pytest.approx(0.0, abs=1e-12)


def test_only_the_shell_counts_towards_the_depth_error(
    room_photo: Float[Tensor, "1 3 64 64"],
    reference: GeometricReference,
    fake_depth_estimator: FakeDepthEstimator,
) -> None:
    outside_the_shell = room_photo.clone()
    outside_the_shell[:, :, -8:, -8:] = 0.0
    inside_the_shell = room_photo.clone()
    inside_the_shell[:, :, :8, :8] = 0.0

    assert depth_error(
        outside_the_shell, reference=reference, depth_estimator=fake_depth_estimator
    ) == pytest.approx(0.0, abs=1e-12)
    assert (
        depth_error(
            inside_the_shell, reference=reference, depth_estimator=fake_depth_estimator
        )
        > 0.0
    )


def test_the_depth_error_is_the_masked_reward_the_guidance_is_steered_by(
    room_photo: Float[Tensor, "1 3 64 64"],
    reference: GeometricReference,
    fake_depth_estimator: FakeDepthEstimator,
    generator: Generator,
) -> None:
    render = torch.rand(room_photo.shape, generator=generator)

    score = depth_error(
        render, reference=reference, depth_estimator=fake_depth_estimator
    )

    expected = masked_depth_error(
        fake_depth_estimator(render), reference.disparity, reference.mask
    )
    assert score == pytest.approx(float(expected.mean()))


def test_the_photos_residual_against_its_own_geometry_is_zero_everywhere(
    room_photo: Float[Tensor, "1 3 64 64"],
    reference: GeometricReference,
    fake_depth_estimator: FakeDepthEstimator,
) -> None:
    residual = signed_depth_residual_map(
        room_photo, reference=reference, depth_estimator=fake_depth_estimator
    )

    assert residual.shape == reference.mask.shape
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-12)


def test_the_residual_map_squares_over_the_shell_to_the_depth_error(
    room_photo: Float[Tensor, "1 3 64 64"],
    reference: GeometricReference,
    fake_depth_estimator: FakeDepthEstimator,
    generator: Generator,
) -> None:
    render = torch.rand(room_photo.shape, generator=generator)

    residual = signed_depth_residual_map(
        render, reference=reference, depth_estimator=fake_depth_estimator
    )

    assert not residual[~reference.mask].any()
    masked_mean = (residual**2).flatten(1).sum(-1) / reference.mask.flatten(1).sum(-1)
    assert float(masked_mean.mean()) == pytest.approx(
        depth_error(render, reference=reference, depth_estimator=fake_depth_estimator)
    )


def test_the_residual_is_positive_where_the_render_reads_closer_than_the_reference(
    room_photo: Float[Tensor, "1 3 64 64"],
    reference: GeometricReference,
    fake_depth_estimator: FakeDepthEstimator,
) -> None:
    # A whole-frame offset is pure shift and the alignment absorbs it, so only
    # part of the reference is pushed away from the camera.
    band = torch.zeros_like(reference.mask)
    band[:, :16, :] = True
    pushed_back = GeometricReference(
        disparity=torch.where(
            band, reference.disparity - _DISPARITY_OFFSET, reference.disparity
        ),
        mask=reference.mask,
    )

    residual = signed_depth_residual_map(
        room_photo, reference=pushed_back, depth_estimator=fake_depth_estimator
    )

    assert float(residual[band & reference.mask].mean()) > 0.0
    assert float(residual[~band & reference.mask].mean()) < 0.0


@dataclass
class _StubClipModel:
    image_features: Tensor
    text_features: Tensor
    pixel_values: list[Tensor] = field(default_factory=list)
    prompts: list[list[str]] = field(default_factory=list)
    frozen: bool = False
    training: bool = True

    def to(self, device: torch.device) -> "_StubClipModel":
        return self

    def eval(self) -> "_StubClipModel":
        self.training = False
        return self

    def requires_grad_(self, requires_grad: bool) -> "_StubClipModel":
        self.frozen = not requires_grad
        return self

    def get_image_features(self, *, pixel_values: Tensor) -> Tensor:
        self.pixel_values.append(pixel_values)
        return self.image_features.expand(pixel_values.shape[0], -1)

    def get_text_features(self, *, input_ids: list[str]) -> Tensor:
        self.prompts.append(input_ids)
        return self.text_features.expand(len(input_ids), -1)


class _StubTokens(dict[str, list[str]]):
    def to(self, device: torch.device) -> "_StubTokens":
        return self


def _stub_clip(
    monkeypatch: pytest.MonkeyPatch, *, image_features: Tensor, text_features: Tensor
) -> _StubClipModel:
    model = _StubClipModel(image_features=image_features, text_features=text_features)
    monkeypatch.setattr(
        clip_module,
        "CLIPModel",
        SimpleNamespace(from_pretrained=lambda checkpoint, cache_dir: model),
    )
    monkeypatch.setattr(
        clip_module,
        "AutoTokenizer",
        SimpleNamespace(
            from_pretrained=lambda checkpoint, cache_dir: (
                lambda prompts, padding, return_tensors: _StubTokens(input_ids=prompts)
            )
        ),
    )
    monkeypatch.setattr(
        clip_module,
        "CLIPImageProcessor",
        SimpleNamespace(
            from_pretrained=lambda checkpoint, cache_dir: SimpleNamespace(
                crop_size={"height": _CROP_SIZE, "width": _CROP_SIZE},
                image_mean=_CLIP_MEAN,
                image_std=_CLIP_STD,
            )
        ),
    )
    return model


def test_the_render_reaches_clip_at_its_own_crop_size_and_normalisation(
    monkeypatch: pytest.MonkeyPatch,
    room_photo: Float[Tensor, "1 3 64 64"],
    reference: GeometricReference,
) -> None:
    stub = _stub_clip(
        monkeypatch,
        image_features=torch.tensor([[1.0, 0.0]]),
        text_features=torch.tensor([[1.0, 0.0]]),
    )
    adherence = ClipPromptAdherence(device=torch.device("cpu"))

    score = adherence(
        torch.full_like(room_photo, 0.5),
        prompt=_PROMPT,
        photo=room_photo,
        reference=reference,
    )

    (pixel_values,) = stub.pixel_values
    assert pixel_values.shape == (1, 3, _CROP_SIZE, _CROP_SIZE)
    normalised_grey = (0.5 - torch.tensor(_CLIP_MEAN)) / torch.tensor(_CLIP_STD)
    assert torch.allclose(
        pixel_values, normalised_grey.view(1, 3, 1, 1).expand_as(pixel_values)
    )
    assert stub.prompts == [[_PROMPT]]
    assert not stub.training
    assert stub.frozen
    assert score == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("text_features", "expected"),
    [((0.5, 0.8660254), 0.5), ((0.0, 1.0), 0.0), ((-1.0, 0.0), 0.0)],
)
def test_adherence_is_the_prompt_cosine_with_an_unrelated_render_floored_at_zero(
    monkeypatch: pytest.MonkeyPatch,
    room_photo: Float[Tensor, "1 3 64 64"],
    reference: GeometricReference,
    text_features: tuple[float, float],
    expected: float,
) -> None:
    _stub_clip(
        monkeypatch,
        image_features=torch.tensor([[1.0, 0.0]]),
        text_features=torch.tensor([text_features]),
    )
    adherence = ClipPromptAdherence(device=torch.device("cpu"))

    score = adherence(room_photo, prompt=_PROMPT, photo=room_photo, reference=reference)

    assert score == pytest.approx(expected, abs=1e-6)


@pytest.mark.models
def test_the_real_checkpoint_prefers_the_prompt_that_describes_the_render(
    room_photo: Float[Tensor, "1 3 64 64"], reference: GeometricReference
) -> None:
    adherence = ClipPromptAdherence(
        device=torch.device("cpu"), cache_dir=RoomgenSettings().hub_cache
    )
    render = torch.zeros_like(room_photo)
    render[:, 0] = 1.0

    described = adherence(
        render, prompt="a solid red image", photo=room_photo, reference=reference
    )
    unrelated = adherence(
        render,
        prompt="a black and white photograph of a snowy forest",
        photo=room_photo,
        reference=reference,
    )

    assert described > unrelated
