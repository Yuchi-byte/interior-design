import os
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import Generator, Tensor
from torch.nn import Module, Parameter

from roomgen_fakes import assert_gradient_is_finite_and_nonzero
from roomgen_geometry.depth import depth_anything
from roomgen_geometry.depth.depth_anything import DEFAULT_CHECKPOINT, DepthAnythingV2

# What the DA2-hf preprocessor config declares, mirrored here so the stubbed
# tests assert against the same contract the real checkpoint imposes.
_INPUT_SIZE = 518
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
# Big enough for the synthetic room's perspective cues to survive the
# checkpoint's own resize.
_ROOM_SIZE = 256


@dataclass(frozen=True)
class _StubImageProcessor:
    size: dict[str, int]
    image_mean: tuple[float, ...]
    image_std: tuple[float, ...]


@dataclass(frozen=True)
class _StubDepthOutput:
    predicted_depth: Tensor


class _RecordingDepthModel(Module):
    def __init__(self) -> None:
        super().__init__()
        self.gain = Parameter(torch.tensor(2.0))
        self.processor_loads: list[tuple[str, Path | None]] = []
        self.model_loads: list[tuple[str, Path | None, torch.dtype]] = []
        self.pixel_values: list[Tensor] = []
        self.was_training: list[bool] = []
        self.was_frozen: list[bool] = []

    def forward(self, pixel_values: Tensor) -> _StubDepthOutput:
        self.pixel_values.append(pixel_values)
        self.was_training.append(self.training)
        self.was_frozen.append(not self.gain.requires_grad)
        # Half the checkpoint's resolution, so only the backend's own upsample
        # can bring the disparity back to the caller's frame.
        return _StubDepthOutput(self.gain * pixel_values.mean(dim=1)[:, ::2, ::2])


@dataclass
class _StubHub:
    model: _RecordingDepthModel = field(default_factory=_RecordingDepthModel)

    def load_processor(
        self, checkpoint: str, *, cache_dir: Path | None
    ) -> _StubImageProcessor:
        self.model.processor_loads.append((checkpoint, cache_dir))
        return _StubImageProcessor(
            size={"height": _INPUT_SIZE, "width": _INPUT_SIZE},
            image_mean=_IMAGENET_MEAN,
            image_std=_IMAGENET_STD,
        )

    def load_model(
        self, checkpoint: str, *, torch_dtype: torch.dtype, cache_dir: Path | None
    ) -> _RecordingDepthModel:
        self.model.model_loads.append((checkpoint, cache_dir, torch_dtype))
        return self.model


@pytest.fixture
def stub_hub(monkeypatch: pytest.MonkeyPatch) -> _StubHub:
    hub = _StubHub()
    monkeypatch.setattr(
        depth_anything,
        "DPTImageProcessor",
        SimpleNamespace(from_pretrained=hub.load_processor),
    )
    monkeypatch.setattr(
        depth_anything,
        "AutoModelForDepthEstimation",
        SimpleNamespace(from_pretrained=hub.load_model),
    )
    return hub


def test_both_halves_of_the_checkpoint_load_from_the_configured_hub_cache(
    stub_hub: _StubHub, tmp_path: Path
) -> None:
    DepthAnythingV2(device=torch.device("cpu"), cache_dir=tmp_path)

    assert stub_hub.model.processor_loads == [(DEFAULT_CHECKPOINT, tmp_path)]
    assert stub_hub.model.model_loads == [(DEFAULT_CHECKPOINT, tmp_path, torch.float32)]


def test_the_image_reaches_the_checkpoint_at_its_own_size_and_normalisation(
    stub_hub: _StubHub, tmp_path: Path
) -> None:
    estimator = DepthAnythingV2(device=torch.device("cpu"), cache_dir=tmp_path)

    estimator(torch.full((1, 3, 40, 64), 0.5))

    (pixel_values,) = stub_hub.model.pixel_values
    assert pixel_values.shape == (1, 3, _INPUT_SIZE, _INPUT_SIZE)
    normalised_grey = (0.5 - torch.tensor(_IMAGENET_MEAN)) / torch.tensor(_IMAGENET_STD)
    assert torch.allclose(
        pixel_values, normalised_grey.view(1, 3, 1, 1).expand_as(pixel_values)
    )


def test_the_disparity_comes_back_at_the_input_resolution_still_on_the_graph(
    stub_hub: _StubHub, tmp_path: Path, generator: Generator
) -> None:
    estimator = DepthAnythingV2(device=torch.device("cpu"), cache_dir=tmp_path)
    images = torch.rand((2, 3, 40, 64), generator=generator, requires_grad=True)

    disparity = estimator(images)

    assert disparity.shape == (2, 40, 64)
    gradient = assert_gradient_is_finite_and_nonzero(disparity.mean(), images)
    assert gradient.shape == images.shape


def test_the_backbone_runs_frozen_and_in_eval_mode(
    stub_hub: _StubHub, tmp_path: Path, generator: Generator
) -> None:
    estimator = DepthAnythingV2(device=torch.device("cpu"), cache_dir=tmp_path)

    estimator(torch.rand((1, 3, 32, 32), generator=generator))

    assert stub_hub.model.was_training == [False]
    assert stub_hub.model.was_frozen == [True]


def _hub_cache() -> Path | None:
    # `RoomgenSettings` lives in the guide package, which sits above this one,
    # so the models tests read the one variable that locates the cache directly.
    hf_home = os.environ.get("ROOMGEN_HF_HOME")
    return Path(hf_home) / "hub" if hf_home else None


@pytest.fixture(scope="module")
def depth_anything_v2() -> DepthAnythingV2:
    return DepthAnythingV2(device=torch.device("cpu"), cache_dir=_hub_cache())


@pytest.mark.models
def test_the_real_checkpoint_steers_the_image_it_was_given(
    depth_anything_v2: DepthAnythingV2, generator: Generator
) -> None:
    images = torch.rand((1, 3, 64, 64), generator=generator, requires_grad=True)

    disparity = depth_anything_v2(images)

    assert disparity.shape == (1, 64, 64)
    gradient = assert_gradient_is_finite_and_nonzero(disparity.mean(), images)
    assert gradient.shape == images.shape


@pytest.mark.models
def test_the_real_checkpoint_repeats_itself_on_one_image(
    depth_anything_v2: DepthAnythingV2, generator: Generator
) -> None:
    images = torch.rand((1, 3, 64, 64), generator=generator)

    assert torch.equal(depth_anything_v2(images), depth_anything_v2(images))


def _perspective_room() -> Tensor:
    """A one-point perspective room: floor, ceiling and side walls at a central horizon."""
    rows = torch.linspace(-0.5, 0.5, _ROOM_SIZE)
    columns = torch.linspace(-1.0, 1.0, _ROOM_SIZE)
    height, width = torch.meshgrid(rows, columns, indexing="ij")
    faces_the_camera = height.abs() > width.abs()
    is_floor = (height > 0.0) & faces_the_camera
    image = torch.full((1, 3, _ROOM_SIZE, _ROOM_SIZE), 0.55)
    image[0, :, is_floor] = 0.30
    image[0, :, (height < 0.0) & faces_the_camera] = 0.80
    # Floorboards crowding towards the horizon, the cue that makes the floor
    # recede rather than merely be darker.
    receding = (0.5 / height.clamp(min=1e-3)).frac()
    image[0, :, is_floor & (receding < 0.08)] = 0.15
    return image


@pytest.mark.models
def test_the_real_checkpoint_reads_the_floor_as_nearer_than_the_ceiling(
    depth_anything_v2: DepthAnythingV2,
) -> None:
    # The backend returns *inverse* depth: near is large. Every sign downstream
    # rests on that — `signed_depth_residual_map` calls a positive residual "the
    # surface reads closer than it was" — and a backend swap returning depth
    # instead would negate all of it while every other test still passed. On a
    # room whose floor recedes to a central horizon the convention is visible in
    # the frame itself: the near floor across the bottom must read larger than
    # the far ceiling across the top (measured margin ~0.9, on an upper third of
    # ~2.5).
    disparity = depth_anything_v2(_perspective_room())

    third = _ROOM_SIZE // 3
    assert float(disparity[:, -third:].mean()) > float(disparity[:, :third].mean())
