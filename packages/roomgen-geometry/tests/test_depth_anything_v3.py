import os
from pathlib import Path

import pytest
import torch
from torch import Generator, Tensor
from torch.nn import Module, Parameter

from roomgen_fakes import assert_gradient_is_finite_and_nonzero
from roomgen_geometry.depth import depth_anything_v3
from roomgen_geometry.depth.depth_anything_v3 import (
    DEFAULT_CHECKPOINT,
    DEFAULT_INPUT_SIZE,
    DepthAnythingV3Metric,
)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
# Big enough for the synthetic room's perspective cues to survive the
# checkpoint's own resize.
_ROOM_SIZE = 256


class _RecordingBackbone(Module):
    def __init__(self) -> None:
        super().__init__()
        self.gain = Parameter(torch.tensor(2.0))
        self.pixel_values: list[Tensor] = []
        self.was_training: list[bool] = []
        self.was_frozen: list[bool] = []

    def forward(
        self, pixel_values: Tensor, *, cam_token: None, export_feat_layers: list[int]
    ) -> tuple[tuple[Tensor, ...], list[Tensor]]:
        assert cam_token is None
        assert export_feat_layers == []
        self.pixel_values.append(pixel_values)
        self.was_training.append(self.training)
        self.was_frozen.append(not self.gain.requires_grad)
        return (self.gain * pixel_values,), []


class _RecordingHead(Module):
    def __init__(self) -> None:
        super().__init__()
        self.sizes: list[tuple[int, int]] = []

    def forward(
        self,
        features: tuple[Tensor, ...],
        height: int,
        width: int,
        *,
        patch_start_idx: int,
    ) -> dict[str, Tensor]:
        assert patch_start_idx == 0
        self.sizes.append((height, width))
        (activations,) = features
        # *Depth*, positive and rising with the pixel value, at half the
        # backbone's resolution: only the backend's own reciprocal can turn this
        # into a disparity, and only its upsample can return it to the caller's
        # frame. `exp` mirrors the checkpoint's own head activation.
        return {"depth": activations.mean(dim=2)[..., ::2, ::2].exp()}


class _StubNetwork(Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _RecordingBackbone()
        self.head = _RecordingHead()


class _StubHub:
    def __init__(self) -> None:
        self.network = _StubNetwork()
        self.loads: list[tuple[str, Path | None]] = []

    def load_network(self, checkpoint: str, *, cache_dir: Path | None) -> Module:
        self.loads.append((checkpoint, cache_dir))
        return self.network


@pytest.fixture
def stub_hub(monkeypatch: pytest.MonkeyPatch) -> _StubHub:
    hub = _StubHub()
    monkeypatch.setattr(depth_anything_v3, "load_network", hub.load_network)
    return hub


def test_the_checkpoint_loads_from_the_configured_hub_cache(
    stub_hub: _StubHub, tmp_path: Path
) -> None:
    DepthAnythingV3Metric(device=torch.device("cpu"), cache_dir=tmp_path)

    assert stub_hub.loads == [(DEFAULT_CHECKPOINT, tmp_path)]


def test_the_image_reaches_the_network_at_its_own_size_and_normalisation(
    stub_hub: _StubHub, tmp_path: Path
) -> None:
    estimator = DepthAnythingV3Metric(device=torch.device("cpu"), cache_dir=tmp_path)

    estimator(torch.full((1, 3, 40, 64), 0.5))

    (pixel_values,) = stub_hub.network.backbone.pixel_values
    # The view axis the multi-view backbone indexes, with one photograph in it.
    assert pixel_values.shape == (1, 1, 3, DEFAULT_INPUT_SIZE, DEFAULT_INPUT_SIZE)
    normalised_grey = (0.5 - torch.tensor(_IMAGENET_MEAN)) / torch.tensor(_IMAGENET_STD)
    assert torch.allclose(
        pixel_values, normalised_grey.view(1, 1, 3, 1, 1).expand_as(pixel_values)
    )
    assert stub_hub.network.head.sizes == [(DEFAULT_INPUT_SIZE, DEFAULT_INPUT_SIZE)]


def test_the_disparity_comes_back_at_the_input_resolution_still_on_the_graph(
    stub_hub: _StubHub, tmp_path: Path, generator: Generator
) -> None:
    estimator = DepthAnythingV3Metric(device=torch.device("cpu"), cache_dir=tmp_path)
    images = torch.rand((2, 3, 40, 64), generator=generator, requires_grad=True)

    disparity = estimator(images)

    assert disparity.shape == (2, 40, 64)
    gradient = assert_gradient_is_finite_and_nonzero(disparity.mean(), images)
    assert gradient.shape == images.shape


def test_the_metric_depth_is_handed_back_as_its_reciprocal(
    stub_hub: _StubHub, tmp_path: Path
) -> None:
    # The network predicts depth and the package speaks disparity, so the whole
    # sign convention downstream rests on this one reciprocal. A uniform image
    # makes the head's output uniform too, so the expected value is closed form
    # and survives the resize either side of it.
    estimator = DepthAnythingV3Metric(device=torch.device("cpu"), cache_dir=tmp_path)
    grey = 0.5
    channel_mean = (
        sum(
            (grey - mean) / std
            for mean, std in zip(_IMAGENET_MEAN, _IMAGENET_STD, strict=True)
        )
        / 3.0
    )

    disparity = estimator(torch.full((1, 3, 32, 32), grey))

    expected_depth = torch.tensor(2.0 * channel_mean).exp()
    assert torch.allclose(disparity, 1.0 / expected_depth.expand_as(disparity))


def test_the_network_runs_frozen_and_in_eval_mode(
    stub_hub: _StubHub, tmp_path: Path, generator: Generator
) -> None:
    estimator = DepthAnythingV3Metric(device=torch.device("cpu"), cache_dir=tmp_path)

    estimator(torch.rand((1, 3, 32, 32), generator=generator))

    assert stub_hub.network.backbone.was_training == [False]
    assert stub_hub.network.backbone.was_frozen == [True]


def test_an_input_size_the_patch_grid_cannot_tile_is_rejected(
    stub_hub: _StubHub, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="multiple of the patch size 14"):
        DepthAnythingV3Metric(
            device=torch.device("cpu"), cache_dir=tmp_path, input_size=512
        )


def _hub_cache() -> Path | None:
    # `RoomgenSettings` lives in the guide package, which sits above this one,
    # so the models tests read the one variable that locates the cache directly.
    hf_home = os.environ.get("ROOMGEN_HF_HOME")
    return Path(hf_home) / "hub" if hf_home else None


@pytest.fixture(scope="module")
def metric_estimator() -> DepthAnythingV3Metric:
    pytest.importorskip(
        "depth_anything_3",
        reason="upstream `depth_anything_3` is installed out of band; see the backend",
    )
    return DepthAnythingV3Metric(device=torch.device("cpu"), cache_dir=_hub_cache())


@pytest.mark.models
def test_the_real_checkpoint_steers_the_image_it_was_given(
    metric_estimator: DepthAnythingV3Metric, generator: Generator
) -> None:
    images = torch.rand((1, 3, 64, 64), generator=generator, requires_grad=True)

    disparity = metric_estimator(images)

    assert disparity.shape == (1, 64, 64)
    gradient = assert_gradient_is_finite_and_nonzero(disparity.mean(), images)
    assert gradient.shape == images.shape


@pytest.mark.models
def test_the_real_checkpoint_repeats_itself_on_one_image(
    metric_estimator: DepthAnythingV3Metric, generator: Generator
) -> None:
    images = torch.rand((1, 3, 64, 64), generator=generator)

    assert torch.equal(metric_estimator(images), metric_estimator(images))


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
    receding = (0.5 / height.clamp(min=1e-3)).frac()
    image[0, :, is_floor & (receding < 0.08)] = 0.15
    return image


@pytest.mark.models
def test_the_real_checkpoint_reads_the_floor_as_nearer_than_the_ceiling(
    metric_estimator: DepthAnythingV3Metric,
) -> None:
    # The polarity check that the stubbed reciprocal test cannot make: the real
    # head predicts metres, and a backend that forwarded them unconverted would
    # negate every sign downstream while every other test still passed.
    disparity = metric_estimator(_perspective_room())

    third = _ROOM_SIZE // 3
    assert float(disparity[:, -third:].mean()) > float(disparity[:, :third].mean())
