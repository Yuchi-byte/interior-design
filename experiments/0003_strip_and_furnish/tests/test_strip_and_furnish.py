"""The framework's weightless surface: the config guards and the two galleries' helpers.

Everything here runs on a `tmp_path` and the shipped `conf/`, so this file
carries no `models` marker and costs milliseconds. The parts that need weights
are covered by running a variant, not by a test.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from strip_and_furnish.config import LATENT_ALIGNMENT, StudyConfig
from strip_and_furnish.images import (
    DisparityScale,
    write_disparity_map,
    write_image,
    write_mask_overlay,
)
from torchvision.io import ImageReadMode, read_image

CONF_DIR = Path(__file__).resolve().parent.parent / "conf"


@pytest.mark.parametrize(
    "variant", sorted(path.stem for path in CONF_DIR.glob("*.yaml"))
)
def test_every_shipped_variant_validates(variant: str) -> None:
    """A conf/ file that no longer matches the config is a run that dies on startup."""
    config = StudyConfig.from_file(CONF_DIR / f"{variant}.yaml")

    assert config.scenes
    assert config.seeds
    assert config.resolution % LATENT_ALIGNMENT == 0


def test_scoring_adherence_with_the_guided_checkpoint_is_refused() -> None:
    # Arm A descends this very checkpoint's cosine, so scoring adherence with it
    # would report how well the sampler hit its own target.
    with pytest.raises(ValueError, match="graded on the model it optimised"):
        StudyConfig.from_file(
            CONF_DIR / "study.yaml",
            ["adherence.checkpoint=openai/clip-vit-base-patch32"],
        )


def test_a_dotlist_override_reaches_a_nested_field() -> None:
    config = StudyConfig.from_file(
        CONF_DIR / "smoke.yaml", ["furnish.strength=12.5", "strip.dilation_radius=5"]
    )

    assert config.furnish.strength == pytest.approx(12.5)
    assert config.strip.dilation_radius == 5


def test_the_overlay_tints_the_masked_region_and_leaves_the_rest_of_the_photo(
    tmp_path: Path,
) -> None:
    photo = torch.full((1, 3, 16, 16), 0.5)
    mask = torch.zeros((1, 16, 16), dtype=torch.bool)
    mask[:, 4:8, 4:8] = True
    path = tmp_path / "overlay.png"

    write_mask_overlay(photo, mask, path)

    overlay = read_image(str(path), mode=ImageReadMode.RGB)
    assert not torch.equal(overlay[:, 4:8, 4:8], overlay[:, 12:16, 12:16])
    assert bool((overlay[:, 12:16, 12:16] == 128).all())


def test_two_depth_maps_of_the_same_room_are_painted_the_same_way(
    tmp_path: Path,
) -> None:
    # The point of fitting before painting: an affine-rescaled prediction is the
    # same geometry, and a gallery that paints it differently is reporting the
    # backend's arbitrary scale as a change in the room.
    disparity = torch.linspace(0.0, 1.0, 256).reshape(1, 16, 16)
    mask = torch.ones((1, 16, 16), dtype=torch.bool)
    scale = DisparityScale.of(disparity)
    paths = (tmp_path / "one.png", tmp_path / "two.png")

    write_disparity_map(disparity, scale=scale, mask=mask, path=paths[0])
    write_disparity_map(3.0 * disparity - 2.0, scale=scale, mask=mask, path=paths[1])

    painted = [
        read_image(str(path), mode=ImageReadMode.RGB).to(torch.int16) for path in paths
    ]
    # Within a quantisation step: the fit reproduces the map to float precision,
    # and 8-bit rounding puts a handful of pixels either side of a boundary.
    assert int((painted[0] - painted[1]).abs().max()) <= 1


def test_a_render_survives_the_round_trip_through_a_png(tmp_path: Path) -> None:
    render = torch.rand((1, 3, 16, 16), generator=torch.Generator().manual_seed(0))
    path = tmp_path / "render.png"

    write_image(render, path)

    decoded = read_image(str(path), mode=ImageReadMode.RGB).to(torch.float32) / 255.0
    torch.testing.assert_close(decoded.unsqueeze(0), render, atol=1 / 255, rtol=0)
