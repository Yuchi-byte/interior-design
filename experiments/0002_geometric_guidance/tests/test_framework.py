"""The framework's weightless surface: the config bridge and the gallery helpers.

Everything here runs on a `tmp_path` and the shipped `conf/`, so this file
carries no `models` marker and costs milliseconds. The parts that need weights
are covered by running a variant, not by a test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from framework.config import LATENT_ALIGNMENT, BenchmarkConfig
from framework.gallery import GALLERY_FILENAME, write_gallery

CONF_DIR = Path(__file__).resolve().parent.parent / "conf"


@pytest.mark.parametrize(
    "variant", sorted(path.stem for path in CONF_DIR.glob("*.yaml"))
)
def test_every_shipped_variant_validates(variant: str) -> None:
    """A conf/ file that no longer matches the config is a run that dies on startup."""
    config = BenchmarkConfig.from_file(CONF_DIR / f"{variant}.yaml")

    assert config.scenes
    assert config.seeds
    assert config.resolution % LATENT_ALIGNMENT == 0


def test_a_dotlist_override_reaches_a_nested_field() -> None:
    config = BenchmarkConfig.from_file(
        CONF_DIR / "smoke.yaml", ["guidance.strength=12.5", "sampler.steps=3"]
    )

    assert config.guidance.strength == pytest.approx(12.5)
    assert config.sampler.steps == 3


def test_an_override_addressing_a_list_element_says_so() -> None:
    """OmegaConf's own error names neither the override nor the list it hit."""
    with pytest.raises(ValueError, match="cannot address list elements") as raised:
        BenchmarkConfig.from_file(CONF_DIR / "smoke.yaml", ["scenes.0.prompt=a loft"])

    assert "scenes.0.prompt" in str(raised.value)


def test_an_unknown_override_key_is_still_rejected() -> None:
    with pytest.raises(ValueError, match="guidnce"):
        BenchmarkConfig.from_file(CONF_DIR / "smoke.yaml", ["guidnce.strength=1"])


def _results(tmp_path: Path) -> Path:
    results = tmp_path / "results.json"
    results.write_text(
        json.dumps(
            {
                "config": {"name": "0002_smoke", "guidance": {"strength": 500.0}},
                "git_commit": "0123456789abcdef",
                "runs": [
                    {
                        "scene": "scandi_living_room",
                        "prompt": "a scandinavian living room",
                        "seed": seed,
                        "guided_depth_error": 0.01,
                        "control_depth_error": 0.02,
                        "depth_error_reduction": 0.5,
                        "adherence_ratio": 0.9,
                        "common_coverage": 0.42,
                    }
                    for seed in (0, 1)
                ],
            }
        ),
        encoding="utf-8",
    )
    return results


def test_the_gallery_renders_from_the_numbers_a_run_recorded(tmp_path: Path) -> None:
    """No images beside the results: every figure is skipped, the page still stands."""
    page = write_gallery(_results(tmp_path))

    assert page == tmp_path / GALLERY_FILENAME
    rendered = page.read_text(encoding="ascii")
    assert "0002_smoke" in rendered
    assert "Scandi living room" in rendered
    # The numbers are the run's, read off the file rather than recomputed.
    assert "&minus;50%" in rendered
    assert "42%" in rendered
    assert "01234567" in rendered
    assert "<img" not in rendered


def test_the_gallery_refuses_a_results_file_with_no_runs(tmp_path: Path) -> None:
    empty = tmp_path / "results.json"
    empty.write_text(json.dumps({"config": {"name": "0002"}, "runs": []}), "utf-8")

    with pytest.raises(ValueError, match="no run entries"):
        write_gallery(empty)
