from pathlib import Path

import pytest
from pydantic import ValidationError

from roomgen_guide.config import Config


class _SamplerConfig(Config):
    steps: int
    strength: float = 0.5


def test_unknown_key_raises_rather_than_being_dropped() -> None:
    with pytest.raises(ValidationError, match="strenght"):
        _SamplerConfig.model_validate({"steps": 30, "strenght": 0.4})


def test_validated_config_cannot_be_mutated() -> None:
    config = _SamplerConfig(steps=30)
    with pytest.raises(ValidationError):
        config.steps = 40


def test_dotlist_override_patches_a_loaded_yaml(tmp_path: Path) -> None:
    path = tmp_path / "sampler.yaml"
    path.write_text("steps: 30\nstrength: 0.5\n")
    config = _SamplerConfig.from_file(path, overrides=["strength=0.8"])
    assert (config.steps, config.strength) == (30, 0.8)


def test_unknown_key_from_a_dotlist_override_raises(tmp_path: Path) -> None:
    path = tmp_path / "sampler.yaml"
    path.write_text("steps: 30\n")
    with pytest.raises(ValidationError, match="strenght"):
        _SamplerConfig.from_file(path, overrides=["strenght=0.8"])
