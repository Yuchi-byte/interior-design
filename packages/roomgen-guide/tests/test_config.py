from pathlib import Path

import pytest
from pydantic import ValidationError

from roomgen_guide.config import Config


class _SamplerConfig(Config):
    steps: int
    prompt: str


def test_a_typoed_field_raises_instead_of_being_silently_dropped() -> None:
    with pytest.raises(ValidationError, match="stpes"):
        _SamplerConfig(steps=20, prompt="a bright room", stpes=50)  # type: ignore[call-arg]


def test_a_validated_config_cannot_be_mutated_afterwards() -> None:
    config = _SamplerConfig(steps=20, prompt="a bright room")

    with pytest.raises(ValidationError, match="frozen"):
        config.steps = 50


def test_a_typoed_key_in_a_yaml_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "sampler.yaml"
    path.write_text("steps: 20\nprompt: a bright room\nstpes: 50\n")

    with pytest.raises(ValidationError, match="stpes"):
        _SamplerConfig.from_file(path)


def test_dotlist_overrides_patch_the_loaded_file(tmp_path: Path) -> None:
    path = tmp_path / "sampler.yaml"
    path.write_text("steps: 20\nprompt: a bright room\n")

    config = _SamplerConfig.from_file(path, ["steps=50"])

    assert config.steps == 50
    assert config.prompt == "a bright room"


def test_an_override_of_an_unknown_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "sampler.yaml"
    path.write_text("steps: 20\nprompt: a bright room\n")

    with pytest.raises(ValidationError, match="stpes"):
        _SamplerConfig.from_file(path, ["stpes=50"])


def test_a_yaml_document_that_is_not_a_mapping_raises(tmp_path: Path) -> None:
    path = tmp_path / "sampler.yaml"
    path.write_text("- 20\n- a bright room\n")

    with pytest.raises(TypeError, match="mapping"):
        _SamplerConfig.from_file(path)
