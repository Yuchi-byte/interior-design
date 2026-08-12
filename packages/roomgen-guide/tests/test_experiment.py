from pathlib import Path

import pytest
from pydantic import ValidationError

from roomgen_guide.experiment import ExperimentConfig


def test_a_framework_config_loads_its_seeds_from_yaml_with_overrides(
    tmp_path: Path,
) -> None:
    path = tmp_path / "baseline.yaml"
    path.write_text("name: 0002_geometric_guidance\nseeds: [0, 1, 2]\n")

    config = ExperimentConfig.from_file(path, overrides=["name=0001_renamed"])

    assert config.seeds == (0, 1, 2)
    assert config.name == "0001_renamed"


def test_a_framework_without_an_explicit_seed_is_rejected() -> None:
    with pytest.raises(ValidationError, match="seeds"):
        ExperimentConfig.model_validate({"name": "0001_slug", "seeds": []})


def test_an_unknown_field_is_a_validation_error_rather_than_a_silent_no_op() -> None:
    with pytest.raises(ValidationError, match="gate"):
        ExperimentConfig.model_validate(
            {"name": "0001_slug", "seeds": [0], "gate": {"depth_error_margin": 0.2}}
        )
