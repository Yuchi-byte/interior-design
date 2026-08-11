from pathlib import Path

import pytest
from pydantic import ValidationError

from roomgen_guide.experiment import ExperimentConfig, VerdictGateConfig

_GATE = {"depth_error_margin": 0.2, "prompt_adherence_floor": 0.9}


def test_a_framework_config_loads_its_gate_as_a_typed_nested_config(
    tmp_path: Path,
) -> None:
    path = tmp_path / "baseline.yaml"
    path.write_text(
        "name: 0001_unguided_baseline\n"
        "seeds: [0, 1, 2]\n"
        "gate:\n"
        "  depth_error_margin: 0.2\n"
        "  prompt_adherence_floor: 0.9\n"
    )

    config = ExperimentConfig.from_file(path, overrides=["gate.depth_error_margin=0.3"])

    assert config.seeds == (0, 1, 2)
    assert config.gate == VerdictGateConfig(
        depth_error_margin=0.3, prompt_adherence_floor=0.9
    )


def test_a_framework_without_an_explicit_seed_is_rejected() -> None:
    with pytest.raises(ValidationError, match="seeds"):
        ExperimentConfig.model_validate(
            {"name": "0001_slug", "seeds": [], "gate": _GATE}
        )


def test_a_gate_that_demands_no_improvement_is_rejected() -> None:
    with pytest.raises(ValidationError, match="depth_error_margin"):
        VerdictGateConfig(depth_error_margin=0.0, prompt_adherence_floor=0.9)
