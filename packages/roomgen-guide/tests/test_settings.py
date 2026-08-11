import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from roomgen_guide.settings import RoomgenSettings


@pytest.fixture(autouse=True)
def _without_inherited_roomgen_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [name for name in os.environ if name.startswith("ROOMGEN_")]:
        monkeypatch.delenv(name)


def test_wdir_defaults_to_outputs_at_the_enclosing_repo_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    (root / "uv.lock").touch()
    nested = root / "experiments" / "0001_slug"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    assert RoomgenSettings().wdir == root / "outputs"


def test_roomgen_variables_override_every_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ROOMGEN_WDIR", str(tmp_path / "runs"))
    monkeypatch.setenv("ROOMGEN_WANDB_MODE", "offline")
    monkeypatch.setenv("ROOMGEN_HF_HOME", str(tmp_path / "weights"))

    settings = RoomgenSettings()

    assert settings.wdir == tmp_path / "runs"
    assert settings.wandb_mode == "offline"
    assert settings.hf_home == tmp_path / "weights"


def test_unprefixed_wandb_and_hf_variables_are_not_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "weights"))

    settings = RoomgenSettings()

    assert settings.wandb_mode == "disabled"
    assert settings.hf_home is None


def test_a_setting_outside_the_contract_is_rejected() -> None:
    with pytest.raises(ValidationError, match="checkpoint_dir"):
        RoomgenSettings.model_validate({"checkpoint_dir": "weights"})


def test_wandb_mode_outside_the_documented_set_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROOMGEN_WANDB_MODE", "enabled")
    with pytest.raises(ValidationError, match="wandb_mode"):
        RoomgenSettings()
