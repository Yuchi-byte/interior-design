from pathlib import Path

import pytest
from pydantic import ValidationError

from roomgen_guide.settings import RoomgenSettings


def test_outputs_default_under_the_repo_root_not_the_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = RoomgenSettings()

    assert settings.wdir.is_absolute()
    assert settings.wdir.name == "outputs"
    assert settings.hf_home == settings.wdir.parent / ".cache" / "huggingface"


def test_environment_overrides_every_setting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ROOMGEN_WDIR", str(tmp_path / "runs"))
    monkeypatch.setenv("ROOMGEN_WANDB_MODE", "offline")
    monkeypatch.setenv("ROOMGEN_HF_HOME", str(tmp_path / "weights"))

    settings = RoomgenSettings()

    assert settings.wdir == tmp_path / "runs"
    assert settings.wandb_mode == "offline"
    assert settings.hf_home == tmp_path / "weights"


def test_unprefixed_wandb_mode_is_not_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_MODE", "disabled")

    assert RoomgenSettings().wandb_mode == "online"


def test_an_unknown_tracking_mode_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROOMGEN_WANDB_MODE", "ofline")

    with pytest.raises(ValidationError):
        RoomgenSettings()
