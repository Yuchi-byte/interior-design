import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from roomgen_guide.settings import RoomgenSettings


@pytest.fixture(autouse=True)
def _without_inherited_roomgen_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [name for name in os.environ if name.startswith("ROOMGEN_")]:
        monkeypatch.delenv(name)


def test_defaults_sit_at_the_enclosing_repo_root_not_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    (root / "uv.lock").touch()
    nested = root / "experiments" / "0001_slug"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    settings = RoomgenSettings()

    assert settings.wdir == root / "outputs"
    assert settings.hf_home == root / ".cache" / "huggingface"
    assert settings.hub_cache == root / ".cache" / "huggingface" / "hub"


def test_roomgen_variables_override_every_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ROOMGEN_WDIR", str(tmp_path / "runs"))
    monkeypatch.setenv("ROOMGEN_HF_HOME", str(tmp_path / "weights"))

    settings = RoomgenSettings()

    assert settings.wdir == tmp_path / "runs"
    assert settings.hf_home == tmp_path / "weights"


def test_the_environment_alone_is_enough_outside_any_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both variables set is a complete configuration: no repo root is consulted."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setenv("ROOMGEN_WDIR", str(tmp_path / "runs"))
    monkeypatch.setenv("ROOMGEN_HF_HOME", str(tmp_path / "weights"))

    settings = RoomgenSettings()

    assert settings.wdir == tmp_path / "runs"
    assert settings.hub_cache == tmp_path / "weights" / "hub"


def test_a_default_outside_any_repo_raises_naming_its_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The alternative is a 10GB weight cache seeded under whatever `cd` was last run."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    with pytest.raises(RuntimeError, match="ROOMGEN_WDIR"):
        RoomgenSettings()

    monkeypatch.setenv("ROOMGEN_WDIR", str(tmp_path / "runs"))
    with pytest.raises(RuntimeError, match="ROOMGEN_HF_HOME"):
        RoomgenSettings()


def test_unprefixed_hf_variables_are_not_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    (root / "uv.lock").touch()
    monkeypatch.chdir(root)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "weights"))

    settings = RoomgenSettings()

    assert settings.hf_home == root / ".cache" / "huggingface"


def test_a_setting_outside_the_contract_is_rejected() -> None:
    with pytest.raises(ValidationError, match="checkpoint_dir"):
        RoomgenSettings.model_validate({"checkpoint_dir": "weights"})
