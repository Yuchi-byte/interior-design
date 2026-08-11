"""The one ``ROOMGEN_*`` environment contract in the repo."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

WandbMode = Literal["online", "offline", "disabled"]

# Defaults are repo-relative, not cwd-relative, so a run launched from anywhere
# writes to the same gitignored directories. The workspace is installed
# editable, so this file sits at <repo>/packages/roomgen-guide/src/roomgen_guide.
_REPO_ROOT: Final = Path(__file__).resolve().parents[4]


class RoomgenSettings(BaseSettings):
    """Machine-local paths and tracking mode, overridden per machine by ``ROOMGEN_*`` env vars.

    Nothing in the repo reads a working directory, cache location or wandb mode
    from anywhere else; plain ``WANDB_MODE`` and ``HF_HOME`` are not consulted.
    """

    model_config = SettingsConfigDict(
        env_prefix="ROOMGEN_", extra="forbid", frozen=True
    )

    wdir: Path = _REPO_ROOT / "outputs"
    wandb_mode: WandbMode = "online"
    hf_home: Path = _REPO_ROOT / ".cache" / "huggingface"
