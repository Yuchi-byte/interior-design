"""The repo's whole environment contract: ``ROOMGEN_*`` and nothing else."""

from pathlib import Path
from typing import Final, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

WandbMode = Literal["online", "offline", "disabled"]

# Files that exist only at the workspace root, so the nearest one at or above
# the working directory identifies it without any path being written down.
# `.git` is a directory in a clone and a file in a worktree, hence existence
# rather than type is the test.
_REPO_ROOT_MARKERS: Final = ("uv.lock", ".git")


def _repo_root() -> Path:
    working_directory = Path.cwd().resolve()
    for candidate in (working_directory, *working_directory.parents):
        if any((candidate / marker).exists() for marker in _REPO_ROOT_MARKERS):
            return candidate
    return working_directory


class RoomgenSettings(BaseSettings):
    """Where a run's artifacts, weights and tracking go, read from ``ROOMGEN_*`` alone.

    Nothing in the repo reads a working directory, cache location or wandb mode
    from anywhere else; plain ``WANDB_MODE`` and ``HF_HOME`` are not consulted.

    Args:
        wdir: root of every local artifact a run writes; defaults to
            ``outputs/`` at the enclosing repo root, which is gitignored.
        hf_home: pretrained-weight cache, pinned under the repo root by default
            so tens of GB of weights land in one gitignored place rather than
            wherever the Hub client would otherwise put them.
    """

    # `extra="forbid"` extends the config rule to the environment: a misspelled
    # `ROOMGEN_*` variable is a startup error rather than a setting that
    # silently never took effect.
    model_config = SettingsConfigDict(
        env_prefix="ROOMGEN_", extra="forbid", frozen=True
    )

    wdir: Path = Field(default_factory=lambda: _repo_root() / "outputs")
    wandb_mode: WandbMode = "disabled"
    hf_home: Path = Field(
        default_factory=lambda: _repo_root() / ".cache" / "huggingface"
    )
