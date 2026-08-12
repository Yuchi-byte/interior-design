"""The repo's whole environment contract: ``ROOMGEN_*`` and nothing else."""

from pathlib import Path
from typing import Final

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Files that exist only at the workspace root, so the nearest one at or above
# the working directory identifies it without any path being written down.
# `.git` is a directory in a clone and a file in a worktree, hence existence
# rather than type is the test.
_REPO_ROOT_MARKERS: Final = ("uv.lock", ".git")


def _repo_root(*, setting: str) -> Path:
    """The enclosing repo root, or an error naming the variable that replaces it.

    Only reached for a setting the environment did not supply. Falling back to
    the working directory instead would silently re-root a cache: a
    `roomgen-prefetch` run from anywhere outside the repo would quietly seed
    tens of GB under wherever it happened to be started.
    """
    working_directory = Path.cwd().resolve()
    for candidate in (working_directory, *working_directory.parents):
        if any((candidate / marker).exists() for marker in _REPO_ROOT_MARKERS):
            return candidate
    raise RuntimeError(
        f"{working_directory} is not inside the roomgen repo (no "
        f"{' or '.join(_REPO_ROOT_MARKERS)} at or above it), so `{setting}` has no "
        f"default. Set {setting} to an explicit path, or run from inside a clone."
    )


class RoomgenSettings(BaseSettings):
    """Where a run's artifacts and weights go, read from ``ROOMGEN_*`` alone.

    Nothing in the repo reads a working directory or cache location from
    anywhere else; plain ``HF_HOME`` is not consulted. Both defaults are
    relative to the enclosing repo root, so a process started outside a clone
    must name the one it uses rather than inherit a directory by accident.

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

    wdir: Path = Field(
        default_factory=lambda: _repo_root(setting="ROOMGEN_WDIR") / "outputs"
    )
    hf_home: Path = Field(
        default_factory=lambda: (
            _repo_root(setting="ROOMGEN_HF_HOME") / ".cache" / "huggingface"
        )
    )

    @property
    def hub_cache(self) -> Path:
        """The Hub snapshot cache under `hf_home`, matching `HF_HOME`'s layout."""
        return self.hf_home / "hub"
