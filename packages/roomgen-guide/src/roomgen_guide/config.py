"""Pydantic configuration base with OmegaConf-backed YAML and dotlist loading."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Self

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ConfigTypeError
from pydantic import BaseModel, ConfigDict


class Config(BaseModel):
    """Base for every typed experiment config; an unknown key raises instead of a silent no-op."""

    # Frozen because a config is the record of the run it configured: the
    # snapshot written next to a run's artifacts has to be the values that run
    # actually used, which mutation after validation would silently break.
    model_config = ConfigDict(extra="forbid", frozen=True)

    @classmethod
    def from_file(cls, path: Path, overrides: Sequence[str] | None = None) -> Self:
        """Load this config from a YAML file, optionally patched with dotlist overrides."""
        cfg = OmegaConf.load(path)
        if not isinstance(cfg, DictConfig):
            raise TypeError(f"Expected a mapping at {path}, got {type(cfg).__name__}")
        if overrides:
            try:
                cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
            except ConfigTypeError as error:
                raise ValueError(_merge_failure(overrides, error)) from error
            if not isinstance(cfg, DictConfig):  # pragma: no cover - defensive
                raise TypeError("Merging overrides did not produce a mapping")
        return cls.model_validate(OmegaConf.to_container(cfg, resolve=True))


def _merge_failure(overrides: Sequence[str], error: ConfigTypeError) -> str:
    """What a dotlist that cannot be merged into this file actually hit.

    Raised in place of OmegaConf's own traceback, which reports the internal
    "Cannot merge DictConfig with ListConfig" and never names the override that
    caused it.
    """
    indexed = [
        override
        for override in overrides
        if any(segment.isdigit() for segment in _key(override).split("."))
    ]
    if indexed:
        return (
            f"dotlist overrides cannot address list elements: {', '.join(indexed)}. "
            "A numeric segment is read as a key, so the merge tries to write a "
            "mapping over a list. Override the whole list "
            "(--set scenes='[{photo: a, prompt: b}]') or add a conf/ variant."
        )
    return (
        f"override {', '.join(overrides)} could not be merged into the config: {error}"
    )


def _key(override: str) -> str:
    return override.split("=", 1)[0]
