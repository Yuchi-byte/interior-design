"""Pydantic configuration base with OmegaConf-backed YAML and dotlist loading."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Self

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict


class Config(BaseModel):
    """Base for every typed experiment config; an unknown key raises instead of a silent no-op."""

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_raw(cls, cfg: DictConfig) -> Self:
        container = OmegaConf.to_container(cfg, resolve=True)
        return cls.model_validate(container)

    @classmethod
    def from_file(cls, path: Path, overrides: Sequence[str] | None = None) -> Self:
        """Load this config from a YAML file, optionally patched with dotlist overrides."""
        cfg = OmegaConf.load(path)
        if not isinstance(cfg, DictConfig):
            raise TypeError(f"Expected a mapping at {path}, got {type(cfg).__name__}")
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
            if not isinstance(cfg, DictConfig):  # pragma: no cover - defensive
                raise TypeError("Merging overrides did not produce a mapping")
        return cls.from_raw(cfg)
