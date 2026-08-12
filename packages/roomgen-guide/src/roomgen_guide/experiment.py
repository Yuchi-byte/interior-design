"""The config shape every experiment framework builds from its `conf/`."""

from typing import Annotated

from pydantic import Field

from roomgen_guide.config import Config


class ExperimentConfig(Config):
    """Base config of an experiment framework; a framework extends it with what it varies.

    Args:
        name: identifies this config wherever a run is recorded — its directory
            under `RoomgenSettings.wdir` and its tracking run.
        seeds: every seed the framework runs, listed rather than drawn, so a
            reported number can be reproduced exactly.
    """

    name: Annotated[str, Field(min_length=1)]
    seeds: Annotated[tuple[int, ...], Field(min_length=1)]
