"""The config shape every experiment framework builds from its `conf/`."""

from typing import Annotated

from pydantic import Field

from roomgen_guide.config import Config


class VerdictGateConfig(Config):
    """The two thresholds a framework's verdict is gated on, calibrated rather than picked.

    Both are expressed relative to the unguided control at the same seed and
    prompt, because masked depth error is in arbitrary disparity units and an
    absolute threshold would not carry from one scene to the next.

    Args:
        depth_error_margin: fraction by which guided masked depth error must
            fall below the control's for the run to count as a win.
        prompt_adherence_floor: fraction of the control's prompt adherence the
            guided run must still reach, which is what stops a sampler from
            winning on geometry by declining to renovate.
    """

    depth_error_margin: Annotated[float, Field(gt=0.0, lt=1.0)]
    prompt_adherence_floor: Annotated[float, Field(gt=0.0, le=1.0)]


class ExperimentConfig(Config):
    """Base config of an experiment framework; a framework extends it with what it varies.

    Args:
        name: identifies this config wherever a run is recorded — its directory
            under `RoomgenSettings.wdir` and its tracking run.
        seeds: every seed the framework runs, listed rather than drawn, and the
            gate must hold on each of them.
    """

    name: Annotated[str, Field(min_length=1)]
    seeds: Annotated[tuple[int, ...], Field(min_length=1)]
    gate: VerdictGateConfig
