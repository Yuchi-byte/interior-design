"""The backend-agnostic shape of reward-steered sampling.

A backend is an :class:`~roomgen_guide.process.endpoint.EndpointProcess`, a
reward is a :class:`~roomgen_guide.guidance.base.Guidance` on the endpoint
image, and how one becomes a step is a
:class:`~roomgen_guide.process.tilt.TiltPolicy`;
:func:`~roomgen_guide.process.driver.sample_endpoint_process` is the walk over
the three. `docs/process-adapters.md` is what a new backend is written against.
"""

from roomgen_guide.process.driver import sample_endpoint_process
from roomgen_guide.process.endpoint import Endpoint, EndpointProcess, StepContext
from roomgen_guide.process.tilt import (
    ClippedGradientTilt,
    EnergyGradient,
    TiltPolicy,
    clipped_update,
)

__all__ = [
    "ClippedGradientTilt",
    "Endpoint",
    "EndpointProcess",
    "EnergyGradient",
    "StepContext",
    "TiltPolicy",
    "clipped_update",
    "sample_endpoint_process",
]
