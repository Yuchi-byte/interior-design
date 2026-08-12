"""Weightless test doubles for the diffusion side of roomgen, used by the guide suite.

`roomgen_fakes` covers the Protocols the geometry package declares; these cover
the pipeline Protocols `roomgen-guide` declares, and stay here because the
geometry package knows nothing about diffusion. Both reach their suite through
`pythonpath` in the root pytest configuration.
"""

from roomgen_guide_fakes.pipeline import (
    LATENT_CHANNELS,
    FakePipeline,
    FakeScheduler,
    FakeVae,
)

__all__ = [
    "LATENT_CHANNELS",
    "FakePipeline",
    "FakeScheduler",
    "FakeVae",
]
