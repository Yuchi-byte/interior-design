"""Weightless test doubles for the diffusion side of roomgen, used by the guide suite.

`roomgen_fakes` covers the Protocols the geometry package declares; these cover
the pipeline Protocols `roomgen-guide` declares plus a second, non-diffusion
`EndpointProcess`, and stay here because the geometry package knows nothing
about diffusion. Both reach their suite through `pythonpath` in the root pytest
configuration.
"""

from roomgen_guide_fakes.clip import (
    EMBEDDING_WIDTH,
    INPUT_SIZE,
    FakeClipImageEncoder,
    fake_clip_text_guidance,
)
from roomgen_guide_fakes.flow_map import (
    START_TIME,
    ToyFlowMapBackend,
    ToyFlowMapProcess,
)
from roomgen_guide_fakes.inpaint import FakeInpaintPipeline
from roomgen_guide_fakes.pipeline import (
    LATENT_CHANNELS,
    FakePipeline,
    FakeScheduler,
    FakeVae,
)

__all__ = [
    "EMBEDDING_WIDTH",
    "INPUT_SIZE",
    "LATENT_CHANNELS",
    "START_TIME",
    "FakeClipImageEncoder",
    "FakeInpaintPipeline",
    "FakePipeline",
    "FakeScheduler",
    "FakeVae",
    "ToyFlowMapBackend",
    "ToyFlowMapProcess",
    "fake_clip_text_guidance",
]
