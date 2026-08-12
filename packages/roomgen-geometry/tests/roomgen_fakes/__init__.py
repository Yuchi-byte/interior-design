"""Weightless test doubles for the roomgen Protocols, shared by both packages' suites.

The fakes satisfy `DepthEstimator`, `Segmenter` and `ClassLogitSegmenter`, so
the default suite needs no network, no GPU and no Hub cache. They live beside the package that declares
those Protocols and reach `roomgen-guide`'s suite through `pythonpath` in the
root pytest configuration, which mirrors the production dependency direction.
"""

from roomgen_fakes.depth import FakeDepthEstimator
from roomgen_fakes.gradients import assert_gradient_is_finite_and_nonzero
from roomgen_fakes.segmentation import FakeClassLogitSegmenter, FakeSegmenter

__all__ = [
    "FakeClassLogitSegmenter",
    "FakeDepthEstimator",
    "FakeSegmenter",
    "assert_gradient_is_finite_and_nonzero",
]
