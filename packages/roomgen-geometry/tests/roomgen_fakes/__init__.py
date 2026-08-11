"""Weightless test doubles for the roomgen Protocols, shared by both packages' suites.

The fakes satisfy `DepthEstimator` and `Segmenter`, so the default suite needs
no network, no GPU and no Hub cache. They live beside the package that declares
those Protocols and reach `roomgen-guide`'s suite through `pythonpath` in the
root pytest configuration, which mirrors the production dependency direction.
"""

from roomgen_fakes.depth import FakeDepthEstimator
from roomgen_fakes.gradients import assert_gradient_is_finite_and_nonzero
from roomgen_fakes.segmentation import FakeSegmenter

__all__ = [
    "FakeDepthEstimator",
    "FakeSegmenter",
    "assert_gradient_is_finite_and_nonzero",
]
