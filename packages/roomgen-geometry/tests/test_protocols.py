from roomgen_geometry.depth.base import DepthEstimator
from roomgen_geometry.depth.depth_anything import DepthAnythingV2
from roomgen_geometry.depth.depth_anything_v3 import DepthAnythingV3Metric
from roomgen_geometry.segmentation.base import Segmenter
from roomgen_geometry.segmentation.eomt import EoMT


def test_the_real_backends_are_swappable_for_the_fakes() -> None:
    assert issubclass(DepthAnythingV2, DepthEstimator)
    assert issubclass(DepthAnythingV3Metric, DepthEstimator)
    assert issubclass(EoMT, Segmenter)
