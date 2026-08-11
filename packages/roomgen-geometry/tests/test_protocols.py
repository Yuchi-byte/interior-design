import torch
from jaxtyping import Float, Int
from torch import Tensor

from roomgen_geometry.depth.base import DepthEstimator
from roomgen_geometry.depth.depth_anything import DepthAnythingV2
from roomgen_geometry.segmentation.base import Segmenter


class SyntheticDepth:
    def __call__(
        self, images: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, "batch height width"]:
        return images.mean(dim=1)


class SyntheticSegmenter:
    def __call__(
        self, images: Float[Tensor, "batch 3 height width"]
    ) -> Int[Tensor, "batch height width"]:
        return torch.zeros(
            images.shape[0], *images.shape[2:], dtype=torch.long, device=images.device
        )


def test_fakes_stand_in_for_the_model_backends() -> None:
    images = torch.rand(2, 3, 8, 8, requires_grad=True)
    depth = SyntheticDepth()
    segmentation = SyntheticSegmenter()

    assert isinstance(depth, DepthEstimator)
    assert isinstance(segmentation, Segmenter)
    assert depth(images).shape == (2, 8, 8)
    assert depth(images).requires_grad
    assert segmentation(images).shape == (2, 8, 8)


def test_depth_anything_backend_is_swappable_for_a_fake() -> None:
    assert issubclass(DepthAnythingV2, DepthEstimator)
