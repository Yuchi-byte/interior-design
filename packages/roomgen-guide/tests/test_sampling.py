import pytest
import torch
from jaxtyping import Float, Int
from torch import Tensor

from roomgen_guide.guidance.schedule import ConstantStrength
from roomgen_guide.renovate import renovate
from roomgen_guide.sampling.guided import Img2ImgPipeline, sample_guided


class FakePipeline:
    def __init__(self) -> None:
        self.unet = torch.nn.Identity()
        self.vae = torch.nn.Identity()
        self.scheduler = object()


class MeanBrightnessEnergy:
    def __call__(
        self, predicted_image: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, ""]:
        return predicted_image.mean()


class SyntheticDepth:
    def __call__(
        self, images: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, "batch height width"]:
        return images.mean(dim=1)


class SyntheticSegmenter:
    def __call__(
        self, images: Float[Tensor, "batch 3 height width"]
    ) -> Int[Tensor, "batch height width"]:
        return torch.zeros(images.shape[0], *images.shape[2:], dtype=torch.long)


def test_a_fake_pipeline_stands_in_for_a_diffusers_one() -> None:
    assert isinstance(FakePipeline(), Img2ImgPipeline)


def test_the_guided_loop_takes_a_fake_pipeline_and_an_explicit_generator() -> None:
    with pytest.raises(NotImplementedError):
        sample_guided(
            FakePipeline(),
            torch.rand(1, 3, 8, 8),
            "a mid-century living room",
            guidance=MeanBrightnessEnergy(),
            schedule=ConstantStrength(strength=1.0),
            steps=4,
            noise_strength=0.6,
            text_guidance_scale=7.5,
            generator=torch.Generator().manual_seed(0),
        )


def test_the_renovation_entry_point_accepts_fake_backends() -> None:
    with pytest.raises(NotImplementedError):
        renovate(
            torch.rand(1, 3, 8, 8),
            "a mid-century living room",
            pipeline=FakePipeline(),
            depth_estimator=SyntheticDepth(),
            segmenter=SyntheticSegmenter(),
            schedule=ConstantStrength(strength=1.0),
            erosion_radius=2,
            steps=4,
            noise_strength=0.6,
            text_guidance_scale=7.5,
            generator=torch.Generator().manual_seed(0),
        )
