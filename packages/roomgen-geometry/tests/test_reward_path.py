import pytest
import torch

from roomgen_geometry.masking import architectural_mask, erode_boundaries
from roomgen_geometry.reward import gradient_matching_error


def test_the_mask_takes_segmenter_output_as_it_comes() -> None:
    segmentation = torch.zeros(1, 8, 8, dtype=torch.long)

    with pytest.raises(NotImplementedError):
        architectural_mask(segmentation, erosion_radius=2)
    with pytest.raises(NotImplementedError):
        erode_boundaries(torch.ones(1, 8, 8, dtype=torch.bool), radius=2)


def test_the_gradient_matching_term_takes_two_disparity_maps_and_a_mask() -> None:
    prediction = torch.rand(1, 8, 8, requires_grad=True)
    reference = torch.rand(1, 8, 8)
    mask = torch.ones(1, 8, 8, dtype=torch.bool)

    with pytest.raises(NotImplementedError):
        gradient_matching_error(prediction, reference, mask, levels=4)
