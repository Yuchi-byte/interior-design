import pytest
import torch
from jaxtyping import TypeCheckError

from roomgen_geometry.reference import GeometricReference


def test_reference_rejects_tensors_still_on_the_autograd_graph() -> None:
    disparity = torch.rand(1, 4, 4, requires_grad=True)
    with pytest.raises(ValueError, match="constant"):
        GeometricReference(
            disparity=disparity, mask=torch.ones(1, 4, 4, dtype=torch.bool)
        )


def test_reference_rejects_a_mask_of_a_different_shape() -> None:
    # `TypeCheckError` under the suite's jaxtyping hook, `ValueError` from the
    # dataclass's own check unhooked; both name the mask.
    with pytest.raises((TypeCheckError, ValueError), match="mask"):
        GeometricReference(
            disparity=torch.rand(1, 4, 4),
            mask=torch.ones(1, 4, 5, dtype=torch.bool),
        )
