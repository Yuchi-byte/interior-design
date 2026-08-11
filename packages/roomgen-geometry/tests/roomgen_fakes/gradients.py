"""The gradient assertion every differentiability test in this repo is built on."""

import torch
from jaxtyping import Float
from torch import Tensor


def assert_gradient_is_finite_and_nonzero(
    energy: Float[Tensor, ""], source: Tensor
) -> Tensor:
    """Differentiate a scalar energy back to `source` and assert the result can actually steer it.

    A detached graph, a masked-out region, or a guarded division that swallowed
    its branch all leave a loop that runs perfectly normally while the gradient
    is silently zero or NaN, so asserting the energy computes is worth nothing:
    every gradient test asserts through here instead. Returns the gradient for
    any further assertion about its shape or direction.
    """
    if energy.ndim != 0:
        raise ValueError(f"energy must be a scalar, got shape {tuple(energy.shape)}")
    if not source.requires_grad:
        raise ValueError("source does not require grad, so no gradient can reach it")

    (gradient,) = torch.autograd.grad(energy, source)
    assert torch.isfinite(gradient).all(), (
        f"gradient is not finite: {int(torch.isnan(gradient).sum())} NaN, "
        f"{int(torch.isinf(gradient).sum())} inf of {gradient.numel()} elements"
    )
    assert float(gradient.abs().max()) > 0.0, (
        "gradient is identically zero: guidance built on this energy would be a no-op"
    )
    return gradient
