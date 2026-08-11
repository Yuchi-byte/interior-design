import pytest
import torch
from torch import Generator

from roomgen_fakes import assert_gradient_is_finite_and_nonzero


def test_the_gradient_assertion_catches_an_energy_that_cannot_steer_its_source(
    generator: Generator,
) -> None:
    source = torch.rand((4, 4), generator=generator, requires_grad=True)
    # The shape of a detached reward: the energy still depends on `source` in
    # the graph, so it backpropagates cleanly and returns nothing but zeros.
    energy = (source.detach() ** 2).sum() + 0.0 * source.sum()

    with pytest.raises(AssertionError, match="no-op"):
        assert_gradient_is_finite_and_nonzero(energy, source)


def test_the_gradient_assertion_catches_a_non_finite_gradient() -> None:
    source = torch.zeros(4, requires_grad=True)
    energy = source.sqrt().sum()

    with pytest.raises(AssertionError, match="not finite"):
        assert_gradient_is_finite_and_nonzero(energy, source)


def test_the_gradient_assertion_refuses_a_non_scalar_energy(
    generator: Generator,
) -> None:
    source = torch.rand((4, 4), generator=generator, requires_grad=True)

    with pytest.raises(ValueError, match="scalar"):
        assert_gradient_is_finite_and_nonzero(source * 2.0, source)
