import pytest
from pydantic import ValidationError

from roomgen_guide.verdict import Verdict, VerdictThresholds


def make_verdict(*, geometry_improved: bool, adherence_held: bool) -> Verdict:
    return Verdict(
        guided_depth_error=0.01,
        control_depth_error=0.05,
        prompt_adherence=0.3,
        geometry_improved=geometry_improved,
        adherence_held=adherence_held,
    )


@pytest.mark.parametrize(
    ("geometry_improved", "adherence_held", "expected"),
    [(True, True, True), (True, False, False), (False, True, False)],
)
def test_a_run_passes_only_on_both_sides_of_the_gate(
    geometry_improved: bool, adherence_held: bool, expected: bool
) -> None:
    verdict = make_verdict(
        geometry_improved=geometry_improved, adherence_held=adherence_held
    )

    assert verdict.passed is expected


def test_thresholds_reject_an_unknown_key() -> None:
    with pytest.raises(ValidationError):
        VerdictThresholds.model_validate(
            {
                "depth_error_margin": 0.1,
                "prompt_adherence_floor": 0.25,
                "depth_error_margin_typo": 0.1,
            }
        )
