import pytest

from roomgen_guide.experiment import VerdictGateConfig
from roomgen_guide.verdict import Verdict, score_run


def make_verdict(*, geometry_improved: bool, adherence_held: bool) -> Verdict:
    return Verdict(
        guided_depth_error=0.01,
        control_depth_error=0.05,
        guided_prompt_adherence=0.3,
        control_prompt_adherence=0.32,
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


def test_scoring_a_run_takes_both_runs_measurements_and_the_framework_gate() -> None:
    with pytest.raises(NotImplementedError):
        score_run(
            guided_depth_error=0.01,
            control_depth_error=0.05,
            guided_prompt_adherence=0.3,
            control_prompt_adherence=0.32,
            gate=VerdictGateConfig(depth_error_margin=0.2, prompt_adherence_floor=0.9),
        )
