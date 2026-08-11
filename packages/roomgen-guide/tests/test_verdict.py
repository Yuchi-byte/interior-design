import pytest

from roomgen_guide.experiment import VerdictGateConfig
from roomgen_guide.verdict import Verdict, score_run

_GATE = VerdictGateConfig(depth_error_margin=0.2, prompt_adherence_floor=0.9)
_CONTROL_DEPTH_ERROR = 0.05
_CONTROL_ADHERENCE = 0.30


def make_verdict(*, geometry_improved: bool, adherence_held: bool) -> Verdict:
    return Verdict(
        guided_depth_error=0.01,
        control_depth_error=0.05,
        guided_prompt_adherence=0.3,
        control_prompt_adherence=0.32,
        geometry_improved=geometry_improved,
        adherence_held=adherence_held,
    )


def score(
    *,
    guided_depth_error: float = 0.01,
    guided_prompt_adherence: float = _CONTROL_ADHERENCE,
    scale: float = 1.0,
) -> Verdict:
    return score_run(
        guided_depth_error=guided_depth_error * scale,
        control_depth_error=_CONTROL_DEPTH_ERROR * scale,
        guided_prompt_adherence=guided_prompt_adherence * scale,
        control_prompt_adherence=_CONTROL_ADHERENCE * scale,
        gate=_GATE,
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


def test_scoring_a_run_keeps_both_runs_measurements_beside_the_verdict() -> None:
    verdict = score_run(
        guided_depth_error=0.01,
        control_depth_error=0.05,
        guided_prompt_adherence=0.3,
        control_prompt_adherence=0.32,
        gate=_GATE,
    )

    assert verdict.guided_depth_error == 0.01
    assert verdict.control_depth_error == 0.05
    assert verdict.guided_prompt_adherence == 0.3
    assert verdict.control_prompt_adherence == 0.32
    assert verdict.passed


@pytest.mark.parametrize(
    ("guided_depth_error", "expected"),
    # The margin is 0.2 of a control at 0.05, so the win threshold is 0.04.
    [(0.03, True), (0.04, True), (0.045, False), (0.06, False)],
)
def test_geometry_wins_only_by_the_configured_margin_on_the_control(
    guided_depth_error: float, expected: bool
) -> None:
    assert score(guided_depth_error=guided_depth_error).geometry_improved is expected


@pytest.mark.parametrize(
    ("guided_prompt_adherence", "expected"),
    # The floor is 0.9 of a control at 0.30, so adherence holds down to 0.27.
    [(0.33, True), (0.27, True), (0.26, False)],
)
def test_adherence_holds_only_against_the_controls_own_score(
    guided_prompt_adherence: float, expected: bool
) -> None:
    verdict = score(guided_prompt_adherence=guided_prompt_adherence)

    assert verdict.adherence_held is expected


def test_a_run_that_declines_to_renovate_fails_the_gate_it_wins_on_geometry() -> None:
    verdict = score(guided_depth_error=0.0, guided_prompt_adherence=0.1)

    assert verdict.geometry_improved
    assert not verdict.adherence_held
    assert not verdict.passed


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_the_gate_is_unchanged_by_the_scale_the_metrics_report_in(scale: float) -> None:
    # Both thresholds are fractions of the control, so a scene whose depth error
    # runs three orders larger is judged identically.
    verdict = score(guided_depth_error=0.03, guided_prompt_adherence=0.28, scale=scale)

    assert verdict.passed
