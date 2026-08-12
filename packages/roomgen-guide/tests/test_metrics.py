import pytest

from roomgen_guide.metrics import RunMetrics

_CONTROL_DEPTH_ERROR = 0.05
_CONTROL_ADHERENCE = 0.30


def _metrics(
    *,
    guided_depth_error: float = 0.01,
    control_depth_error: float = _CONTROL_DEPTH_ERROR,
    guided_prompt_adherence: float = _CONTROL_ADHERENCE,
    control_prompt_adherence: float = _CONTROL_ADHERENCE,
    common_coverage: float = 0.8,
) -> RunMetrics:
    return RunMetrics(
        guided_depth_error=guided_depth_error,
        control_depth_error=control_depth_error,
        guided_prompt_adherence=guided_prompt_adherence,
        control_prompt_adherence=control_prompt_adherence,
        reconstruction_depth_error=0.002,
        common_coverage=common_coverage,
        guided_shell_coverage=0.9,
        control_shell_coverage=0.85,
    )


@pytest.mark.parametrize(
    ("guided_depth_error", "expected"),
    [(0.05, 0.0), (0.04, 0.2), (0.01, 0.8), (0.10, -1.0)],
)
def test_the_depth_reduction_is_the_share_of_the_controls_error_that_went(
    guided_depth_error: float, expected: float
) -> None:
    metrics = _metrics(guided_depth_error=guided_depth_error)

    assert metrics.depth_error_reduction == pytest.approx(expected)


def test_the_adherence_ratio_is_read_against_the_controls_own_score() -> None:
    metrics = _metrics(guided_prompt_adherence=0.27)

    assert metrics.adherence_ratio == pytest.approx(0.9)


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_both_ratios_are_unchanged_by_the_scale_the_metrics_report_in(
    scale: float,
) -> None:
    # Neither quantity has a scale that carries from one scene to the next, so
    # only the ratios are comparable across scenes — and they must be.
    metrics = _metrics(
        guided_depth_error=0.03 * scale,
        control_depth_error=_CONTROL_DEPTH_ERROR * scale,
        guided_prompt_adherence=0.28 * scale,
        control_prompt_adherence=_CONTROL_ADHERENCE * scale,
    )

    assert metrics.depth_error_reduction == pytest.approx(0.4)
    assert metrics.adherence_ratio == pytest.approx(0.28 / 0.30)


@pytest.mark.parametrize("guided_depth_error", [0.0, 0.01])
def test_a_control_with_no_depth_error_reports_no_reduction_rather_than_a_zero(
    guided_depth_error: float,
) -> None:
    # A zero control leaves nothing to remove, so there is no share to report —
    # including when the guided run also scored zero, where a printed 0.0 would
    # read as a measured "no change" instead of an unmeasurable one.
    metrics = _metrics(guided_depth_error=guided_depth_error, control_depth_error=0.0)

    assert metrics.depth_error_reduction is None


def test_a_control_with_no_adherence_reports_no_ratio_rather_than_a_zero() -> None:
    metrics = _metrics(control_prompt_adherence=0.0, guided_prompt_adherence=0.2)

    assert metrics.adherence_ratio is None


def test_the_measurements_are_kept_beside_the_ratios_read_off_them() -> None:
    metrics = _metrics(guided_depth_error=0.01, guided_prompt_adherence=0.3)

    assert metrics.guided_depth_error == 0.01
    assert metrics.control_depth_error == _CONTROL_DEPTH_ERROR
    assert metrics.guided_prompt_adherence == 0.3
    assert metrics.control_prompt_adherence == _CONTROL_ADHERENCE
    assert metrics.reconstruction_depth_error == 0.002
    assert metrics.common_coverage == 0.8
