from pathlib import Path
from typing import Any

import pytest

from roomgen_geometry.depth.depth_anything import DEFAULT_CHECKPOINT as DEPTH_CHECKPOINT
from roomgen_geometry.segmentation.eomt import (
    DEFAULT_CHECKPOINT as SEGMENTATION_CHECKPOINT,
)
from roomgen_guide import prefetch
from roomgen_guide.eval.clip import DEFAULT_CHECKPOINT as ADHERENCE_CHECKPOINT
from roomgen_guide.prefetch import CHECKPOINTS
from roomgen_guide.renovate import DEFAULT_CHECKPOINT as DIFFUSION_CHECKPOINT


@pytest.fixture
def recorded_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def _fake_snapshot_download(repo_id: str, **kwargs: Any) -> str:
        calls.append({"repo_id": repo_id, **kwargs})
        return repo_id

    monkeypatch.setattr(prefetch, "snapshot_download", _fake_snapshot_download)
    return calls


def test_every_checkpoint_a_benchmark_run_loads_is_registered() -> None:
    assert {checkpoint.repo_id for checkpoint in CHECKPOINTS} == {
        DEPTH_CHECKPOINT,
        SEGMENTATION_CHECKPOINT,
        DIFFUSION_CHECKPOINT,
        ADHERENCE_CHECKPOINT,
    }


def test_prefetch_requests_every_registered_checkpoint(
    tmp_path: Path, recorded_calls: list[dict[str, Any]]
) -> None:
    prefetch.prefetch(tmp_path)

    requested_repo_ids = {call["repo_id"] for call in recorded_calls}
    assert requested_repo_ids == {checkpoint.repo_id for checkpoint in CHECKPOINTS}
    assert all(call["cache_dir"] == tmp_path for call in recorded_calls)


def test_prefetch_filters_the_stable_diffusion_checkpoint(
    tmp_path: Path, recorded_calls: list[dict[str, Any]]
) -> None:
    prefetch.prefetch(tmp_path)

    sd15 = next(checkpoint for checkpoint in CHECKPOINTS if checkpoint.ignore_patterns)
    call = next(call for call in recorded_calls if call["repo_id"] == sd15.repo_id)
    assert call["ignore_patterns"] == list(sd15.ignore_patterns)
    assert "v1-5-pruned*" in call["ignore_patterns"]
    assert "safety_checker/*" in call["ignore_patterns"]


def test_prefetch_downloads_the_other_three_checkpoints_unfiltered(
    tmp_path: Path, recorded_calls: list[dict[str, Any]]
) -> None:
    prefetch.prefetch(tmp_path)

    unfiltered = [
        checkpoint for checkpoint in CHECKPOINTS if not checkpoint.ignore_patterns
    ]
    assert len(unfiltered) == 3
    for checkpoint in unfiltered:
        call = next(
            call for call in recorded_calls if call["repo_id"] == checkpoint.repo_id
        )
        assert call["ignore_patterns"] is None


def test_verify_checks_every_checkpoint_without_network(
    tmp_path: Path, recorded_calls: list[dict[str, Any]]
) -> None:
    prefetch.verify(tmp_path)

    assert len(recorded_calls) == len(CHECKPOINTS)
    assert all(call["local_files_only"] is True for call in recorded_calls)
    assert {call["repo_id"] for call in recorded_calls} == {
        checkpoint.repo_id for checkpoint in CHECKPOINTS
    }
