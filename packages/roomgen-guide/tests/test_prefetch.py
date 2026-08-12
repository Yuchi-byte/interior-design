from dataclasses import dataclass, field
from pathlib import Path

import pytest

from roomgen_geometry.depth.depth_anything import DEFAULT_CHECKPOINT as DEPTH_CHECKPOINT
from roomgen_geometry.segmentation.eomt import (
    DEFAULT_CHECKPOINT as SEGMENTATION_CHECKPOINT,
)
from roomgen_guide import prefetch
from roomgen_guide.eval.clip import DEFAULT_CHECKPOINT as ADHERENCE_CHECKPOINT
from roomgen_guide.prefetch import CHECKPOINTS
from roomgen_guide.renovate import DEFAULT_CHECKPOINT as DIFFUSION_CHECKPOINT


@dataclass(frozen=True)
class _SnapshotRequest:
    repo_id: str
    cache_dir: Path
    ignore_patterns: list[str] | None
    local_files_only: bool


@dataclass
class _RecordedHub:
    requests: list[_SnapshotRequest] = field(default_factory=list)

    def snapshot_download(
        self,
        repo_id: str,
        *,
        cache_dir: Path,
        ignore_patterns: list[str] | None,
        local_files_only: bool,
    ) -> str:
        self.requests.append(
            _SnapshotRequest(repo_id, cache_dir, ignore_patterns, local_files_only)
        )
        return repo_id

    def request_for(self, repo_id: str) -> _SnapshotRequest:
        return next(request for request in self.requests if request.repo_id == repo_id)


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> _RecordedHub:
    recorded = _RecordedHub()
    monkeypatch.setattr(prefetch, "snapshot_download", recorded.snapshot_download)
    return recorded


def test_every_checkpoint_a_benchmark_run_loads_is_registered() -> None:
    assert {checkpoint.repo_id for checkpoint in CHECKPOINTS} == {
        DEPTH_CHECKPOINT,
        SEGMENTATION_CHECKPOINT,
        DIFFUSION_CHECKPOINT,
        ADHERENCE_CHECKPOINT,
    }


def test_prefetch_requests_every_registered_checkpoint(
    tmp_path: Path, hub: _RecordedHub
) -> None:
    prefetch.prefetch(tmp_path)

    assert {request.repo_id for request in hub.requests} == {
        checkpoint.repo_id for checkpoint in CHECKPOINTS
    }
    assert all(request.cache_dir == tmp_path for request in hub.requests)
    assert not any(request.local_files_only for request in hub.requests)


def test_prefetch_filters_the_stable_diffusion_checkpoint(
    tmp_path: Path, hub: _RecordedHub
) -> None:
    prefetch.prefetch(tmp_path)

    sd15 = next(checkpoint for checkpoint in CHECKPOINTS if checkpoint.ignore_patterns)
    patterns = hub.request_for(sd15.repo_id).ignore_patterns
    assert patterns == list(sd15.ignore_patterns)
    assert "v1-5-pruned*" in sd15.ignore_patterns
    assert "safety_checker/*" in sd15.ignore_patterns


def test_prefetch_downloads_the_other_three_checkpoints_unfiltered(
    tmp_path: Path, hub: _RecordedHub
) -> None:
    prefetch.prefetch(tmp_path)

    unfiltered = [
        checkpoint for checkpoint in CHECKPOINTS if not checkpoint.ignore_patterns
    ]
    assert len(unfiltered) == 3
    for checkpoint in unfiltered:
        assert hub.request_for(checkpoint.repo_id).ignore_patterns is None


def test_verify_checks_every_checkpoint_without_network(
    tmp_path: Path, hub: _RecordedHub
) -> None:
    prefetch.verify(tmp_path)

    assert len(hub.requests) == len(CHECKPOINTS)
    assert all(request.local_files_only for request in hub.requests)
    assert {request.repo_id for request in hub.requests} == {
        checkpoint.repo_id for checkpoint in CHECKPOINTS
    }
