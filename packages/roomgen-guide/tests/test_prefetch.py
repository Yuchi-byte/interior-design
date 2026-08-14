from dataclasses import dataclass, field
from pathlib import Path

import pytest

from roomgen_geometry.depth.depth_anything import DEFAULT_CHECKPOINT as DEPTH_CHECKPOINT
from roomgen_geometry.segmentation.eomt import (
    DEFAULT_CHECKPOINT as SEGMENTATION_CHECKPOINT,
)
from roomgen_guide import prefetch
from roomgen_guide.eval.clip import DEFAULT_CHECKPOINT as ADHERENCE_CHECKPOINT
from roomgen_guide.prefetch import CHECKPOINTS, Checkpoint
from roomgen_guide.renovate import DEFAULT_CHECKPOINT as DIFFUSION_CHECKPOINT
from roomgen_guide.sampling.inpaint import (
    DEFAULT_CHECKPOINT as INPAINTING_CHECKPOINT,
)


@dataclass(frozen=True)
class _SnapshotRequest:
    repo_id: str
    cache_dir: Path
    ignore_patterns: list[str] | None
    local_files_only: bool


@dataclass
class _RecordedHub:
    """A `snapshot_download` that records its calls and hands back a real directory.

    The directory is what `verify` inspects, so a test decides what a cached
    checkpoint contains by writing files into `snapshot_for`.
    """

    root: Path
    contents: tuple[str, ...] = ("model.safetensors", "config.json")
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
        snapshot = self.snapshot_for(repo_id)
        for name in self.contents:
            path = snapshot / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
        return str(snapshot)

    def snapshot_for(self, repo_id: str) -> Path:
        snapshot = self.root / repo_id.replace("/", "--")
        snapshot.mkdir(parents=True, exist_ok=True)
        return snapshot

    def request_for(self, repo_id: str) -> _SnapshotRequest:
        return next(request for request in self.requests if request.repo_id == repo_id)


@pytest.fixture
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _RecordedHub:
    recorded = _RecordedHub(tmp_path / "snapshots")
    monkeypatch.setattr(prefetch, "snapshot_download", recorded.snapshot_download)
    return recorded


def _registered(repo_id: str) -> Checkpoint:
    return next(
        checkpoint for checkpoint in CHECKPOINTS if checkpoint.repo_id == repo_id
    )


def test_every_checkpoint_a_benchmark_run_loads_is_registered() -> None:
    assert {checkpoint.repo_id for checkpoint in CHECKPOINTS} == {
        DEPTH_CHECKPOINT,
        SEGMENTATION_CHECKPOINT,
        DIFFUSION_CHECKPOINT,
        INPAINTING_CHECKPOINT,
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


def test_prefetch_passes_each_checkpoints_own_patterns(
    tmp_path: Path, hub: _RecordedHub
) -> None:
    prefetch.prefetch(tmp_path)

    for checkpoint in CHECKPOINTS:
        assert hub.request_for(checkpoint.repo_id).ignore_patterns == (
            list(checkpoint.ignore_patterns) or None
        )


def test_the_diffusion_checkpoint_skips_the_copies_the_pipeline_never_opens() -> None:
    patterns = _registered(DIFFUSION_CHECKPOINT).ignore_patterns

    # The single-file checkpoints, the EMA-less shards and the safety checker,
    # plus the fp16 variant: `load_pipeline` passes no `variant`, so diffusers
    # reads the full-precision shards and never asks for `*.fp16.*`.
    assert {"v1-5-pruned*", "*.non_ema.*", "safety_checker/*", "*.fp16.*"} <= set(
        patterns
    )


def test_the_segmentation_checkpoint_skips_its_duplicate_pickle_weights() -> None:
    # EoMT ships model.safetensors and pytorch_model.bin, ~1.2GB each.
    assert _registered(SEGMENTATION_CHECKPOINT).ignore_patterns == ("*.bin",)


def test_the_adherence_checkpoint_keeps_the_only_weights_it_ships() -> None:
    patterns = _registered(ADHERENCE_CHECKPOINT).ignore_patterns

    # CLIP's TensorFlow and Flax copies are unloadable here, but `main` carries
    # no safetensors, so dropping `*.bin` too would leave nothing to load.
    assert set(patterns) == {"*.h5", "*.msgpack"}
    assert "*.bin" not in patterns


def test_the_depth_checkpoint_is_small_enough_to_take_whole() -> None:
    assert _registered(DEPTH_CHECKPOINT).ignore_patterns == ()


def test_verify_checks_every_checkpoint_without_network(
    tmp_path: Path, hub: _RecordedHub
) -> None:
    prefetch.verify(tmp_path)

    assert len(hub.requests) == len(CHECKPOINTS)
    assert all(request.local_files_only for request in hub.requests)
    assert {request.repo_id for request in hub.requests} == {
        checkpoint.repo_id for checkpoint in CHECKPOINTS
    }


def test_verify_accepts_weights_and_config_below_the_snapshot_root(
    tmp_path: Path, hub: _RecordedHub
) -> None:
    """A diffusers pipeline keeps both a level down, under `unet/`."""
    hub.contents = ("unet/diffusion_pytorch_model.safetensors", "unet/config.json")

    prefetch.verify(tmp_path)


def test_verify_rejects_an_empty_snapshot_directory(
    tmp_path: Path, hub: _RecordedHub
) -> None:
    """The failure the offline snapshot call cannot see: a directory and nothing in it."""
    hub.contents = ()

    with pytest.raises(RuntimeError, match="not cached"):
        prefetch.verify(tmp_path)


def test_verify_rejects_weights_with_no_config_beside_them(
    tmp_path: Path, hub: _RecordedHub
) -> None:
    hub.contents = ("model.safetensors",)

    with pytest.raises(RuntimeError, match=r"config\.json"):
        prefetch.verify(tmp_path)


def test_verify_reports_a_checkpoint_the_hub_client_cannot_find(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Hub's own offline miss reaches the caller as the same failure as an empty one."""

    def missing(repo_id: str, **_: object) -> str:
        raise FileNotFoundError(f"no snapshot for {repo_id}")

    monkeypatch.setattr(prefetch, "snapshot_download", missing)

    with pytest.raises(RuntimeError, match="not in the cache"):
        prefetch.verify(tmp_path)
