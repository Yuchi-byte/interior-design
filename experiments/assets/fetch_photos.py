"""Downloads the benchmark room photos listed in `photos.json`.

Reproducible fetch: every entry is a direct URL to a CC0/public-domain/CC-BY
photo on Wikimedia Commons, verified against a recorded sha256 after
download. Run via `uv run python experiments/assets/fetch_photos.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = Path(__file__).resolve().parent / "photos.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "benchmark_photos"
USER_AGENT = (
    "roomgen-benchmark-fetch/1.0 (mark.haoxiang@gmail.com; research use; "
    "see experiments/assets/photos.json)"
)
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 5.0


@dataclass(frozen=True)
class PhotoEntry:
    id: str
    url: str
    license: str
    source_page: str
    sha256: str


def load_manifest(path: Path) -> list[PhotoEntry]:
    raw = json.loads(path.read_text())
    return [PhotoEntry(**entry) for entry in raw["photos"]]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_path(output_dir: Path, entry: PhotoEntry) -> Path:
    suffix = Path(entry.url.split("?", 1)[0]).suffix or ".jpg"
    return output_dir / f"{entry.id}{suffix}"


def download(entry: PhotoEntry, dest: Path) -> None:
    """Fetches `entry.url` to `dest`, retrying on 429/5xx with exponential backoff.

    Wikimedia's edge cache throttles bursts of requests; a transient 429 is
    expected under normal use, not an error worth failing the whole run over.
    """
    request = urllib.request.Request(entry.url, headers={"User-Agent": USER_AGENT})
    backoff = INITIAL_BACKOFF_SECONDS
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(request) as response, dest.open("wb") as f:
                f.write(response.read())
            return
        except urllib.error.HTTPError as error:
            if error.code != 429 and error.code < 500:
                raise
            if attempt == MAX_RETRIES:
                raise
            print(
                f"           retrying after HTTP {error.code} "
                f"(attempt {attempt}/{MAX_RETRIES}, waiting {backoff:.0f}s)",
                file=sys.stderr,
            )
            time.sleep(backoff)
            backoff *= 2


def fetch_one(entry: PhotoEntry, output_dir: Path) -> bool:
    dest = target_path(output_dir, entry)
    if dest.exists() and sha256_of(dest) == entry.sha256:
        print(f"[skip]     {entry.id}: already present and verified ({dest})")
        return True

    print(f"[download] {entry.id}: {entry.url}")
    download(entry, dest)
    actual = sha256_of(dest)
    if actual != entry.sha256:
        print(
            f"[FAIL]     {entry.id}: sha256 mismatch "
            f"(expected {entry.sha256}, got {actual})",
            file=sys.stderr,
        )
        dest.unlink(missing_ok=True)
        return False

    print(f"[ok]       {entry.id}: verified ({dest})")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory to download photos into (default: outputs/benchmark_photos/ at repo root)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_PATH,
        help="path to the photos.json manifest",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    entries = load_manifest(args.manifest)

    results = []
    for i, entry in enumerate(entries):
        if i > 0:
            time.sleep(1.0)  # stay well under Wikimedia's burst-rate throttling
        results.append(fetch_one(entry, args.output_dir))
    failures = results.count(False)
    print(f"\n{len(results) - failures}/{len(results)} photos verified.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
