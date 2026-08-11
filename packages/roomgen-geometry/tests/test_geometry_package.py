from importlib.metadata import version

import roomgen_geometry


def test_version_matches_installed_distribution() -> None:
    """The hardcoded __version__ must not drift from the packaged version."""
    assert roomgen_geometry.__version__ == version("roomgen-geometry")
