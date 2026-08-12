from importlib.metadata import version

import roomgen_geometry


def test_the_declared_version_matches_the_installed_distribution() -> None:
    assert roomgen_geometry.__version__ == version("roomgen-geometry")
