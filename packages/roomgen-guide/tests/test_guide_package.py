from importlib.metadata import version

import roomgen_geometry
import roomgen_guide


def test_version_matches_installed_distribution() -> None:
    """The hardcoded __version__ must not drift from the packaged version."""
    assert roomgen_guide.__version__ == version("roomgen-guide")


def test_geometry_workspace_dependency_is_importable() -> None:
    """roomgen-guide is useless without the grounding stack it guides sampling with."""
    assert roomgen_geometry.__version__ == version("roomgen-geometry")
