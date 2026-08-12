from importlib.metadata import version

import roomgen_geometry
import roomgen_guide


def test_the_declared_version_matches_the_installed_distribution() -> None:
    assert roomgen_guide.__version__ == version("roomgen-guide")


def test_the_geometry_workspace_dependency_is_importable() -> None:
    assert roomgen_geometry.__version__ == version("roomgen-geometry")
