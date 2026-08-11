"""Turns the geometry package's shape annotations into runtime checks, and registers the fakes.

`pytest_plugins` is only honoured in the rootdir conftest, which is why this
file exists at all; `roomgen_fakes` itself is importable via the `pythonpath`
entry in the root pyproject.
"""

from jaxtyping import install_import_hook

# An import hook only rewrites modules imported *after* it is installed, and
# this file is the earliest thing pytest executes — hence the call at module
# scope, above every other import, rather than in a fixture.
#
# Scoped to the geometry package on purpose: it is the one that speaks in
# `Float[Tensor, "batch height width"]`, so a fake or a caller that transposes
# an axis fails here instead of producing a plausible number. `roomgen_guide`
# is left alone because beartype rejects `Config.from_raw`'s PEP 673 `Self`
# outside a class it decorates itself.
install_import_hook("roomgen_geometry", "beartype.beartype")

pytest_plugins = ["roomgen_fakes.fixtures"]
