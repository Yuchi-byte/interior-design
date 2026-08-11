"""Registers the shared fake backends as a plugin, so both suites see their fixtures.

`pytest_plugins` is only honoured in the rootdir conftest, which is why this
file exists at all; `roomgen_fakes` itself is importable via the `pythonpath`
entry in the root pyproject.
"""

pytest_plugins = ["roomgen_fakes.fixtures"]
