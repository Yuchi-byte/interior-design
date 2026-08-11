"""Machinery of the 0002 benchmark that `roomgen_guide` does not yet provide.

Everything here is a candidate for promotion into the package: the metrics
belong in `roomgen_guide.eval`, the scoring in `roomgen_guide.verdict`, the
console output behind `roomgen_guide.logging`. It lives beside the framework
only because those modules are missing or stubbed today (see `report.md`,
"Gaps"), not because a framework is supposed to carry its own machinery.
"""
