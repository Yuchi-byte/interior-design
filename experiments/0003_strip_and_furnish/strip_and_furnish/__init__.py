"""This study's own glue over `roomgen_guide`: config, photos, backends, the two stages.

Named for the experiment rather than `framework` because a top-level import name
is global: two frameworks on one pytest `pythonpath` cannot both own `framework`,
and the second one to be imported would resolve every submodule against the
first.
"""
