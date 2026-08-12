"""Both sides of the gate, measured on a finished image.

Scoring is not guidance: everything here reads a render that is already
finished, so it runs under `no_grad`. The differentiable form of the same depth
objective lives in `roomgen_guide.guidance`.
"""

from roomgen_guide.eval.adherence import PromptAdherence
from roomgen_guide.eval.clip import DEFAULT_CHECKPOINT, ClipPromptAdherence
from roomgen_guide.eval.depth import depth_error, depth_residual_map

__all__ = [
    "DEFAULT_CHECKPOINT",
    "ClipPromptAdherence",
    "PromptAdherence",
    "depth_error",
    "depth_residual_map",
]
