"""The Depth Anything 3 metric backend: a DPT head on a DINOv2 backbone, at absolute scale.

Unlike V2, whose inverse depth is affine-invariant by construction, the
`DA3METRIC-LARGE` checkpoint predicts *depth* — up to one focal-length factor
its own model card gives in closed form (``metres = focal * output / 300``).
Two renders of one room at one framing share that focal, so the factor cancels
between them and the two maps are comparable *without* fitting anything. That
is the whole point of this backend: it is the only way absolute scale can enter
the reward, since a scale-and-shift fit removes exactly the quantity the owner
observed drifting. Whether to actually spend the freedom is the caller's, via
:data:`roomgen_geometry.alignment.AlignmentMode`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import torch
from jaxtyping import Float
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import interpolate

# The metric tier, and Apache-2.0 — as are DA3-Small, DA3-Base and DA3Mono-Large.
# The Large, Giant and Nested any-view checkpoints are CC-BY-NC, so the V2 trap
# repeats one tier up: the *metric* model is the permissive one here, and the
# relative model of the same size is not.
DEFAULT_CHECKPOINT: Final = "depth-anything/DA3METRIC-LARGE"

# Upstream's `InputProcessor` constants (`process_res`, `PATCH_SIZE`, and the
# ImageNet statistics), which no `preprocessor_config.json` on the Hub carries
# for this checkpoint the way the V2-hf repos do.
DEFAULT_INPUT_SIZE: Final = 504
_PATCH_SIZE: Final = 14
_IMAGE_MEAN: Final = (0.485, 0.456, 0.406)
_IMAGE_STD: Final = (0.229, 0.224, 0.225)

# The head's depth activation is `exp`, so its output is positive everywhere and
# the reciprocal below can only blow up in the limit rather than divide by zero.
# The floor is four orders below the ~0.1 canonical output a surface pressed
# against the lens would give, so it never binds on a room; it exists so that a
# pathological frame yields a large finite disparity and a bounded gradient
# instead of an inf that poisons the whole energy.
_MIN_DEPTH: Final = 1e-4

_MISSING_PACKAGE = """\
The Depth Anything 3 backend needs upstream's `depth_anything_3` package, which
is deliberately *not* a roomgen-geometry dependency: it pins `numpy<2` against
this package's `numpy>=2`, and hard-requires xformers, moviepy, open3d and
pycolmap for export paths this backend never reaches. Its model code needs none
of that, so install it alongside without its dependency set:

    uv pip install --no-deps depth-anything-3
    uv pip install addict omegaconf einops
"""


def load_network(checkpoint: str, *, cache_dir: Path | None) -> Module:
    """Build the DA3 network with pretrained weights, off the guidance path.

    Upstream's own ``DepthAnything3.from_pretrained`` is not used, for two
    reasons: its module imports the whole GLB/COLMAP/gaussian-splat export tree
    and so needs moviepy, trimesh, open3d and pycolmap to be importable at all,
    and its ``forward`` is wrapped in ``torch.inference_mode``, which would hand
    back a tensor that cannot be differentiated. The checkpoint's own
    ``config.json`` describes the network completely, so it is built from that
    and the weights loaded into it directly — the same object either route
    produces, minus the tail.
    """
    try:
        # Deferred, and the only reference to the package anywhere: it is
        # installed out of band, so importing it at module scope would make
        # `roomgen_geometry.depth` unimportable on a CI runner that has neither
        # it nor any weights.
        from depth_anything_3.cfg import create_object  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - exercised by the message
        raise ImportError(_MISSING_PACKAGE) from error
    from huggingface_hub import hf_hub_download  # noqa: PLC0415
    from omegaconf import OmegaConf  # noqa: PLC0415
    from safetensors.torch import load_file  # noqa: PLC0415

    def fetch(filename: str) -> Path:
        return Path(hf_hub_download(checkpoint, filename, cache_dir=cache_dir))

    config = json.loads(fetch("config.json").read_text())
    network: Module = create_object(OmegaConf.create(config["config"]))
    # The published state dict is upstream's `DepthAnything3` wrapper's, whose
    # network hangs off `.model`; strict, so a checkpoint that does not fit the
    # config it shipped with fails here rather than predicting from half a head.
    weights = {
        name.removeprefix("model."): tensor
        for name, tensor in load_file(fetch("model.safetensors")).items()
    }
    network.load_state_dict(weights)
    return network


class DepthAnythingV3Metric:
    """DA3 metric as a :class:`~roomgen_geometry.depth.base.DepthEstimator`.

    Args:
        checkpoint: Hugging Face Hub id; weights are fetched on demand, never
            vendored.
        device: where the weights live and where inputs are expected.
        dtype: parameter dtype; a reduced-precision backbone still has to
            produce a usable gradient, so it is the caller's choice.
        cache_dir: Hub snapshot cache the weights are read from and downloaded
            into (`RoomgenSettings.hub_cache`); `None` follows the Hub client's
            own environment.
        input_size: the square edge the image is resampled to, which the
            backbone's patch grid requires to be a multiple of 14.

    Returns inverse depth, like every other backend, rather than the metres the
    network predicts. The conversion is a reciprocal and so preserves the
    absolute determination exactly — the freedom that separates this backend
    from V2 is intact in either space — while keeping three things that are not
    free: the sign convention the whole package reads (larger is nearer, which
    `signed_depth_residual_map` and every mask downstream assume), the error
    weighting the reward was tuned under (a squared residual in metres is
    dominated by the far wall, so guidance would spend its gradient on the one
    surface a renovation is least likely to move), and comparability of the
    energy's magnitude against a V2 run. The one thing the reciprocal does not
    preserve is the *shape* of the affine family: an offset in metres is not an
    offset in disparity, so `shift`-mode alignment absorbs a disparity offset
    and nothing more.

    Everything between the input image and the returned disparity stays on the
    autograd graph, preprocessing included: the resize and the normalisation are
    tensor ops, never a PIL round-trip, and nothing on the path detaches, calls
    ``.item()``, or runs under ``torch.no_grad``. A detached backend silently
    turns guidance into a no-op instead of failing.
    """

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        cache_dir: Path | None = None,
        input_size: int = DEFAULT_INPUT_SIZE,
    ) -> None:
        if input_size % _PATCH_SIZE:
            raise ValueError(
                f"input_size must be a multiple of the patch size {_PATCH_SIZE}, "
                f"got {input_size}"
            )
        self._input_size = (input_size, input_size)
        self._image_mean = torch.tensor(_IMAGE_MEAN, device=device).view(1, 3, 1, 1)
        self._image_std = torch.tensor(_IMAGE_STD, device=device).view(1, 3, 1, 1)

        network = load_network(checkpoint, cache_dir=cache_dir)
        # Frozen parameters, not a `no_grad` forward: the guidance gradient has
        # to reach the input image, and only the weights are held constant.
        network = network.to(device=device, dtype=dtype).eval().requires_grad_(False)
        # The two halves are held by name rather than driven through
        # `network(...)`: see `__call__` for what the wrapper's own forward adds
        # that a guidance path cannot have.
        self._backbone = network.get_submodule("backbone")
        self._head = network.get_submodule("head")
        self._dtype = dtype

    def __call__(
        self, images: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, "batch height width"]:
        _, _, height, width = images.shape
        # Square, unlike upstream's aspect-preserving longest-side resize:
        # reference and sample are always one frame size, so a fixed input costs
        # the guidance loop the same on every step — and, more importantly, puts
        # both maps through one identical resampling, which is what makes the
        # focal factor cancel between them.
        resized = interpolate(
            images,
            size=self._input_size,
            mode="bilinear",
            align_corners=False,
            antialias=False,
        )
        normalised = (resized - self._image_mean) / self._image_std
        # The backbone and the head are driven directly rather than through
        # `DepthAnything3Net.forward`, which would append a sky-region fixup
        # that resamples with `torch.randint` off the *global* RNG and branches
        # on tensor values. Neither belongs on a guidance path that must be
        # deterministic at a passed generator and differentiable throughout —
        # and an interior room has no sky for it to fix.
        features, _ = self._backbone(
            # (batch, views, 3, h, w): the network is multi-view by
            # construction, and one photograph is the single-view case.
            normalised.to(self._dtype).unsqueeze(1),
            cam_token=None,
            export_feat_layers=[],
        )
        predicted: Tensor = self._head(features, *self._input_size, patch_start_idx=0)[
            "depth"
        ][:, 0]
        # Metres up to the focal factor, in, disparity out. Back in the caller's
        # precision before the reciprocal: a half-precision backbone is a memory
        # choice, not a decision about the reward's dtype, and 1/x is where the
        # difference would show.
        disparity = 1.0 / predicted.to(images.dtype).clamp(min=_MIN_DEPTH)
        return interpolate(
            disparity.unsqueeze(1),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
