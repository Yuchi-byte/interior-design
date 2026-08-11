"""The EoMT backend: mask queries decoded inside a plain ViT encoder."""

from __future__ import annotations

from pathlib import Path
from typing import Final, cast

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import interpolate
from transformers import EomtForUniversalSegmentation, EomtImageProcessor

# The DINOv2-backed weights are MIT; the otherwise-identical DINOv3 variants
# carry a restrictive licence and are not drop-in replacements here.
DEFAULT_CHECKPOINT: Final = "tue-mps/ade20k_semantic_eomt_large_512"


class EoMT:
    """EoMT as a :class:`~roomgen_geometry.segmentation.base.Segmenter`.

    Args:
        checkpoint: Hugging Face Hub id, resolved against the ambient Hub cache
            (`ROOMGEN_HF_HOME`); weights are fetched on demand, never vendored.
        device: where the weights live and where inputs are expected.
        dtype: parameter dtype.
        cache_dir: Hub cache to resolve `checkpoint` against; the ambient one
            when omitted.

    The returned ids index
    :data:`~roomgen_geometry.segmentation.ade20k.ADE20K_LABELS`, and the call
    runs under ``torch.no_grad``: the segmentation is evaluated once per photo
    and is a constant for the rest of the run, so nothing here belongs on the
    guidance gradient path.
    """

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        cache_dir: Path | None = None,
    ) -> None:
        # `torch_dtype` rather than transformers 5's `dtype`: it is the spelling
        # both major versions accept, and the declared floor is 4.54.
        model: Module = EomtForUniversalSegmentation.from_pretrained(
            checkpoint, torch_dtype=dtype, cache_dir=cache_dir
        )
        self._model = model.to(device).eval()
        # The processor is loaded for its normalisation statistics, its input
        # size, and its mask-logit decoding; the image preprocessing itself is
        # tensor ops below, so no input ever round-trips through PIL.
        self._processor = EomtImageProcessor.from_pretrained(
            checkpoint, cache_dir=cache_dir
        )
        self._device = device
        self._dtype = dtype
        shortest_edge = self._processor.size["shortest_edge"]
        self._input_size = (
            shortest_edge,
            self._processor.size["longest_edge"] or shortest_edge,
        )
        self._image_mean = torch.tensor(
            self._processor.image_mean, device=device, dtype=dtype
        ).view(1, 3, 1, 1)
        self._image_std = torch.tensor(
            self._processor.image_std, device=device, dtype=dtype
        ).view(1, 3, 1, 1)

    def __call__(
        self, images: Float[Tensor, "batch 3 height width"]
    ) -> Int[Tensor, "batch height width"]:
        batch, _, height, width = images.shape
        with torch.no_grad():
            resized = interpolate(
                images.to(device=self._device, dtype=self._dtype),
                size=self._input_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            outputs = self._model(
                pixel_values=(resized - self._image_mean) / self._image_std
            )
            # The ids are decoded at the input resolution from the per-class
            # *logits*, which is what interpolation is defined on: an id map is
            # categorical and would be meaningless resampled bilinearly. The
            # cast drops the per-class scores this call does not ask for.
            maps = cast(
                list[Tensor],
                self._processor.post_process_semantic_segmentation(
                    outputs, target_sizes=[(height, width)] * batch
                ),
            )
        return torch.stack(maps).to(torch.int64)
