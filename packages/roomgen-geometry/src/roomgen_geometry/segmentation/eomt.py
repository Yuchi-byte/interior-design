"""The EoMT backend: mask queries decoded inside a plain ViT encoder."""

from __future__ import annotations

from pathlib import Path
from typing import Final

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
    runs under ``torch.no_grad``: an id map decides which pixels a reward may
    score, which is a discrete decision that carries no useful gradient. The
    scores those ids are the argmax of are reachable *with* gradient through
    :meth:`class_logits`, which is what a class-consistency reward differentiates.
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
        # Frozen like the depth backend, and now load-bearing rather than tidy:
        # `class_logits` runs a backward through this network at every guided
        # step, and a trainable parameter would have that backward accumulate a
        # `.grad` the size of the weights for a gradient nobody reads.
        self._model = model.to(device).eval().requires_grad_(False)
        # The processor is loaded for its normalisation statistics and its
        # input size only. Both the preprocessing and the semantic decode are
        # tensor ops below: no input round-trips through PIL, and the decode
        # does not depend on a `post_process_semantic_segmentation` whose
        # handling of a directly-called model differs between the transformers
        # versions this package supports.
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
        with torch.no_grad():
            return self.class_logits(images).argmax(dim=1)

    def class_logits(
        self, images: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, "batch classes height width"]:
        """Per-pixel class scores on the input frame, **carrying gradient into the image**.

        Deliberately not under ``no_grad``, unlike :meth:`__call__`, which
        takes the argmax of this under one: this is the one path into the
        segmentation network a reward may differentiate, and it costs a forward
        *and a backward* through the encoder at every guided step it is
        evaluated on.

        The values are EoMT's semantic scores — each query's class distribution
        (null class dropped) weighted by that query's mask probability and summed
        — so they are non-negative mass rather than log-odds, and
        :meth:`__call__` is exactly their argmax: the ids that gate a reward and
        the scores it differentiates are one decode of one forward.
        """
        _, _, height, width = images.shape
        outputs = self._model(pixel_values=self._preprocess(images))
        # The processor's `post_process_semantic_segmentation` computes this
        # and then throws it away: it returns ids, and the keyword that keeps
        # the scores beside them is newer than the transformers floor this
        # package declares. Some versions also assume the processor did the
        # tiling and fail outright on a model called directly, which this
        # backend does. The decode is four differentiable lines: spelled out
        # here, and both paths read it.
        queries = outputs.class_queries_logits.softmax(dim=-1)[..., :-1]
        masks = interpolate(
            outputs.masks_queries_logits,
            size=self._input_size,
            mode="bilinear",
            align_corners=False,
        ).sigmoid()
        scores = torch.einsum("bqc,bqhw->bchw", queries, masks)
        return interpolate(
            scores, size=(height, width), mode="bilinear", align_corners=False
        )

    def _preprocess(
        self, images: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, "batch 3 _ _"]:
        """Resize and normalise as tensor ops, so the gradient survives the round trip.

        Shared by both paths: whether the result is differentiated is the
        caller's business, and the two must see the same pixels or the ids would
        gate scores taken from a different image.
        """
        resized = interpolate(
            images.to(device=self._device, dtype=self._dtype),
            size=self._input_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return (resized - self._image_mean) / self._image_std
