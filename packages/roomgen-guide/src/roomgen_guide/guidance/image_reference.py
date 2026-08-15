"""The image-reference energies: hold the render near the user's photo in a frozen vision model's feature space."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, cast, runtime_checkable

import torch
from jaxtyping import Bool, Float
from torch import Tensor
from torch.nn import Module, Sequential
from torch.nn.functional import adaptive_avg_pool2d, cosine_similarity
from transformers import (
    AutoModel,
    BitImageProcessor,
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    SiglipImageProcessor,
    SiglipVisionModel,
)

from roomgen_geometry.masking import scorable_mask
from roomgen_geometry.segmentation.base import Segmenter
from roomgen_guide.guidance.base import Guidance
from roomgen_guide.guidance.clip import ClipImageEncoder, ClipPreprocessing

DEFAULT_CLIP_CHECKPOINT: Final = "openai/clip-vit-base-patch32"
DEFAULT_DINOV2_CHECKPOINT: Final = "facebook/dinov2-base"

# FLUX.1-Redux-dev ships the SigLIP so400m vision tower as a subfolder, so a
# cache that already holds the Redux weights pays no download for this backend.
DEFAULT_SIGLIP_CHECKPOINT: Final = "black-forest-labs/FLUX.1-Redux-dev"
DEFAULT_SIGLIP_SUBFOLDER: Final = "image_encoder"
DEFAULT_SIGLIP_PROCESSOR_SUBFOLDER: Final = "feature_extractor"

# LPIPS evaluates at 256 and VGG's stride-32 pyramid wants nothing finer for a
# similarity score; a 512 render backward through five conv stacks would only
# spend VRAM the sampler needs.
DEFAULT_VGG_INPUT_SIZE: Final = 256

# The three energies score and reduce in float32 whatever dtype the tower runs
# in: a bfloat16 cosine quantises the energy, not just the model.
_ENERGY_DTYPE: Final = torch.float32


@dataclass(frozen=True)
class ImageReferenceGuidance:
    """Negative cosine similarity between the render and the reference photo, in one frozen encoder's embedding space.

    The photo's embedding is computed once and frozen — a constant of the walk,
    exactly as `GeometricReference` is — while the render's is re-encoded
    differentiably at every step. Bounded in ``[-1, 1]`` and lowest where the
    encoder cannot tell the render from the photo, which names the known risk:
    a full-frame embedding carries the furniture as well as the shell, so
    descending it can pull the old furnishings back. That is a property to
    measure beside an adherence judge, not something this term can see.

    Args:
        encoder: frozen and in eval, and never under `no_grad` — the gradient
            reaches the latent through it.
        reference_features: the photo's embedding, frozen once: a constant, so
            construction rejects one still carrying gradient.
    """

    encoder: ClipImageEncoder
    reference_features: Float[Tensor, "1 embedding"]
    preprocessing: ClipPreprocessing

    def __post_init__(self) -> None:
        if self.reference_features.requires_grad:
            raise ValueError(
                "the reference photo's embedding is a constant of the walk: "
                "detach it before freezing it into an ImageReferenceGuidance"
            )

    def __call__(
        self, predicted_image: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, "batch"]:
        image_features = self.encoder(self.preprocessing(predicted_image))
        return -cosine_similarity(
            image_features.to(_ENERGY_DTYPE),
            self.reference_features.to(
                device=image_features.device, dtype=_ENERGY_DTYPE
            ),
        )


@runtime_checkable
class PerceptualFeatureExtractor(Protocol):
    """A feature pyramid for perceptual distance: preprocessed pixels in, one map per stage out.

    Args:
        pixel_values: already resized and normalised for the backbone, and still
            attached to the latent it came from — every returned map must carry
            the gradient back, so nothing here may detach.
    """

    def __call__(
        self, pixel_values: Float[Tensor, "batch 3 size size"]
    ) -> tuple[Float[Tensor, "batch channels height width"], ...]: ...


@dataclass(frozen=True)
class PerceptualReferenceGuidance:
    """LPIPS-shaped perceptual distance between the render and the reference photo (arXiv:1801.03924).

    Per stage, both feature maps are unit-normalised along channels, differenced,
    and the squared difference averaged over space and channels; the stages sum.
    LPIPS with uniform stage weights in place of the learned calibration — an
    energy only ever competes against its own gradient scale, so the calibration
    that makes LPIPS a human-aligned *metric* buys nothing here. Zero exactly at
    the photo and spatial by construction, where an embedding cosine is global.

    Args:
        extractor: frozen and in eval, never under `no_grad`.
        reference_features: the photo's pyramid, frozen once: constants, so
            construction rejects any still carrying gradient.
    """

    extractor: PerceptualFeatureExtractor
    reference_features: tuple[Float[Tensor, "1 channels height width"], ...]
    preprocessing: ClipPreprocessing

    def __post_init__(self) -> None:
        if not self.reference_features:
            raise ValueError(
                "a perceptual distance needs at least one feature stage; an "
                "empty pyramid has no gradient to steer by"
            )
        for features in self.reference_features:
            if features.requires_grad:
                raise ValueError(
                    "the reference photo's feature maps are constants of the "
                    "walk: detach them before freezing them into a "
                    "PerceptualReferenceGuidance"
                )

    def __call__(
        self, predicted_image: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, "batch"]:
        stages = self.extractor(self.preprocessing(predicted_image))
        if len(stages) != len(self.reference_features):
            raise ValueError(
                f"the extractor returned {len(stages)} feature stages against "
                f"{len(self.reference_features)} frozen from the reference: the "
                f"two pyramids must come from the same extractor"
            )
        energy: Tensor | None = None
        for features, reference in zip(stages, self.reference_features, strict=True):
            difference = _unit_channels(features.to(_ENERGY_DTYPE)) - _unit_channels(
                reference.to(device=features.device, dtype=_ENERGY_DTYPE)
            )
            term = difference.square().mean(dim=(1, 2, 3))
            energy = term if energy is None else energy + term
        assert energy is not None  # the empty pyramid was rejected at construction
        return energy


@runtime_checkable
class SpatialPatchEncoder(Protocol):
    """A vision tower kept spatial: preprocessed pixels in, the patch-token grid out.

    Args:
        pixel_values: already resized and normalised for the backbone, and still
            attached to the latent it came from — the tokens must carry the
            gradient back, so nothing here may detach. The returned tokens are
            the square patch grid in row-major order, with any CLS or register
            summaries already stripped.
    """

    def __call__(
        self, pixel_values: Float[Tensor, "batch 3 size size"]
    ) -> Float[Tensor, "batch tokens embedding"]: ...


def patch_positions_from_mask(
    mask: Bool[Tensor, "batch height width"],
    *,
    tokens: int,
    min_coverage: float = 0.5,
) -> Bool[Tensor, "batch tokens"]:
    """A pixel mask downsampled to a square patch grid, in the grid's row-major order.

    Args:
        tokens: how many patch tokens the encoder returns; must be a square
            number, since the grid it names is square.
        min_coverage: the share of a patch's pixels the mask must cover before
            the patch counts. The default is a majority vote — a patch is scored
            when at least half of it is masked, so segment fringes neither claim
            a patch nor erase one.
    """
    side = math.isqrt(tokens)
    if side * side != tokens:
        raise ValueError(
            f"{tokens} patch tokens do not form a square grid; a spatial encoder "
            f"must strip its CLS and register tokens before its grid is masked"
        )
    coverage = adaptive_avg_pool2d(mask.unsqueeze(1).to(torch.float32), side)
    return (coverage >= min_coverage).flatten(1)


@dataclass(frozen=True)
class MaskedPatchReferenceGuidance:
    """Negative mean patch-token cosine against the reference photo, over masked patch positions only.

    The spatial contrast to :class:`ImageReferenceGuidance`'s global embedding,
    restricted to where the reference's mask is: each scored patch is compared
    with the reference's token *at the same position*, so layout is part of the
    similarity, and the unmasked patches — the region the renovation is free to
    change — contribute nothing, forwards or backwards. Built for an empty-room
    reference, whose full frame would otherwise punish furnishing itself.

    The mask philosophy mirrors :class:`~roomgen_guide.guidance.geometric.GeometricGuidance`:
    with a `segmenter`, each step's scored set is the reference's patch mask
    intersected with the prediction's own resegmented shell (downsampled to the
    same grid), so furniture the render adds is excused rather than pulled back
    toward the reference's bare surfaces. A batch element whose intersection is
    empty scores zero with no gradient, the same convention as
    `masked_depth_error`'s empty mask.

    Args:
        encoder: frozen and in eval, and never under `no_grad` — the gradient
            reaches the latent through it.
        reference_tokens: the photo's patch grid, frozen once: a constant, so
            construction rejects one still carrying gradient.
        patch_mask: the reference-side mask on the patch grid, as
            :func:`patch_positions_from_mask` builds it.
        segmenter: re-segments the predicted image at every step; `None` scores
            the reference's patch mask alone.
        erosion_radius: for the prediction's shell mask, required with
            `segmenter` and meaningless without it.
        min_coverage: the patch vote threshold the prediction's shell is
            downsampled under, matching the reference side's.
    """

    encoder: SpatialPatchEncoder
    reference_tokens: Float[Tensor, "1 tokens embedding"]
    patch_mask: Bool[Tensor, "1 tokens"]
    preprocessing: ClipPreprocessing
    segmenter: Segmenter | None = None
    erosion_radius: int | None = None
    min_coverage: float = 0.5

    def __post_init__(self) -> None:
        if self.reference_tokens.requires_grad:
            raise ValueError(
                "the reference photo's patch tokens are a constant of the walk: "
                "detach them before freezing them into a "
                "MaskedPatchReferenceGuidance"
            )
        if self.patch_mask.shape[-1] != self.reference_tokens.shape[1]:
            raise ValueError(
                f"patch_mask covers {self.patch_mask.shape[-1]} positions against "
                f"{self.reference_tokens.shape[1]} reference tokens: the mask and "
                f"the grid must come from the same encoder"
            )
        if not bool(self.patch_mask.any()):
            raise ValueError(
                "the reference's patch mask is empty: nothing would be scored and "
                "the energy would steer by nothing"
            )
        if (self.segmenter is None) != (self.erosion_radius is None):
            raise ValueError(
                "re-segmenting the prediction needs both a segmenter and an "
                "erosion_radius; pass neither to score the reference's patch "
                "mask alone"
            )

    def __call__(
        self, predicted_image: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, "batch"]:
        tokens = self.encoder(self.preprocessing(predicted_image))
        similarity = cosine_similarity(
            tokens.to(_ENERGY_DTYPE),
            self.reference_tokens.to(device=tokens.device, dtype=_ENERGY_DTYPE),
            dim=-1,
        )
        weight = self._scored_patches(predicted_image, tokens.shape[1]).to(
            _ENERGY_DTYPE
        )
        return -(similarity * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)

    def _scored_patches(
        self, predicted_image: Float[Tensor, "batch 3 height width"], tokens: int
    ) -> Bool[Tensor, "batch tokens"]:
        segmenter, erosion_radius = self.segmenter, self.erosion_radius
        if segmenter is None or erosion_radius is None:
            return self.patch_mask
        # Detached *and* under `no_grad`, exactly as `GeometricGuidance` reads
        # its per-step mask: the segmentation is a constant of the step, and the
        # undetached `predicted_image` above is what carries the gradient back.
        with torch.no_grad():
            segmentation = segmenter(predicted_image.detach())
        prediction_shell = patch_positions_from_mask(
            scorable_mask(segmentation, erosion_radius=erosion_radius),
            tokens=tokens,
            min_coverage=self.min_coverage,
        )
        return self.patch_mask & prediction_shell


@dataclass(frozen=True)
class MaskedPatchReferenceEnergies:
    """One loaded spatial tower, turning a (photo, mask) pair into the patch energy toward it.

    Not an :class:`ImageReferenceEnergyFactory`: the masked energy needs the
    reference's pixel mask beside the photo, and optionally the segmenter that
    keeps its per-step scored set honest.
    """

    encoder: SpatialPatchEncoder
    preprocessing: ClipPreprocessing
    module: Module

    def __call__(
        self,
        photo: Float[Tensor, "1 3 height width"],
        *,
        mask: Bool[Tensor, "1 height width"],
        segmenter: Segmenter | None = None,
        erosion_radius: int | None = None,
        min_coverage: float = 0.5,
    ) -> MaskedPatchReferenceGuidance:
        with torch.no_grad():
            reference = self.encoder(self.preprocessing(photo))
        return MaskedPatchReferenceGuidance(
            encoder=self.encoder,
            reference_tokens=reference.detach(),
            patch_mask=patch_positions_from_mask(
                mask, tokens=reference.shape[1], min_coverage=min_coverage
            ),
            preprocessing=self.preprocessing,
            segmenter=segmenter,
            erosion_radius=erosion_radius,
            min_coverage=min_coverage,
        )


@runtime_checkable
class ImageReferenceEnergyFactory(Protocol):
    """What every helper model reduces to: a reference photo in, the energy pulling renders toward it out.

    `module` is the weights behind the energies, exposed so a caller squeezing
    past a full card can park it off-device between walks; a factory must be
    called only while its module sits on the device the photos do.
    """

    @property
    def module(self) -> Module: ...

    def __call__(self, photo: Float[Tensor, "1 3 height width"]) -> Guidance: ...


@dataclass(frozen=True)
class ImageReferenceEnergies:
    """One loaded vision encoder, turning any reference photo into the energy that pulls renders toward it."""

    encoder: ClipImageEncoder
    preprocessing: ClipPreprocessing
    module: Module

    def __call__(
        self, photo: Float[Tensor, "1 3 height width"]
    ) -> ImageReferenceGuidance:
        with torch.no_grad():
            reference = self.encoder(self.preprocessing(photo))
        return ImageReferenceGuidance(
            encoder=self.encoder,
            reference_features=reference.detach(),
            preprocessing=self.preprocessing,
        )


@dataclass(frozen=True)
class PerceptualReferenceEnergies:
    """One loaded feature backbone, turning any reference photo into the perceptual energy toward it."""

    extractor: PerceptualFeatureExtractor
    preprocessing: ClipPreprocessing
    module: Module

    def __call__(
        self, photo: Float[Tensor, "1 3 height width"]
    ) -> PerceptualReferenceGuidance:
        with torch.no_grad():
            reference = self.extractor(self.preprocessing(photo))
        return PerceptualReferenceGuidance(
            extractor=self.extractor,
            reference_features=tuple(stage.detach() for stage in reference),
            preprocessing=self.preprocessing,
        )


def load_clip_image_reference_energies(
    checkpoint: str = DEFAULT_CLIP_CHECKPOINT,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    cache_dir: Path | None = None,
) -> ImageReferenceEnergies:
    """CLIP's image tower with its projection: caption-aligned appearance features.

    A run that also *scores* image similarity with CLIP must score it with a
    different checkpoint from this one, or it grades the sampler on the very
    model the sampler optimised.

    Args:
        cache_dir: Hub snapshot cache (`RoomgenSettings.hub_cache`); `None`
            follows the Hub client's own environment.
    """
    model: Module = CLIPVisionModelWithProjection.from_pretrained(
        checkpoint, cache_dir=cache_dir
    )
    module = _frozen(model, device=device, dtype=dtype)
    processor = CLIPImageProcessor.from_pretrained(checkpoint, cache_dir=cache_dir)
    return ImageReferenceEnergies(
        encoder=_TowerEncoder(module, _clip_embedding),
        preprocessing=_preprocessing(
            size=int(processor.crop_size["height"]),
            mean=list(processor.image_mean),
            std=list(processor.image_std),
            device=device,
            dtype=dtype,
        ),
        module=module,
    )


def load_dinov2_image_reference_energies(
    checkpoint: str = DEFAULT_DINOV2_CHECKPOINT,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    cache_dir: Path | None = None,
) -> ImageReferenceEnergies:
    """DINOv2's patch tokens, flattened as one embedding: self-supervised layout features.

    The cosine runs over the whole token grid rather than a pooled vector, so
    *where* things sit in the frame is part of the similarity — the structural
    contrast to CLIP's caption-aligned global embedding.
    """
    model: Module = AutoModel.from_pretrained(checkpoint, cache_dir=cache_dir)
    module = _frozen(model, device=device, dtype=dtype)
    processor = BitImageProcessor.from_pretrained(checkpoint, cache_dir=cache_dir)
    return ImageReferenceEnergies(
        encoder=_TowerEncoder(module, _dinov2_patch_tokens),
        preprocessing=_preprocessing(
            size=int(processor.crop_size["height"]),
            mean=list(processor.image_mean),
            std=list(processor.image_std),
            device=device,
            dtype=dtype,
        ),
        module=module,
    )


def load_dinov2_masked_patch_energies(
    checkpoint: str = DEFAULT_DINOV2_CHECKPOINT,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    cache_dir: Path | None = None,
) -> MaskedPatchReferenceEnergies:
    """DINOv2's patch grid kept spatial, for the mask-restricted patch cosine.

    The same tower `load_dinov2_image_reference_energies` flattens into one
    embedding, handed over un-flattened so the energy can score patch positions
    the reference's mask names and excuse the rest.
    """
    model: Module = AutoModel.from_pretrained(checkpoint, cache_dir=cache_dir)
    module = _frozen(model, device=device, dtype=dtype)
    processor = BitImageProcessor.from_pretrained(checkpoint, cache_dir=cache_dir)
    return MaskedPatchReferenceEnergies(
        encoder=_SpatialTowerEncoder(module, _dinov2_patch_grid),
        preprocessing=_preprocessing(
            size=int(processor.crop_size["height"]),
            mean=list(processor.image_mean),
            std=list(processor.image_std),
            device=device,
            dtype=dtype,
        ),
        module=module,
    )


def load_siglip_image_reference_energies(
    checkpoint: str = DEFAULT_SIGLIP_CHECKPOINT,
    *,
    subfolder: str = DEFAULT_SIGLIP_SUBFOLDER,
    processor_subfolder: str = DEFAULT_SIGLIP_PROCESSOR_SUBFOLDER,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    cache_dir: Path | None = None,
) -> ImageReferenceEnergies:
    """SigLIP's vision tower, pooled through its own attention head: sigmoid-contrastive appearance features.

    The defaults read the tower out of the FLUX.1-Redux-dev snapshot, so a cache
    that already holds Redux needs no download; a standalone SigLIP checkpoint
    passes empty subfolders instead.
    """
    model: Module = SiglipVisionModel.from_pretrained(
        checkpoint, subfolder=subfolder, cache_dir=cache_dir
    )
    module = _frozen(model, device=device, dtype=dtype)
    processor = SiglipImageProcessor.from_pretrained(
        checkpoint, subfolder=processor_subfolder, cache_dir=cache_dir
    )
    return ImageReferenceEnergies(
        encoder=_TowerEncoder(module, _siglip_embedding),
        preprocessing=_preprocessing(
            size=int(processor.size["height"]),
            mean=list(processor.image_mean),
            std=list(processor.image_std),
            device=device,
            dtype=dtype,
        ),
        module=module,
    )


def load_vgg_perceptual_energies(
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    cache_dir: Path | None = None,
    input_size: int = DEFAULT_VGG_INPUT_SIZE,
) -> PerceptualReferenceEnergies:
    """VGG16's five ReLU stacks as the perceptual pyramid LPIPS is built on.

    Args:
        cache_dir: where torchvision's weight download lands
            (`RoomgenSettings.hf_home`-adjacent rather than the home-directory
            hub default); `None` follows torch hub's own environment.
    """
    from torchvision.models import VGG16_Weights, vgg16  # noqa: PLC0415

    weights = VGG16_Weights.IMAGENET1K_V1
    network = vgg16()
    network.load_state_dict(
        weights.get_state_dict(
            progress=False,
            model_dir=None if cache_dir is None else str(cache_dir / "torchvision"),
            check_hash=True,
        )
    )
    features = cast(Sequential, _frozen(network.features, device=device, dtype=dtype))
    transforms = weights.transforms()
    return PerceptualReferenceEnergies(
        extractor=_VggFeaturePyramid(features),
        preprocessing=_preprocessing(
            size=input_size,
            mean=list(transforms.mean),
            std=list(transforms.std),
            device=device,
            dtype=dtype,
        ),
        module=features,
    )


# The indices just past relu1_2, relu2_2, relu3_3, relu4_3 and relu5_3 in
# torchvision's `vgg16().features` — the five stages LPIPS reads.
_VGG_STAGE_ENDS: Final = (4, 9, 16, 23, 30)


@dataclass(frozen=True)
class _VggFeaturePyramid:
    features: Sequential

    def __call__(
        self, pixel_values: Float[Tensor, "batch 3 size size"]
    ) -> tuple[Float[Tensor, "batch channels height width"], ...]:
        stages: list[Tensor] = []
        current: Tensor = pixel_values
        start = 0
        for end in _VGG_STAGE_ENDS:
            for index in range(start, end):
                current = self.features[index](current)
            stages.append(current)
            start = end
        return tuple(stages)


@dataclass(frozen=True)
class _TowerEncoder:
    module: Module
    read: Callable[[object], Tensor]

    def __call__(
        self, pixel_values: Float[Tensor, "batch 3 size size"]
    ) -> Float[Tensor, "batch embedding"]:
        return self.read(self.module(pixel_values=pixel_values))


@dataclass(frozen=True)
class _SpatialTowerEncoder:
    module: Module
    read: Callable[[object], Tensor]

    def __call__(
        self, pixel_values: Float[Tensor, "batch 3 size size"]
    ) -> Float[Tensor, "batch tokens embedding"]:
        return self.read(self.module(pixel_values=pixel_values))


def _clip_embedding(output: object) -> Tensor:
    embedding = getattr(output, "image_embeds", None)
    if isinstance(embedding, Tensor):
        return embedding
    raise TypeError(
        f"CLIP's vision tower returned no image_embeds, got {type(output).__name__}"
    )


def _siglip_embedding(output: object) -> Tensor:
    pooled = getattr(output, "pooler_output", None)
    if isinstance(pooled, Tensor):
        return pooled
    raise TypeError(
        f"SigLIP's vision tower returned no pooler_output, got {type(output).__name__}"
    )


def _dinov2_patch_tokens(output: object) -> Tensor:
    # Flattened into one embedding: position stays inside the similarity, but
    # the grid is gone and no mask can name a patch. See `_dinov2_patch_grid`.
    return _dinov2_patch_grid(output).flatten(1)


def _dinov2_patch_grid(output: object) -> Tensor:
    hidden = getattr(output, "last_hidden_state", None)
    if isinstance(hidden, Tensor):
        # Token 0 is the CLS summary; the patch grid behind it is what carries
        # the layout.
        return hidden[:, 1:, :]
    raise TypeError(
        f"DINOv2 returned no last_hidden_state, got {type(output).__name__}"
    )


def _frozen(model: Module, *, device: torch.device, dtype: torch.dtype) -> Module:
    return model.to(device=device, dtype=dtype).eval().requires_grad_(False)


def _preprocessing(
    *,
    size: int,
    mean: list[float],
    std: list[float],
    device: torch.device,
    dtype: torch.dtype,
) -> ClipPreprocessing:
    return ClipPreprocessing(
        size=size,
        image_mean=torch.tensor(mean, device=device, dtype=dtype).view(1, 3, 1, 1),
        image_std=torch.tensor(std, device=device, dtype=dtype).view(1, 3, 1, 1),
    )


def _unit_channels(
    features: Float[Tensor, "batch channels height width"],
) -> Float[Tensor, "batch channels height width"]:
    # LPIPS's unit normalisation: each spatial position's channel vector to unit
    # length, so a stage's magnitude does not decide its vote.
    norm = features.square().sum(dim=1, keepdim=True).sqrt()
    return features / (norm + 1e-10)
