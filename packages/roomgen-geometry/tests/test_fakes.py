import torch
from torch import Generator, Tensor

from roomgen_fakes import (
    FakeClassLogitSegmenter,
    FakeDepthEstimator,
    FakeSegmenter,
    assert_gradient_is_finite_and_nonzero,
)
from roomgen_fakes.segmentation import BRIGHTNESS_BY_CLASS
from roomgen_geometry.alignment import align_scale_shift
from roomgen_geometry.depth.base import DepthEstimator
from roomgen_geometry.segmentation.ade20k import (
    ADE20K_LABELS,
    ARCHITECTURAL_CLASSES,
    CONTENTS_CLASSES,
    NUM_CLASSES,
)
from roomgen_geometry.segmentation.base import ClassLogitSegmenter, Segmenter


def test_fake_depth_estimator_returns_the_disparity_shape_the_protocol_promises(
    fake_depth_estimator: FakeDepthEstimator, generator: Generator
) -> None:
    images = torch.rand((3, 3, 32, 48), generator=generator)

    disparity = fake_depth_estimator(images)

    assert isinstance(fake_depth_estimator, DepthEstimator)
    assert disparity.shape == (3, 32, 48)
    assert disparity.dtype == images.dtype


def test_fake_disparity_is_larger_toward_the_bottom_of_the_frame(
    fake_depth_estimator: FakeDepthEstimator, room_photo: Tensor
) -> None:
    disparity = fake_depth_estimator(room_photo)

    assert float(disparity[:, -1].mean()) > float(disparity[:, 0].mean())


def test_fake_depth_gradient_reaches_every_input_pixel(
    fake_depth_estimator: FakeDepthEstimator, room_photo: Tensor
) -> None:
    photo = room_photo.requires_grad_(True)

    gradient = assert_gradient_is_finite_and_nonzero(
        fake_depth_estimator(photo).sum(), photo
    )

    assert (gradient.abs() > 0.0).all()


def test_two_fake_backends_differ_by_exactly_the_affine_freedom_alignment_removes(
    room_photo: Tensor,
) -> None:
    reference = FakeDepthEstimator()(room_photo)
    rescaled = FakeDepthEstimator(scale=3.0, shift=-2.0)(room_photo)
    mask = torch.ones_like(reference, dtype=torch.bool)

    fit = align_scale_shift(reference, rescaled, mask)

    torch.testing.assert_close(fit.scale, torch.tensor([3.0]))
    torch.testing.assert_close(fit.shift, torch.tensor([-2.0]))


def test_fake_segmenter_labels_a_room_with_both_shell_and_contents(
    fake_segmenter: FakeSegmenter, room_photo: Tensor
) -> None:
    labels = fake_segmenter(room_photo)

    assert isinstance(fake_segmenter, Segmenter)
    assert labels.shape == (1, 64, 64)
    assert labels.dtype == torch.int64
    present = {int(label) for label in labels.unique()}
    assert present <= set(range(NUM_CLASSES))
    assert present & ARCHITECTURAL_CLASSES
    assert present & CONTENTS_CLASSES


def test_fake_segmenter_puts_the_ceiling_above_the_floor(
    fake_segmenter: FakeSegmenter, room_photo: Tensor
) -> None:
    labels = fake_segmenter(room_photo)

    top, bottom = int(labels[0, 0, 0]), int(labels[0, -1, 0])
    assert top in ARCHITECTURAL_CLASSES
    assert bottom in ARCHITECTURAL_CLASSES
    assert top != bottom


def test_fake_segmenter_output_is_independent_of_pixel_values(
    fake_segmenter: FakeSegmenter, room_photo: Tensor, generator: Generator
) -> None:
    other_photo = torch.rand(room_photo.shape, generator=generator)

    assert torch.equal(fake_segmenter(room_photo), fake_segmenter(other_photo))


def _painted(image: Tensor, brightness: float, region: tuple[slice, slice]) -> Tensor:
    changed = image.clone()
    changed[:, :, region[0], region[1]] = brightness
    return changed


def test_fake_class_logits_carry_the_shape_and_the_protocol_the_reward_needs(
    fake_class_logit_segmenter: FakeClassLogitSegmenter, room_photo: Tensor
) -> None:
    logits = fake_class_logit_segmenter.class_logits(room_photo)

    assert isinstance(fake_class_logit_segmenter, ClassLogitSegmenter)
    assert isinstance(fake_class_logit_segmenter, Segmenter)
    assert logits.shape == (1, NUM_CLASSES, 64, 64)
    assert float(logits.min()) >= 0.0


def test_fake_class_ids_are_the_argmax_of_the_very_scores_they_gate(
    fake_class_logit_segmenter: FakeClassLogitSegmenter, room_photo: Tensor
) -> None:
    # The property EoMT has and this fake must copy: the mask deciding which
    # pixels are scored cannot disagree with the scores it gates.
    labels = fake_class_logit_segmenter(room_photo)

    assert labels.dtype == torch.int64
    assert not labels.requires_grad
    assert torch.equal(
        labels, fake_class_logit_segmenter.class_logits(room_photo).argmax(dim=1)
    )


def test_fake_class_logits_label_a_painted_region_by_its_brightness(
    fake_class_logit_segmenter: FakeClassLogitSegmenter, room_photo: Tensor
) -> None:
    region = (slice(8, 24), slice(8, 24))

    for label, brightness in BRIGHTNESS_BY_CLASS.items():
        painted = _painted(room_photo, brightness, region)

        labels = fake_class_logit_segmenter(painted)

        assert set(labels[:, region[0], region[1]].unique().tolist()) == {label}


def test_fake_class_logits_gradient_reaches_the_pixels_that_moved(
    fake_class_logit_segmenter: FakeClassLogitSegmenter, room_photo: Tensor
) -> None:
    photo = room_photo.clone().requires_grad_(True)
    windowpane = ADE20K_LABELS.index("windowpane")

    gradient = assert_gradient_is_finite_and_nonzero(
        fake_class_logit_segmenter.class_logits(photo)[:, windowpane].sum(), photo
    )

    assert gradient.shape == photo.shape
