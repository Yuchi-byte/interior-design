from roomgen_geometry.segmentation.ade20k import (
    ADE20K_LABELS,
    ARCHITECTURAL_CLASSES,
    CONTENTS_CLASSES,
    NUM_CLASSES,
)


def test_the_label_table_is_the_150_classes_the_models_emit() -> None:
    assert NUM_CLASSES == 150
    assert ADE20K_LABELS[0] == "wall"
    assert ADE20K_LABELS[149] == "flag"


def test_both_class_lists_name_real_classes_and_neither_is_empty() -> None:
    assert ARCHITECTURAL_CLASSES
    assert CONTENTS_CLASSES
    assert not (ARCHITECTURAL_CLASSES | CONTENTS_CLASSES) - set(range(NUM_CLASSES))


def test_no_class_is_both_shell_and_contents() -> None:
    assert not ARCHITECTURAL_CLASSES & CONTENTS_CLASSES


def test_the_two_lists_deliberately_leave_the_rest_of_the_table_unclaimed() -> None:
    # Not a partition, on purpose: an outdoor or stuff class is neither geometry
    # a renovation must preserve nor an object it may replace, and the gap
    # between the lists is exactly where the two mask modes disagree.
    unclaimed = set(range(NUM_CLASSES)) - ARCHITECTURAL_CLASSES - CONTENTS_CLASSES

    assert unclaimed
    assert {
        ADE20K_LABELS.index(name) for name in ("sky", "tree", "person")
    } <= unclaimed


def test_the_glass_a_window_is_made_of_is_scored_as_neither() -> None:
    # Depth behind glass is the outdoor scene's, not the room's, so the glazing
    # is not shell; it is not contents either, or a prediction's window would
    # excuse itself from the reward under `subtract_contents`.
    windowpane = ADE20K_LABELS.index("windowpane")

    assert windowpane not in ARCHITECTURAL_CLASSES
    assert windowpane not in CONTENTS_CLASSES


def test_the_furniture_a_renovation_replaces_is_named_contents() -> None:
    assert {
        ADE20K_LABELS.index(name)
        for name in ("sofa", "bed", "table", "chair", "lamp", "rug", "curtain")
    } <= CONTENTS_CLASSES
