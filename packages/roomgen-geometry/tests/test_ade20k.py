from roomgen_geometry.segmentation.ade20k import (
    ADE20K_LABELS,
    ARCHITECTURAL_CLASSES,
    CONTENTS_CLASSES,
    FILL_CLASSES,
    NUM_CLASSES,
    REMOVABLE_CLASSES,
    SOFT_FURNISHING_CLASSES,
    STRUCTURAL_CLASSES,
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


def test_the_openings_are_shell_and_never_contents() -> None:
    # Owner decision: a window and a door are openings a renovation must leave
    # where they are, so their depth is scored rather than masked out — whatever
    # a monocular network reads on glass (the pane or the view beyond), a render
    # that kept the window reads the same thing, and one that walled it over is
    # caught. Contents membership would instead *excuse* a prediction's glazing
    # under `subtract_contents`.
    windowpane = ADE20K_LABELS.index("windowpane")
    door = ADE20K_LABELS.index("door")

    assert windowpane in ARCHITECTURAL_CLASSES
    assert door in ARCHITECTURAL_CLASSES
    assert windowpane not in CONTENTS_CLASSES
    assert door not in CONTENTS_CLASSES


def test_the_furniture_a_renovation_replaces_is_named_contents() -> None:
    assert {
        ADE20K_LABELS.index(name) for name in ("sofa", "bed", "table", "chair", "lamp")
    } <= CONTENTS_CLASSES


def test_the_structural_classes_are_the_shell_minus_its_soft_furnishings() -> None:
    # The class term's set: what a renovation may not turn into something else.
    # Depth and class part ways exactly at the soft furnishings — a curtain's
    # depth is scored, its identity is the renovation's to change.
    assert STRUCTURAL_CLASSES == ARCHITECTURAL_CLASSES - SOFT_FURNISHING_CLASSES
    assert ADE20K_LABELS.index("windowpane") in STRUCTURAL_CLASSES
    assert ADE20K_LABELS.index("door") in STRUCTURAL_CLASSES


def test_the_soft_furnishings_are_scored_stripped_and_never_frozen() -> None:
    # Owner decision: a rug reads the floor's depth and a curtain the window's,
    # so masking them out was surrendering valid comparison area. They are
    # scored as shell, still cut by a strip, never held to their photographed
    # identity, and never an acceptable inpainting fill.
    assert {
        ADE20K_LABELS.index(name) for name in ("curtain", "rug")
    } == SOFT_FURNISHING_CLASSES
    assert SOFT_FURNISHING_CLASSES <= ARCHITECTURAL_CLASSES
    assert not SOFT_FURNISHING_CLASSES & CONTENTS_CLASSES
    assert not SOFT_FURNISHING_CLASSES & STRUCTURAL_CLASSES
    assert not SOFT_FURNISHING_CLASSES & FILL_CLASSES
    assert REMOVABLE_CLASSES == CONTENTS_CLASSES | SOFT_FURNISHING_CLASSES


def test_the_fill_whitelist_is_the_five_bare_surfaces_and_nothing_else() -> None:
    # Owner decision: an inpainted hole may only be filled with wall, floor,
    # ceiling, door or windowpane. A proper subset of the shell on purpose —
    # railing, column and stairs keep their depth in the reward where the
    # photograph shows them, but a strip that grows one inside a hole has
    # invented structure the room never had.
    assert {
        ADE20K_LABELS.index(name)
        for name in ("wall", "floor", "ceiling", "windowpane", "door")
    } == FILL_CLASSES
    assert FILL_CLASSES < ARCHITECTURAL_CLASSES


def test_nothing_a_renovation_may_replace_is_held_to_its_photographed_class() -> None:
    # A class that were both would be held to its identity and excused from that
    # same check by any render that put furniture over it: a term doing nothing.
    assert not STRUCTURAL_CLASSES & CONTENTS_CLASSES
    assert not STRUCTURAL_CLASSES - set(range(NUM_CLASSES))
