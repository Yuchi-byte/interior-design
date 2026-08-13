"""The ADE20K-150 label table and its two affirmative class lists.

The shell is the set of surfaces a renovation must leave where they are; the
contents are the objects a renovation is free to move, replace, or delete. Both
are named class by class and neither is the other's complement: the ids in
neither list — outdoors, people, stuff — are the ones a mask has to decide about
without being told, which is the whole reason to spell the two lists out.
"""

from typing import Final

# The id -> name table of the 150 semantic classes the ADE20K-trained
# segmentation heads emit, cross-checked against two independent sources that
# agree entry for entry (upstream ships id 7 as "bed " with a trailing space;
# only that whitespace is normalised here):
#   https://huggingface.co/datasets/huggingface/label-files/raw/main/ade20k-id2label.json
#   https://huggingface.co/tue-mps/ade20k_semantic_eomt_large_512/raw/main/config.json
# Note the indexing: a model's `id2label` is 0-based with 0 == "wall", while the
# raw scene_parse_150 annotation PNGs are 1-based with 0 reserved for
# "unlabelled". Every id here is a *model output* id, which is what `Segmenter`
# returns.
ADE20K_LABELS: Final[tuple[str, ...]] = (
    "wall",
    "building",
    "sky",
    "floor",
    "tree",
    "ceiling",
    "road",
    "bed",
    "windowpane",
    "grass",
    "cabinet",
    "sidewalk",
    "person",
    "earth",
    "door",
    "table",
    "mountain",
    "plant",
    "curtain",
    "chair",
    "car",
    "water",
    "painting",
    "sofa",
    "shelf",
    "house",
    "sea",
    "mirror",
    "rug",
    "field",
    "armchair",
    "seat",
    "fence",
    "desk",
    "rock",
    "wardrobe",
    "lamp",
    "bathtub",
    "railing",
    "cushion",
    "base",
    "box",
    "column",
    "signboard",
    "chest of drawers",
    "counter",
    "sand",
    "sink",
    "skyscraper",
    "fireplace",
    "refrigerator",
    "grandstand",
    "path",
    "stairs",
    "runway",
    "case",
    "pool table",
    "pillow",
    "screen door",
    "stairway",
    "river",
    "bridge",
    "bookcase",
    "blind",
    "coffee table",
    "toilet",
    "flower",
    "book",
    "hill",
    "bench",
    "countertop",
    "stove",
    "palm",
    "kitchen island",
    "computer",
    "swivel chair",
    "boat",
    "bar",
    "arcade machine",
    "hovel",
    "bus",
    "towel",
    "light",
    "truck",
    "tower",
    "chandelier",
    "awning",
    "streetlight",
    "booth",
    "television receiver",
    "airplane",
    "dirt track",
    "apparel",
    "pole",
    "land",
    "bannister",
    "escalator",
    "ottoman",
    "bottle",
    "buffet",
    "poster",
    "stage",
    "van",
    "ship",
    "fountain",
    "conveyer belt",
    "canopy",
    "washer",
    "plaything",
    "swimming pool",
    "stool",
    "barrel",
    "basket",
    "waterfall",
    "tent",
    "bag",
    "minibike",
    "cradle",
    "oven",
    "ball",
    "food",
    "step",
    "tank",
    "trade name",
    "microwave",
    "pot",
    "animal",
    "bicycle",
    "lake",
    "dishwasher",
    "screen",
    "blanket",
    "sculpture",
    "hood",
    "sconce",
    "vase",
    "traffic light",
    "tray",
    "ashcan",
    "fan",
    "pier",
    "crt screen",
    "plate",
    "monitor",
    "bulletin board",
    "shower",
    "radiator",
    "glass",
    "clock",
    "flag",
)

NUM_CLASSES: Final[int] = len(ADE20K_LABELS)

# The load-bearing surfaces and fixed openings of a room, the geometry a
# renovation is not allowed to move. Membership is deliberately narrow: a class
# admitted here freezes its depth, so a wrongly admitted movable object (a rug,
# a fireplace surround, a shower enclosure) would forbid exactly the change the
# renovation is for. Everything seen *through* a window is excluded for the
# opposite reason — "building", "sky", "tree", and the "windowpane" itself:
# monocular depth answers a glazed pixel with the depth of the outdoor scene, so
# scoring it scores the view rather than the room. The opening is still held in
# place by the wall around it, and the boundary erosion drops the band either
# side of the glazing, so the frame transition is never scored either.
# Windowpane is deliberately in neither list rather than moved to contents: a
# window is not furniture a renovation adds, and contents membership would
# *exempt* a prediction's glazing under `subtract_contents` instead of merely
# declining to score it.
ARCHITECTURAL_CLASSES: Final[frozenset[int]] = frozenset(
    {
        0,  # wall
        3,  # floor
        5,  # ceiling
        14,  # door
        38,  # railing
        42,  # column
        53,  # stairs
        59,  # stairway
        95,  # bannister
        121,  # step
    }
)


def _named_contents(ids: set[int]) -> frozenset[int]:
    """The contents list, checked against the shell it must not overlap.

    The two lists are hand-curated and mutually exclusive by hand, so the
    conditions that make `subtract_contents` mean anything — every id real, both
    halves non-empty, no id claimed twice — are asserted where they are stated
    rather than trusted to a reviewer.
    """
    contents = frozenset(ids)
    if not contents or not ARCHITECTURAL_CLASSES:
        raise ValueError("neither half of the class partition may be empty")
    unknown = (contents | ARCHITECTURAL_CLASSES) - frozenset(range(NUM_CLASSES))
    if unknown:
        raise ValueError(f"class ids outside the ADE20K-150 table: {sorted(unknown)}")
    both = contents & ARCHITECTURAL_CLASSES
    if both:
        raise ValueError(f"classes claimed as shell and contents: {sorted(both)}")
    return contents


# The objects a renovation legitimately adds, moves, or replaces: furniture,
# furnishings, appliances, decor, lighting, textiles, potted plants,
# electronics. Named affirmatively, never as the complement of the shell —
# `subtract_contents` excuses a pixel from the reward only where the segmenter
# commits to one of these, so a class in neither list (windowpane, sky, tree,
# person, water, road: the glass, what it shows, or what walks through the
# frame) is never excused here. Borderline calls:
#   curtain, blind: hang off the wall and swing clear of it, so a new window
#     dressing is not held to the old one's depth.
#   painting, poster, mirror, bulletin board: on the wall plane, but depth
#     reads the canvas, or the reflected room, not the wall behind it.
#   fireplace, bathtub, shower, toilet, sink, counter, countertop, kitchen
#     island, radiator: plumbed or fixed, yet exactly what a renovation rips
#     out — freezing them forbids the change the reward exists to allow.
#   plant, flower, pot: indoors these are pots standing on the floor; the
#     greenery a window shows is tree/grass/palm, which is in neither list.
#   fan, hood, light, chandelier, sconce: mounted, but swapped per fit-out.
CONTENTS_CLASSES: Final[frozenset[int]] = _named_contents(
    {
        7,  # bed
        10,  # cabinet
        15,  # table
        17,  # plant
        18,  # curtain
        19,  # chair
        22,  # painting
        23,  # sofa
        24,  # shelf
        27,  # mirror
        28,  # rug
        30,  # armchair
        31,  # seat
        33,  # desk
        35,  # wardrobe
        36,  # lamp
        37,  # bathtub
        39,  # cushion
        41,  # box
        44,  # chest of drawers
        45,  # counter
        47,  # sink
        49,  # fireplace
        50,  # refrigerator
        55,  # case
        56,  # pool table
        57,  # pillow
        62,  # bookcase
        63,  # blind
        64,  # coffee table
        65,  # toilet
        66,  # flower
        67,  # book
        69,  # bench
        70,  # countertop
        71,  # stove
        73,  # kitchen island
        74,  # computer
        75,  # swivel chair
        77,  # bar
        78,  # arcade machine
        81,  # towel
        82,  # light
        85,  # chandelier
        89,  # television receiver
        92,  # apparel
        97,  # ottoman
        98,  # bottle
        99,  # buffet
        100,  # poster
        107,  # washer
        108,  # plaything
        110,  # stool
        112,  # basket
        115,  # bag
        117,  # cradle
        118,  # oven
        119,  # ball
        120,  # food
        124,  # microwave
        125,  # pot
        129,  # dishwasher
        130,  # screen
        131,  # blanket
        132,  # sculpture
        133,  # hood
        134,  # sconce
        135,  # vase
        137,  # tray
        138,  # ashcan
        139,  # fan
        141,  # crt screen
        142,  # plate
        143,  # monitor
        144,  # bulletin board
        145,  # shower
        146,  # radiator
        147,  # glass
        148,  # clock
    }
)


def _disjoint_from_contents(ids: frozenset[int]) -> frozenset[int]:
    """The structural set, checked against the contents it must not overlap.

    Asserted where it is stated, like the partition above: a class that was both
    structural and contents would be held to its photographed identity and
    excused from that same check by any prediction rendering furniture over it,
    which reads as a term that quietly does nothing.
    """
    both = ids & CONTENTS_CLASSES
    if both:
        raise ValueError(f"classes claimed as structural and contents: {sorted(both)}")
    return ids


# The classes whose *identity* a renovation may not change — the shell, plus the
# glazing. Windowpane is here and deliberately not in `ARCHITECTURAL_CLASSES`,
# because the two lists answer different questions about the same pixel. Depth
# behind glass reads the outdoor view, so scoring a glazed pixel's *depth*
# scores the view rather than the room; its *class* is still the room's, and
# "there is still a window here" is exactly the constraint depth cannot express
# and a renovation must not break. A door is the same case in weaker form: depth
# pins the plane the door sits in, which a flat wall painted over the doorway
# satisfies exactly, while its class does not.
STRUCTURAL_CLASSES: Final[frozenset[int]] = _disjoint_from_contents(
    ARCHITECTURAL_CLASSES | {ADE20K_LABELS.index("windowpane")}
)
