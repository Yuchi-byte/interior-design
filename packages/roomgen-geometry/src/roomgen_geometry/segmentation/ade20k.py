"""The ADE20K-150 label table and its affirmative class lists.

The shell is the set of surfaces whose depth a renovation must leave where it
is; the contents are the objects a renovation is free to move, replace, or
delete. Both are named class by class and neither is the other's complement:
the ids in neither list — outdoors, people, stuff — are the ones a mask has to
decide about without being told, which is the whole reason to spell the lists
out. The derived sets answer narrower questions: what keeps its identity
(structural), what a strip removes (removable), what an inpainted hole may be
filled with (fill).
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
# admitted here freezes its depth, so a wrongly admitted movable object (a
# fireplace surround, a shower enclosure) would forbid exactly the change the
# renovation is for. Everything seen *through* a window is excluded —
# "building", "sky", "tree": monocular depth answers those pixels with the
# depth of the outdoor scene, so scoring them scores the view rather than the
# room. The windowpane itself is shell, by owner decision: a window and a door
# are openings a renovation must leave where they are, so their depth is scored
# rather than masked out. What a monocular network reads on glass is ambiguous —
# the pane, or the scene beyond it — and that ambiguity is accepted: whatever
# the network reads there, it must read the *same* thing on a render that kept
# the window, and a render that walled it over is caught either way. The two
# soft furnishings (curtain, rug) are shell by the same owner's later decision:
# their pixels read the depth of the window and the floor they dress, and
# masking them out was shrinking the valid comparison area for no geometric
# gain. The accepted trade: a curtain hangs proud of the window plane and a
# renovation may legitimately add or remove curtains and rugs, so reference and
# render can honestly disagree on those pixels — the larger valid area is worth
# that noise, by owner decision.
ARCHITECTURAL_CLASSES: Final[frozenset[int]] = frozenset(
    {
        0,  # wall
        3,  # floor
        5,  # ceiling
        8,  # windowpane
        14,  # door
        18,  # curtain
        28,  # rug
        38,  # railing
        42,  # column
        53,  # stairs
        59,  # stairway
        95,  # bannister
        121,  # step
    }
)


# The soft furnishings inside the shell: reward-valid depth (their pixels read
# the floor and window planes they dress) yet still something a renovation adds
# or removes at will — so a strip must keep cutting them, their identity is
# never frozen by the class term, and an inpainted hole may not be filled with
# them. Every set below that needs one of those three answers derives it from
# this pair rather than restating it.
SOFT_FURNISHING_CLASSES: Final[frozenset[int]] = frozenset(
    {
        18,  # curtain
        28,  # rug
    }
)


def _shell_subset(ids: set[int]) -> frozenset[int]:
    """A whitelist that must stay inside the shell it narrows, asserted where stated."""
    subset = frozenset(ids)
    outside = subset - ARCHITECTURAL_CLASSES
    if outside:
        raise ValueError(f"fill classes that are not shell: {sorted(outside)}")
    return subset


# What an inpainted hole may be filled *with*, by owner decision: the surfaces a
# bare room is made of — wall, floor, ceiling — and its two fixed openings, and
# nothing else. Deliberately stricter than `ARCHITECTURAL_CLASSES`: a railing, a
# column or a staircase is legitimate shell for the *reward* — where the
# photograph shows one, its depth is held — but a strip that grows one inside a
# hole has invented structure the room never had, so none of them is an
# acceptable fill. The whitelist reads the segmenter's labels, which is its
# honest limit: a fill the segmenter MISlabels as wall sails through it, and
# only the inpainter's negative prompt guards that side.
FILL_CLASSES: Final[frozenset[int]] = _shell_subset(
    {
        0,  # wall
        3,  # floor
        5,  # ceiling
        8,  # windowpane
        14,  # door
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
# commits to one of these, so a class in neither list (sky, tree, person,
# water, road: what a window shows, or what walks through the frame) is never
# excused here. Curtain and rug are deliberately absent: they moved to the
# shell (see `SOFT_FURNISHING_CLASSES`) so their depth is scored rather than
# excused, while `REMOVABLE_CLASSES` below keeps them strippable. Borderline
# calls:
#   blind: hangs off the wall and swings clear of it, so a new window
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
        19,  # chair
        22,  # painting
        23,  # sofa
        24,  # shelf
        27,  # mirror
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


# The classes whose *identity* a renovation may not change: the shell minus its
# soft furnishings. The two lists answer different questions about a pixel —
# where it is (depth) versus what it is (class) — and they no longer coincide:
# a curtain or a rug is scored for depth (it reads the plane it dresses) but a
# renovation may re-dress a window or bare a floor, so freezing their *class*
# would forbid exactly that change. The class term still earns its keep on the
# openings: depth pins the plane a door or a window sits in, which a flat wall
# painted over the opening satisfies almost exactly, while its class does not.
STRUCTURAL_CLASSES: Final[frozenset[int]] = _disjoint_from_contents(
    ARCHITECTURAL_CLASSES - SOFT_FURNISHING_CLASSES
)


# What a strip removes: everything a renovation replaces, plus the soft
# furnishings that are reward-valid shell yet must still leave an emptied room —
# a stripped room has no curtains and no rugs, whatever the reward scores them
# as. Kept beside the two lists it is built from so the three questions a
# curtain raises (score it? cut it? accept it as fill?) each have exactly one
# written answer: yes, yes, no.
REMOVABLE_CLASSES: Final[frozenset[int]] = CONTENTS_CLASSES | SOFT_FURNISHING_CLASSES
