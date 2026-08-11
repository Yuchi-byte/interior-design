"""The ADE20K-150 label table and its architectural-shell / contents partition.

The shell is the set of surfaces a renovation must leave where they are; the
contents are everything a renovation is free to move, replace, or delete.
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
# renovation is for. Outdoor structure seen through a window ("building",
# "sky", "tree") is excluded too — monocular depth behind glass is unreliable
# and the facade is not this room's geometry.
ARCHITECTURAL_CLASSES: Final[frozenset[int]] = frozenset(
    {
        0,  # wall
        3,  # floor
        5,  # ceiling
        8,  # windowpane
        14,  # door
        38,  # railing
        42,  # column
        53,  # stairs
        59,  # stairway
        95,  # bannister
        121,  # step
    }
)

# The complement, named because the partition is the domain statement: every
# class that is not shell is something a renovation may change, so it is
# excluded from the reward rather than merely unmentioned.
CONTENTS_CLASSES: Final[frozenset[int]] = (
    frozenset(range(NUM_CLASSES)) - ARCHITECTURAL_CLASSES
)


def label_name(index: int) -> str:
    """Human-readable name of an ADE20K-150 model output id."""
    if not 0 <= index < NUM_CLASSES:
        raise IndexError(
            f"ADE20K class index {index} out of range: expected 0..{NUM_CLASSES - 1}"
        )
    return ADE20K_LABELS[index]
