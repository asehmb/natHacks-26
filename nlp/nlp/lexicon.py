"""Vocabulary used by the rule parser."""

PICK_VERBS = {"pick", "grab", "take", "fetch", "grasp", "lift", "snatch", "collect", "get"}
PUT_VERBS = {"put", "place", "set", "drop", "lay", "insert", "stick"}
GIVE_VERBS = {"give", "hand", "pass", "bring"}
MOVE_VERBS = {"come", "go", "move", "step", "back", "scoot", "slide", "lean", "closer", "forward"}
ROTATE_VERBS = {"rotate", "turn", "spin"}
SHAKE_VERBS = {"shake", "nod", "shrug"}

HAND_WORDS = {"hand", "claw", "claws", "gripper", "fingers", "grip"}

CONTAINER_PREPS = {"in": "in", "into": "in", "inside": "in", "on": "on", "onto": "on"}
OBJECT_PREPS = {"on", "at", "from", "near", "next", "by", "of", "in", "inside", "beside"}

DETERMINERS = {"the", "that", "this", "those", "these", "a", "an", "my", "your", "some", "me", "to", "over"}
PRONOUNS = {"it", "them"}
PLACEHOLDER_NOUNS = {"one", "ones", "thing", "object"}

COLORS = {
    "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown",
    "black", "white", "gray", "silver",
}
COLOR_ALIASES = {"grey": "gray", "violet": "purple"}
SIZES = {"big", "large", "small", "tiny", "long", "short", "huge"}
POSITIONS = {
    "left", "right", "top", "bottom", "middle", "nearest", "closest",
    "farthest", "leftmost", "rightmost", "front",
}

DIRECTION_WORDS = {
    "closer": "forward", "forward": "forward", "forwards": "forward", "ahead": "forward",
    "nearer": "forward", "here": "forward",
    "back": "backward", "backward": "backward", "backwards": "backward",
    "away": "backward", "farther": "backward", "further": "backward",
    "left": "left", "right": "right",
    "up": "up", "higher": "up", "raise": "up",
    "down": "down", "lower": "down",
}
SMALL_WORDS = {"bit", "little", "slightly", "tiny", "touch", "smidge", "notch"}
LARGE_WORDS = {"lot", "far", "much", "way"}
UNIT_MM = {
    "mm": 1.0, "millimeter": 1.0, "millimeters": 1.0,
    "cm": 10.0, "centimeter": 10.0, "centimeters": 10.0,
    "inch": 25.4, "inches": 25.4, "in": 25.4,
    "m": 1000.0, "meter": 1000.0, "meters": 1000.0,
}

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
# number words are only converted after these tokens ("joint three", "return to zero")
NUMBER_CONTEXT = {"joint", "to", "by", "number", "axis"}

COMPOUNDS = {
    "screw driver": "screwdriver",
    "pick-up": "pickup",
    "allen key": "allenkey",
}

POLITE_PREFIX = (
    r"(?:hey|hi|hello|yo|howdy|ok|okay|robot|please|kindly|could you|can you|would you|will you|"
    r"i want you to|i need you to|i would like you to|i'd like you to|go ahead and|let's|lets)"
)
POLITE_SUFFIX = r"(?:please|for me|now|thanks|thank you)"

# Words the typo-corrector may snap unknown words onto.
VOCAB = (
    PICK_VERBS | PUT_VERBS | GIVE_VERBS | MOVE_VERBS | ROTATE_VERBS | SHAKE_VERBS | HAND_WORDS
    | COLORS | SIZES | POSITIONS | set(DIRECTION_WORDS) | set(CONTAINER_PREPS)
    | {"open", "close", "home", "zero", "joint", "down", "yes", "no", "maybe", "return", "reset",
       "degrees", "closer", "then", "and"}
)

START_VOCAB = (
    PICK_VERBS | PUT_VERBS | GIVE_VERBS | MOVE_VERBS | ROTATE_VERBS | SHAKE_VERBS
    | {"open", "close", "home", "return", "reset"}
)
REST_VOCAB = VOCAB - START_VOCAB | {"open", "close"}
