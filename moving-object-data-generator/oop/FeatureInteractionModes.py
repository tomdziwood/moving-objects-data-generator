from enum import Enum, auto


class IdenticalFeaturesInteractionMode(Enum):
    ATTRACT = auto()
    REPEL = auto()


class DifferentFeaturesInteractionMode(Enum):
    ATTRACT = auto()
    REPEL = auto()
    COLLOCATION_ATTRACT_OTHER_NEUTRAL = auto()
    COLLOCATION_ATTRACT_OTHER_REPEL = auto()
