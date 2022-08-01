from enum import Enum, auto


class MassMode(Enum):
    CONSTANT = auto()
    FEATURE_CONSTANT = auto()
    NORMAL = auto()


class VelocityMode(Enum):
    CONSTANT = auto()
    GAMMA = auto()


class IdenticalFeaturesInteractionMode(Enum):
    ATTRACT = auto()
    REPEL = auto()


class DifferentFeaturesInteractionMode(Enum):
    ATTRACT = auto()
    REPEL = auto()
    COLLOCATION_ATTRACT_OTHER_NEUTRAL = auto()
    COLLOCATION_ATTRACT_OTHER_REPEL = auto()
