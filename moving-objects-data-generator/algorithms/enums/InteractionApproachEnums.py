from enum import Enum, auto


class MassMethod(Enum):
    CONSTANT = auto()
    FEATURE_CONSTANT = auto()
    NORMAL = auto()


class VelocityMethod(Enum):
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
