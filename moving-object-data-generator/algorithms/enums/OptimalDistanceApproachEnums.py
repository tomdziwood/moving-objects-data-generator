from enum import Enum, auto


class MassMode(Enum):
    CONSTANT = auto()
    FEATURE_CONSTANT = auto()
    NORMAL = auto()


class VelocityMode(Enum):
    CONSTANT = auto()
    GAMMA = auto()
