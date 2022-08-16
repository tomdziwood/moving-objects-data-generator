from enum import Enum, auto


class MassMethod(Enum):
    CONSTANT = auto()
    FEATURE_CONSTANT = auto()
    NORMAL = auto()


class VelocityMethod(Enum):
    CONSTANT = auto()
    GAMMA = auto()
