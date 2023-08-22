from enum import Enum, auto


class MassMethod(Enum):
    """
    MassMethod enum is used to distinguish different strategies of choosing the mass for the given instance of the specified feature type.
    """

    CONSTANT = auto()
    """
    Every feature instance has constant mass, which is equal to the `mass_mean` parameter value.
    """

    FEATURE_CONSTANT = auto()
    """
    Every instance of the specified feature type has constant mass, which is equal to the feature's constant mass value. The constant mass value of the specified feature type
    is drawn from the gamma distribution with the given shape of `mass_mean` parameter value and the scale equals ``1.0``. This distribution is an extension from the integer
    to the real domain of the Poisson distribution with the lambda equals the given shape.
    """

    NORMAL = auto()
    """
    Every instance of the specified feature type has a mass drawn from a normal distribution with the given mean and standard deviation values.
    Different features types have different normal distributions with different mean and standard deviation values.
    """


class VelocityMethod(Enum):
    """
    VelocityMethod enum is used to distinguish different strategies of choosing the initial velocity for a feature instance.
    """

    CONSTANT = auto()
    """
    Every feature instance has a constant initial velocity, which is equal to the `velocity_mean` parameter value. When the velocity is different than ``0.0``,
    the velocity of the given feature instance is oriented in a random direction.
    """

    GAMMA = auto()
    """
    Every feature instance has an initial velocity drawn from the gamma distribution with the given shape of `velocity_mean` parameter value and the scale equals ``1.0``.
    This distribution is an extension from the integer to the real domain of the Poisson distribution with the lambda equals the given shape.
    The velocity of features instances is oriented in a random direction.
    """


class IdenticalFeaturesInteractionMode(Enum):
    """
    IdenticalFeaturesInteractionMode enum is used to distinguish different strategies of interaction between instances of the identical feature type.
    """

    ATTRACT = auto()
    """
    Instances of the identical feature type attract each other.
    """

    REPEL = auto()
    """
    Instances of the identical feature type repel each other.
    """


class DifferentFeaturesInteractionMode(Enum):
    """
    DifferentFeaturesInteractionMode enum is used to distinguish different strategies of interaction between instances of different features types.
    """

    ATTRACT = auto()
    """
    Instances of different features types attract each other.
    """

    REPEL = auto()
    """
    Instances of different features types repel each other.
    """

    COLLOCATION_ATTRACT_OTHER_NEUTRAL = auto()
    """
    Instances of different features types attract each other, only when these features types are ment to be part of co-location pattern.
    Otherwise, there is no interaction between these features types.
    """

    COLLOCATION_ATTRACT_OTHER_REPEL = auto()
    """
    Instances of different features types attract each other, only when these features types are ment to be part of co-location pattern.
    Otherwise, instances of these features types repel each other.
    """

