from enum import Enum, auto


class StepLengthMethod(Enum):
    """
    StepLengthMethod enum is used to distinguish different strategies of choosing step's length for the given instance of the specified feature.
    """

    CONSTANT = auto()
    """
    Every instance of the specified feature changes its location with constant step length. Different feature type has different constant length value.
    """

    UNIFORM = auto()
    """
    Every instance of the specified feature changes its location with step length drawn from uniform distribution with value from `0` to `max_value`.
    Different feature type has different distribution maximum range.
    """

    GAUSS = auto()
    """
    Every instance of the specified feature changes its location with step length drawn from gauss distribution (gamma with scale=1) with the given mean value.
    Different feature type has different gauss distribution with different mean value.
    """

    NORMAL = auto()
    """
    Every instance of the specified feature changes its location with step length drawn from normal distribution with the given mean and standard deviation values.
    Different feature type has different normal distribution with different mean and standard deviation values.
    """


class StepAngleMethod(Enum):
    """
    StepAngleMethod enum is used to distinguish different strategies of choosing step's angle to the direction of destination point
    for the given instance of the specified feature.
    """

    UNIFORM = auto()
    """
    Step's angle of the given feature instance of the specified feature is chosen with uniform distribution in range from `-range` to `range` 
    """

    NORMAL = auto()
    """
    Step's angle of the given feature instance of the specified feature is chosen with normal distribution with mean equals `0` and the given standard deviation value.
    The standard deviation is defined by ratio of its value to the angle range value. If drawn values are exceeding expected range, then angle value is drawn again
    with uniform distribution as it is described in `UNIFORM` option.   
    """