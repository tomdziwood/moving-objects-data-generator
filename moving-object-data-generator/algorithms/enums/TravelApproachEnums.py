from enum import Enum, auto


class StepLengthMethod(Enum):
    """
    StepLengthMethod enum is used to distinguish different strategies of choosing step's length for the given instance of the specified feature.
    """

    UNIFORM = auto()
    """
    Every instance of the specified feature type changes its location with step length drawn from uniform distribution with value from ``min_value`` to ``max_value``.
    Boundary values are calculated based on the given mean (``step_length_mean``) and the given ratio of minimal value to the mean (``step_length_uniform_low_to_mean_ratio``).
    For details, see documentation of the `TravelApproachParameters` class. Different feature types has different uniform distribution.
    """

    GAMMA = auto()
    """
    Every instance of the specified feature type changes its location with step length drawn from gamma distribution with the given shape and the scale equals ``1.0``.
    This distribution is an extension from integer to real domain of the Poisson distribution with lambda equals the given shape.
    Different feature types has different gauss distribution with different mean value.
    """

    NORMAL = auto()
    """
    Every instance of the specified feature type changes its location with step length drawn from normal distribution with the given mean and standard deviation values.
    Different feature types has different normal distribution with different mean and standard deviation values.
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