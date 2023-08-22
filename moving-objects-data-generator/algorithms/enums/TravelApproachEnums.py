from enum import Enum, auto


class StepLengthMethod(Enum):
    """
    StepLengthMethod enum is used to distinguish different strategies of choosing step's length for the given instance of the specified feature type.
    """

    UNIFORM = auto()
    """
    Every instance of the specified feature type changes its location with step length drawn from the uniform distribution with value from ``min_value`` to ``max_value``.
    Boundary values are calculated based on the given mean (``step_length_mean``) and the given ratio of minimal value to the mean (``step_length_uniform_low_to_mean_ratio``).
    For details, see documentation of the `TravelApproachParameters` class. Different features types have different uniform distributions.
    """

    GAMMA = auto()
    """
    Every instance of the specified feature type changes its location with step length drawn from a gamma distribution with the given shape and the scale equals ``1.0``.
    This distribution is an extension from the integer to the real domain of the Poisson distribution with the lambda equals the given shape.
    Different features types have different gauss distributions with different mean value.
    """

    NORMAL = auto()
    """
    Every instance of the specified feature type changes its location with step length drawn from a normal distribution with the given mean and standard deviation values.
    Different features types have different normal distributions with different mean and standard deviation values.
    """


class StepAngleMethod(Enum):
    """
    StepAngleMethod enum is used to distinguish different strategies of choosing step's angle to the direction of destination point
    for the given instance of the specified feature type.
    """

    UNIFORM = auto()
    """
    Step's angle of the given instance of the specified feature is chosen with the uniform distribution in range from `-range` to `range`
    """

    NORMAL = auto()
    """
    Step's angle of the given instance of the specified feature is chosen with a normal distribution with the mean equals `0` and the given standard deviation value.
    The standard deviation is defined by ratio of its value to the angle range value. If drawn values are exceeding expected range, then angle value is drawn again
    with a uniform distribution as it is described in the `UNIFORM` option.
    """