import numpy as np

from oop.BasicParameters import BasicParameters
from oop.TravelApproachEnums import StepLengthMethod, StepAngleMethod


class TravelApproachParameters(BasicParameters):
    """
    The class represents set of parameters used by the `SpatioTemporalTravelApproachGenerator` class of a spatio-temporal data generator.
    """

    def __init__(
            self,
            step_length_mean: float = 10.0,
            step_length_method: StepLengthMethod = StepLengthMethod.UNIFORM,
            step_length_uniform_low_to_mean_ratio: float = 1.0,
            step_length_normal_std_ratio: float = 1 / 3,
            step_angle_range_mean: float = np.pi / 9,
            step_angle_range_limit: float = np.pi / 2,
            step_angle_method: StepAngleMethod = StepAngleMethod.UNIFORM,
            step_angle_normal_std_ratio: float = 1 / 3,
            waiting_time_frames: int = np.inf,
            **kwargs):
        """
        Construct object which holds all the required parameters of the `SpatioTemporalTravelApproachGenerator` class of a spatio-temporal data generator.

        Parameters
        ----------
        step_length_mean : float
            Mean value of the step's length of displacement of each feature type is determined by the gamma distribution with shape equals ``step_length_mean``
            and scale equals ``1.0``. This distribution is an extension from integer to real domain of the Poisson distribution with lambda equals ``step_length_mean``.

        step_length_method : StepLengthMethod
            Enum value is used to distinguish different strategies of choosing the step's length for the given instance of the specified feature. For the detailed
            description of the available values, see `StepLengthMethod` enum class documentation.

        step_length_uniform_low_to_mean_ratio : float
            The parameter's value is used when ``step_length_method=StepLengthMethod.UNIFORM``. The parameter's value is used to determine the boundaries values
            while drawing value of the step's length from uniform distribution. Parameter describes the ratio of the lower boundary value to the mean value
            of the uniform distribution. The parameter's value is from the range ``[0; 1]``.
            When value equals ``1``, then the step's length is drawn from the widest possible distribution from ``0`` to ``2 * step_length_mean``.
            With lower values of this parameter, uniform distribution is getting more narrow. When value equals ``0``, then uniform distribution is in range
            of the single ``step_length_mean`` value. Because of that, the step's length has a constant value of ``step_length_mean``.
            This ratio is applied across all features' types.

        step_length_normal_std_ratio : float
            The parameter's value is used when ``step_length_method=StepLengthMethod.NORMAL``. The parameter's value is used to determine the value of standard deviation
            while drawing value of the step's length from normal distribution. Parameter describes the ratio of the standard deviation value to the mean value
            of normal distribution. This ratio is applied across all features' types.

        step_angle_range_mean : float
            The exact value of the range of the step's angle to the direction of destination point of the given feature's type is determined by the gamma distribution
            with shape equals ``step_angle_range_mean`` and scale equals ``1.0``. This distribution is an extension from integer domain to real domain
            of the Poisson distribution with lambda equals ``step_angle_range_mean``.

        step_angle_range_limit : float
            Additional parameter which controls the range of the step's angle to the direction of destination point of all features' types. If drawn value of the range
            of the given feature's type is greater than ``step_angle_range_limit``, then the range is decreased to the ``step_angle_range_limit`` value.

        step_angle_method : StepAngleMethod
            Enum value is used to distinguish different strategies of choosing step's angle for the given instance of the specified feature. For the detailed
            description of the available values, see `StepAngleMethod` enum class documentation.

        step_angle_normal_std_ratio : float
            The parameter's value is used when ``step_angle_method=StepAngleMethod.NORMAL``. The parameter's value is used to determine the value of standard deviation
            while drawing value of the range of the step's angle from normal distribution. Parameter describes the ratio of the standard deviation value
            of normal distribution to the range of the step's angle. This ratio is applied across all features' types.

        waiting_time_frames : int
            Number of time frames which are counted down when the first feature instance of the given co-location instance reach the destination point.
            If still not all the features instances reach the current destination points after the given ``waiting_time_frames`` time,
            then new destination points will be defined.

        kwargs
            Other parameters passed to super constructor of derived class `BasicParameters`.
        """

        super().__init__(**kwargs)

        # check step_length_mean value
        if step_length_mean < 0:
            step_length_mean = 0

        # check step_length_uniform_low_to_mean_ratio value
        if step_length_uniform_low_to_mean_ratio < 0:
            step_length_uniform_low_to_mean_ratio = 0
        if step_length_uniform_low_to_mean_ratio > 1:
            step_length_uniform_low_to_mean_ratio = 1

        # check step_length_normal_std_ratio value
        if step_length_normal_std_ratio < 0:
            step_length_normal_std_ratio = 0

        # check step_angle_range_mean value
        if step_angle_range_mean < 0:
            step_angle_range_mean = 0

        # check step_angle_range_limit value
        if step_angle_range_limit < 0:
            step_angle_range_limit = 0

        # check step_angle_normal_std_ratio value
        if step_angle_normal_std_ratio < 0:
            step_angle_normal_std_ratio = 0

        # check waiting_time_frames value
        if waiting_time_frames < 0:
            waiting_time_frames = 0

        self.step_length_mean = step_length_mean
        self.step_length_method = step_length_method
        self.step_length_uniform_low_to_mean_ratio = step_length_uniform_low_to_mean_ratio
        self.step_length_normal_std_ratio = step_length_normal_std_ratio
        self.step_angle_range_mean = step_angle_range_mean
        self.step_angle_range_limit = step_angle_range_limit
        self.step_angle_method = step_angle_method
        self.step_angle_normal_std_ratio = step_angle_normal_std_ratio
        self.waiting_time_frames = waiting_time_frames
