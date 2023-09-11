import numpy as np

from algorithms.enums.OptimalDistanceApproachEnums import MassMethod, VelocityMethod
from algorithms.parameters.StandardTimeFrameParameters import StandardTimeFrameParameters


class OptimalDistanceApproachParameters(StandardTimeFrameParameters):
    """
    The class represents set of parameters used by the `SpatioTemporalOptimalDistanceApproachGenerator` class of a spatio-temporal data generator.
    """

    def __init__(
            self,
            time_unit: float = 1.0,
            approx_steps_number: int = 10,
            k_optimal_distance: float = 1.0,
            k_force: float = 1.0,
            force_limit: float = 5.0,
            velocity_limit: float = 5.0,
            faraway_limit_ratio: float = np.sqrt(2) / 2,
            mass_method: MassMethod = MassMethod.CONSTANT,
            mass_mean: float = 1.0,
            mass_normal_std_ratio: float = 1 / 5,
            velocity_method: VelocityMethod = VelocityMethod.CONSTANT,
            velocity_mean: float = 0.0,
            **kwargs):
        """
        Construct an object which holds all the required parameters of the `SpatioTemporalOptimalDistanceApproachGenerator` class of a spatio-temporal data generator.

        Parameters
        ----------
        time_unit : float
            The time unit expressed by the number of time frames. The parameter is also known as the "t_unit".

        approx_steps_number : int
            The number of equal steps in the time domain between two consecutive time frames. A larger number of intermediate steps allows for a more accurate simulation
            of the interactions occurring between objects that affect the obtained locations in subsequent time frames. The parameter is also known as the "approx_steps".

        k_optimal_distance : float
            The main parameter of the generator, which defines the optimal distance between features instances intended to co-occur within a given co-location instance.
            At this distance, the forces of attraction and repulsion cancel each other out. The parameter is also known as the "k_optimal".

        k_force : float
            The constant scaling factor, which is used in the force interaction formula. The parameter is also known as the "k_f".

        force_limit : float
            The limit of maximum absolute value of resultant force applied to the given feature instance. The parameter is also known as the "f_max".

        velocity_limit : float
            The limit of maximum absolute value of resultant velocity at which the feature instance can move. The parameter is also known as the "v_max".

        faraway_limit_ratio : float
            The ratio of the radius defining the beginning of the attraction force to the initially declared size of the spatial framework. This ratio is used to determine
            the exact radius length that represents the distance from the center of the initial spatial framework. Once an object exceeds this radius distance,
            an additional attraction force starts acting on it, pulling it back towards the center. The center is a fixed point determined at the first time frame
            as the center of mass of all objects. The attractive force acting on a given object is proportional to the mass of that object, the total mass of all objects,
            the constant scaling factor ``k_force`` and is also proportional to the square of the relative distance of exceeding the allowed distance radius.
            The parameter is also known as the "l_faraway_limit".

        mass_method : MassMethod
            The enum value is used to distinguish different strategies of choosing the mass for the given instance of the specified feature type.
            For the detailed description of the available values, see `MassMethod` enum class documentation. The parameter is also known as the "m_method".

        mass_mean : float
            The mean value of the mass of a feature instance. The parameter is also known as the "μ_m".

        mass_normal_std_ratio : float
            The parameter's value is used when ``mass_method=MassMethod.NORMAL``. The parameter's value is used to determine the value of standard deviation
            while drawing value of the mass from a normal distribution. Parameter describes the ratio of the standard deviation value to the mean value
            of a normal distribution. This ratio is applied across all features types. The parameter is also known as the "l_σ_m_normal".

        velocity_method : VelocityMethod
            The enum value is used to distinguish different strategies of choosing the initial velocity for a feature instance. For the detailed description
            of the available values, see `VelocityMethod` enum class documentation. The parameter is also known as the "v_method".

        velocity_mean : float
            The mean value of the initial velocity of a feature instance. The parameter is also known as the "μ_v".

        kwargs
            Other parameters passed to the super constructor of the derived class `StandardTimeFrameParameters`.
        """

        super().__init__(**kwargs)

        # check 'time_unit' value
        if time_unit <= 0.0:
            time_unit = 1.0

        # check 'approx_steps_number' value
        if approx_steps_number <= 0:
            approx_steps_number = 1

        # check 'k_optimal_distance' value
        if k_optimal_distance <= 0.0:
            k_optimal_distance = 1.0

        # check 'k_force' value
        if k_force <= 0.0:
            k_force = 1.0

        # check 'force_limit' value
        if force_limit <= 0.0:
            force_limit = 5.0

        # check 'velocity_limit' value
        if velocity_limit <= 0.0:
            velocity_limit = 5.0

        # check 'faraway_limit_ratio' value
        if faraway_limit_ratio <= 0.0:
            faraway_limit_ratio = np.sqrt(2) / 2

        # check 'mass_mean' value
        if mass_mean <= 0.0:
            mass_mean = 1.0

        # check 'mass_normal_std_ratio' value
        if mass_normal_std_ratio < 0.0:
            mass_normal_std_ratio = 0.0

        # check 'velocity_mean' value
        if velocity_mean < 0.0:
            velocity_mean = 0.0

        self.time_unit = time_unit
        self.approx_steps_number = approx_steps_number
        self.k_optimal_distance = k_optimal_distance
        self.k_force = k_force
        self.force_limit = force_limit
        self.velocity_limit = velocity_limit
        self.faraway_limit_ratio = faraway_limit_ratio
        self.mass_method = mass_method
        self.mass_mean = mass_mean
        self.mass_normal_std_ratio = mass_normal_std_ratio
        self.velocity_method = velocity_method
        self.velocity_mean = velocity_mean
