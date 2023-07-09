import numpy as np

from algorithms.enums.OptimalDistanceApproachEnums import MassMethod, VelocityMethod
from algorithms.initiation.StandardTimeFrameInitiation import StandardTimeFrameInitiation
from algorithms.parameters.OptimalDistanceApproachParameters import OptimalDistanceApproachParameters


class OptimalDistanceApproachInitiation(StandardTimeFrameInitiation):
    """
    The class of a `SpatioTemporalOptimalDistanceApproachGenerator` initiation. Object of this class stores all initial data, which is required to generate
    spatio-temporal data in each time frame.

    Attributes
    ----------
    optimal_distance_approach_parameters : OptimalDistanceApproachParameters
        The object of class `OptimalDistanceApproachParameters`, which holds all the required parameters of the `SpatioTemporalOptimalDistanceApproachGenerator` generator.

    instances_coor : np.ndarray
        The array's size is equal to the number of features instances. The i-th value represents the coordinates of the location of the i-th feature instance
        initiated at the first time frame. This array is used as a shortcut for initiated coordinates available at ``self.spatial_standard_placement.features_instances_coor``.

    mass : np.ndarray
        The array's size is equal to the number of features instances. The i-th value represents the mass of the i-th feature instance.

    mass_sum : float
        The sum of the mass of all features instances.

    center : np.ndarray
        The array contains the coordinates of the center of the spatial framework. The center is the mass center of all features instances
        determined by their initial locations at first time frame and their respective masses. The calculated point of mass center remains constant at every time frames
        and is used in computation of the attraction force that pulls objects back towards the designated center, which is the center of mass, when they have deviated too far.

    force_multiplier_constant : np.ndarray
        The size of the matrix is equal to the number of features instances X the number of features instances. The matrix contains the precalculated constant factor
        of attraction and repulsion forces between each pair of features instances, which depends on the 'k_force' parameter and the mass of both instances.
        The exact value of the forces between each pair of features instances is calculated, taking into account their current location at specific moments.

    force_center_multiplier_constant : np.ndarray
        The array's size is equal to the number of features instances. The i-th value represents the precalculated constant factor of attraction force between mass center
        and the i-th feature instance. The exact value of the attraction force acting on the given object is calculated, taking into account the object's current location
        at specific moments.

    velocity : np.ndarray
        The array's size is equal to the number of features instances. The i-th value represents the velocity of the i-th feature instance initiated at the first time frame.

    time_interval : float
        The value of time interval between two consecutive time frames. The time interval is equal to the inverse of the declared ``time_unit`` parameter.

    approx_step_time_interval : float
        The length of every equal steps in the time domain between two consecutive time frames. The length is calculated by dividing the time interval``time_interval``
        by the number of equal steps ``approx_steps_number``.

    faraway_limit : float
        The distance measured from the center of the spatial framework, beyond which the attraction force starts to act on the moving object. The distance is determined
        with the ``faraway_limit_ratio`` parameter value.

    common_collocation_instance_flag : np.ndarray
        The size of the matrix is equal to the number of features instances X the number of features instances. The matrix contains the boolean values indicating
        whether the given pair of features instances participates in the common co-location instance.
    """

    def __init__(self):
        """
        Construct empty object of the `OptimalDistanceApproachInitiation` class.
        """

        super().__init__()

        self.optimal_distance_approach_parameters: OptimalDistanceApproachParameters = OptimalDistanceApproachParameters()
        self.instances_coor: np.ndarray = np.empty(shape=(0, 2), dtype=np.float64)
        self.mass: np.ndarray = np.array([], dtype=np.float64)
        self.mass_sum: float = 0.0
        self.center: np.ndarray = np.zeros(shape=(1, 2), dtype=np.float64)
        self.force_multiplier_constant: np.ndarray = np.empty(shape=(0, 0), dtype=np.float64)
        self.force_center_multiplier_constant: np.ndarray = np.empty(shape=(0, 0), dtype=np.float64)
        self.velocity: np.ndarray = np.empty(shape=(0, 2), dtype=np.float64)
        self.time_interval: float = 1.0
        self.approx_step_time_interval: float = 1.0
        self.faraway_limit: float = 1000.0 * np.sqrt(2) / 2
        self.common_collocation_instance_flag: np.ndarray = np.empty(shape=(0, 0), dtype=bool)

    def initiate(self, odap: OptimalDistanceApproachParameters = OptimalDistanceApproachParameters()):
        """
        Initiate required data to generate spatio-temporal data in each time frame.

        Parameters
        ----------
        odap : OptimalDistanceApproachParameters
            The object of class `OptimalDistanceApproachParameters`, which holds all the required parameters of the `SpatioTemporalOptimalDistanceApproachGenerator` generator.
            Its attributes will be used to initialize required data.
        """

        # perform the initiation of the super class
        super().initiate(stfp=odap)

        # store parameters of the initiation
        self.optimal_distance_approach_parameters = odap

        # copy coordinates of features instances
        self.instances_coor = np.copy(self.spatial_standard_placement.features_instances_coor)

        # create array of instances' mass
        if odap.mass_method == MassMethod.CONSTANT:
            # create array of instances' mass, all equal to the 'mass_mean' parameter value
            self.mass = np.full(shape=self.features_instances_sum, fill_value=odap.mass_mean, dtype=np.float64)

        elif odap.mass_method == MassMethod.FEATURE_CONSTANT:
            # each type of feature has own constant mass value drawn from a gamma distribution
            feature_mass_const = np.random.gamma(shape=odap.mass_mean, scale=1.0, size=self.features_sum)

            # each instance of the given type feature has mass value which is equal to the feature's constant mass value
            self.mass = feature_mass_const[self.features_ids]

        elif odap.mass_method == MassMethod.NORMAL:
            # each type of feature has own mean mass value drawn from a gamma distribution
            feature_mass_mu = np.random.gamma(shape=odap.mass_mean, scale=1.0, size=self.features_sum)

            # each instance of the given type feature has own mass value drawn from a normal distribution
            self.mass = np.random.normal(loc=feature_mass_mu[self.features_ids], scale=feature_mass_mu[self.features_ids] * odap.mass_normal_std_ratio, size=self.features_instances_sum)
            self.mass[self.mass < 0] *= -1

        self.mass_sum = self.mass.sum()
        self.center = np.sum(a=self.instances_coor * self.mass[:, None], axis=0) / self.mass_sum

        # calculate constant factor of force between each pair of instances, which depends on 'k_force' parameter and mass of the instances
        self.force_multiplier_constant = odap.k_force * self.mass[:, None] * self.mass[None, :]

        # calculate constant factor of force between mass center and each instance, which depends on 'k_force' parameter and mass of the instance and center's mass
        self.force_center_multiplier_constant = odap.k_force * self.mass_sum * self.mass

        # create array of instances' velocity
        if odap.velocity_method == VelocityMethod.CONSTANT:
            if odap.velocity_mean == 0.0:
                # create array of instances velocity all equals to 0
                self.velocity = np.zeros_like(self.instances_coor, dtype=np.float64)

            else:
                # create array of instances' velocities, all with constant value in random direction
                velocity_angle = np.random.uniform(high=2 * np.pi, size=self.features_instances_sum)
                self.velocity = np.column_stack(tup=(np.cos(velocity_angle), np.sin(velocity_angle)))
                self.velocity *= odap.velocity_mean

        elif odap.velocity_method == VelocityMethod.GAMMA:
            # create array of instances' velocities, all with gamma distribution value in random direction
            velocity_angle = np.random.uniform(high=2 * np.pi, size=self.features_instances_sum)
            self.velocity = np.column_stack(tup=(np.cos(velocity_angle), np.sin(velocity_angle)))
            self.velocity *= np.random.gamma(shape=odap.velocity_mean, scale=1.0, size=self.features_instances_sum)[:, None]

        # limit initiated velocity across each dimension with 'velocity_limit' parameter value
        self.velocity[self.velocity > odap.velocity_limit] = odap.velocity_limit
        self.velocity[self.velocity < -odap.velocity_limit] = -odap.velocity_limit

        # define time interval between time frames
        self.time_interval = 1 / odap.time_unit

        # divide time interval of single time frame into steps of equal duration
        self.approx_step_time_interval = self.time_interval / odap.approx_steps_number

        # define faraway limit
        self.faraway_limit = odap.faraway_limit_ratio * odap.area

        # create boolean array, which tell if the given pair of features instances participates in the common co-location instance
        self.common_collocation_instance_flag = self.collocations_clumpy_instances_global_ids[None, :] == self.collocations_clumpy_instances_global_ids[:, None]
