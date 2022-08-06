import numpy as np

from algorithms.enums.OptimalDistanceApproachEnums import MassMode, VelocityMode
from algorithms.initiation.StandardTimeFrameInitiation import StandardTimeFrameInitiation
from algorithms.parameters.OptimalDistanceApproachParameters import OptimalDistanceApproachParameters


class OptimalDistanceApproachInitiation(StandardTimeFrameInitiation):
    def __init__(self):
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

        # perform the initiation of the super class
        super().initiate(stfp=odap)

        # store parameters of the initiation
        self.optimal_distance_approach_parameters = odap

        # copy coordinates of features instances
        self.instances_coor = np.copy(self.spatial_standard_placement.features_instances_coor)

        # create array of instances' mass
        if odap.mass_mode == MassMode.CONSTANT:
            # create array of instances' mass, all equal to the 'mass_mean' parameter value
            self.mass = np.full(shape=self.features_instances_sum, fill_value=odap.mass_mean, dtype=np.float64)

        elif odap.mass_mode == MassMode.FEATURE_CONSTANT:
            # each type of feature has own constant mass value drawn from gamma distribution
            feature_mass_const = np.random.gamma(shape=odap.mass_mean, scale=1.0, size=self.features_sum)

            # each instance of the given type feature has mass value which is equal to the feature's constant mass value
            self.mass = feature_mass_const[self.features_ids]

        elif odap.mass_mode == MassMode.NORMAL:
            # each type of feature has own mean mass value drawn from gamma distribution
            feature_mass_mu = np.random.gamma(shape=odap.mass_mean, scale=1.0, size=self.features_sum)

            # each instance of the given type feature has own mass value drawn from normal distribution
            self.mass = np.random.normal(loc=feature_mass_mu[self.features_ids], scale=feature_mass_mu[self.features_ids] * odap.mass_normal_std_ratio, size=self.features_instances_sum)
            self.mass[self.mass < 0] *= -1

        self.mass_sum = self.mass.sum()
        self.center = np.sum(a=self.instances_coor * self.mass[:, None], axis=0) / self.mass_sum

        # calculate constant factor of force between each pair of instances, which depends on 'k_force' parameter and mass of the instances
        self.force_multiplier_constant = odap.k_force * self.mass[:, None] * self.mass[None, :]

        # calculate constant factor of force between mass center and each instance, which depends on 'k_force' parameter and mass of the instance and center's mass
        self.force_center_multiplier_constant = odap.k_force * self.mass_sum * self.mass

        # create array of instances' velocity
        if odap.velocity_mode == VelocityMode.CONSTANT:
            if odap.velocity_mean == 0.0:
                # create array of instances velocity all equals to 0
                self.velocity = np.zeros_like(self.instances_coor, dtype=np.float64)

            else:
                # create array of instances' velocities, all with constant value in random direction
                velocity_angle = np.random.uniform(high=2 * np.pi, size=self.features_instances_sum)
                self.velocity = np.column_stack(tup=(np.cos(velocity_angle), np.sin(velocity_angle)))
                self.velocity *= odap.velocity_mean

        elif odap.velocity_mode == VelocityMode.GAMMA:
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

        # create boolean array, which tell if the given pair of features instances is located in the common co-location instance
        self.common_collocation_instance_flag = self.collocations_clumpy_instances_global_ids[None, :] == self.collocations_clumpy_instances_global_ids[:, None]
