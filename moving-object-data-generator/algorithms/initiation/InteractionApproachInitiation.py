import numpy as np

from algorithms.enums.InteractionApproachEnums import IdenticalFeaturesInteractionMode, DifferentFeaturesInteractionMode, MassMode, VelocityMode
from algorithms.initiation.StandardTimeFrameInitiation import StandardTimeFrameInitiation
from algorithms.parameters.InteractionApproachParameters import InteractionApproachParameters


class InteractionApproachInitiation(StandardTimeFrameInitiation):
    def __init__(self):
        super().__init__()

        self.interaction_approach_parameters: InteractionApproachParameters = InteractionApproachParameters()
        self.instances_coor: np.ndarray = np.empty(shape=(0, 2), dtype=np.float64)
        self.mass: np.ndarray = np.array([], dtype=np.float64)
        self.mass_sum: float = 0.0
        self.center: np.ndarray = np.zeros(shape=(1, 2), dtype=np.float64)
        self.force_multiplier_constant: np.ndarray = np.empty(shape=(0, 0), dtype=np.float64)
        self.force_center_multiplier_constant: np.ndarray = np.empty(shape=(0, 2), dtype=np.float64)
        self.velocity: np.ndarray = np.empty(shape=(0, 2), dtype=np.float64)
        self.time_interval: float = 1.0
        self.approx_step_time_interval: float = 1.0
        self.faraway_limit: float = 1000.0 * np.sqrt(2) / 2
        self.features_instances_interaction: np.ndarray = np.empty(shape=(0, 0), dtype=np.float64)

    def __define_features_instances_interaction(self, different_collocations_interaction_value=0, identical_features_interaction_value=1, noise_features_interaction_value=1):
        """
        Private method define `self.features_instances_interaction` matrix values

        :param different_collocations_interaction_value: default value of interaction between features which participate in different co-location
        :param identical_features_interaction_value: interaction value between instances of the identical feature
        :param noise_features_interaction_value: default value of interaction noise feature with any other feature
        """

        # initiate interaction matrix of co-location features with default value of interaction between different co-location
        collocation_features_interaction = np.full(fill_value=different_collocations_interaction_value, shape=(self.collocation_features_sum, self.collocation_features_sum), dtype=np.int32)

        collocation_start_feature_id = 0
        for i_colloc in range(self.collocations_sum):
            # get the features ids of current co-location
            collocation_features = np.arange(collocation_start_feature_id, collocation_start_feature_id + self.collocation_lengths[i_colloc])
            collocation_features[-1] += i_colloc % self.interaction_approach_parameters.m_overlap

            # features of given co-location attract each other
            collocation_features_interaction[collocation_features[:, None], collocation_features[None, :]] = 1

            # change starting feature of next co-location according to the m_overlap parameter value
            if (i_colloc + 1) % self.interaction_approach_parameters.m_overlap == 0:
                collocation_start_feature_id += self.collocation_lengths[i_colloc] + self.interaction_approach_parameters.m_overlap - 1

        # initiate interaction matrix of features instances with default value of interaction noise feature with any other feature
        self.features_instances_interaction = np.full(fill_value=noise_features_interaction_value, shape=(self.features_instances_sum, self.features_instances_sum), dtype=np.int32)

        # define interaction between co-location features instances according to 'collocation_features_interaction' matrix
        self.features_instances_interaction[:self.collocation_features_instances_sum, :self.collocation_features_instances_sum] = collocation_features_interaction[
            self.collocation_features_ids[:, None], self.collocation_features_ids[None, :]
        ]

        # set interaction value between instances of the identical feature
        self.features_instances_interaction[self.features_ids[:, None] == self.features_ids[None, :]] = identical_features_interaction_value

    def initiate(self, iap: InteractionApproachParameters = InteractionApproachParameters()):
        super().initiate(stfp=iap)

        self.interaction_approach_parameters = iap

        # copy coordinates of features instances
        self.instances_coor = np.copy(self.spatial_standard_placement.features_instances_coor)

        # create array of instances' mass
        if iap.mass_mode == MassMode.CONSTANT:
            # create array of instances' mass, all equal to the 'mass_mean' parameter value
            self.mass = np.full(shape=self.features_instances_sum, fill_value=iap.mass_mean, dtype=np.float64)

        elif iap.mass_mode == MassMode.FEATURE_CONSTANT:
            # each type of feature has own constant mass value drawn from gamma distribution
            feature_mass_const = np.random.gamma(shape=iap.mass_mean, scale=1.0, size=self.features_sum)

            # each instance of the given type feature has mass value which is equal to the feature's constant mass value
            self.mass = feature_mass_const[self.features_ids]

        elif iap.mass_mode == MassMode.NORMAL:
            # each type of feature has own mean mass value drawn from gamma distribution
            feature_mass_mu = np.random.gamma(shape=iap.mass_mean, scale=1.0, size=self.features_sum)

            # each instance of the given type feature has own mass value drawn from normal distribution
            self.mass = np.random.normal(loc=feature_mass_mu[self.features_ids], scale=feature_mass_mu[self.features_ids] * iap.mass_normal_std_ratio, size=self.features_instances_sum)
            self.mass[self.mass < 0] *= -1

        self.mass_sum = self.mass.sum()
        self.center = np.sum(a=self.instances_coor * self.mass[:, None], axis=0) / self.mass_sum

        # calculate constant factor of force between each pair of instances, which depends on 'k_force' parameter and mass of the instances
        self.force_multiplier_constant = iap.k_force * self.mass[:, None] * self.mass[None, :]

        # calculate constant factor of force between mass center and each instance, which depends on 'k_force' parameter and mass of the instance and center's mass
        self.force_center_multiplier_constant = iap.k_force * self.mass_sum * self.mass

        # create array of instances' velocity
        if iap.velocity_mode == VelocityMode.CONSTANT:
            if iap.velocity_mean == 0.0:
                # create array of instances velocity all equals to 0
                self.velocity = np.zeros_like(self.instances_coor, dtype=np.float64)

            else:
                # create array of instances' velocities, all with constant value in random direction
                velocity_angle = np.random.uniform(high=2 * np.pi, size=self.features_instances_sum)
                self.velocity = np.column_stack(tup=(np.cos(velocity_angle), np.sin(velocity_angle)))
                self.velocity *= iap.velocity_mean

        elif iap.velocity_mode == VelocityMode.GAMMA:
            # create array of instances' velocities, all with gamma distribution value in random direction
            velocity_angle = np.random.uniform(high=2 * np.pi, size=self.features_instances_sum)
            self.velocity = np.column_stack(tup=(np.cos(velocity_angle), np.sin(velocity_angle)))
            self.velocity *= np.random.gamma(shape=iap.velocity_mean, scale=1.0, size=self.features_instances_sum)[:, None]

        # limit initiated velocity across each dimension with 'velocity_limit' parameter value
        self.velocity[self.velocity > iap.velocity_limit] = iap.velocity_limit
        self.velocity[self.velocity < -iap.velocity_limit] = -iap.velocity_limit

        # define time interval between time frames
        self.time_interval = 1 / iap.time_unit

        # divide time interval of single time frame into steps of equal duration
        self.approx_step_time_interval = self.time_interval / iap.approx_steps_number

        # define faraway limit
        self.faraway_limit = iap.faraway_limit_ratio * iap.area

        # define `self.features_instances_interaction` matrix values
        if iap.identical_features_interaction_mode is IdenticalFeaturesInteractionMode.ATTRACT:
            if iap.different_features_interaction_mode is DifferentFeaturesInteractionMode.ATTRACT:
                self.features_instances_interaction = 1

            elif iap.different_features_interaction_mode is DifferentFeaturesInteractionMode.REPEL:
                self.features_instances_interaction = -np.ones(shape=(self.features_instances_sum, self.features_instances_sum), dtype=np.int32)
                self.features_instances_interaction[self.features_ids[:, None] == self.features_ids[None, :]] = 1

            elif iap.different_features_interaction_mode is DifferentFeaturesInteractionMode.COLLOCATION_ATTRACT_OTHER_NEUTRAL:
                self.__define_features_instances_interaction(different_collocations_interaction_value=0, identical_features_interaction_value=1)

            elif iap.different_features_interaction_mode is DifferentFeaturesInteractionMode.COLLOCATION_ATTRACT_OTHER_REPEL:
                self.__define_features_instances_interaction(different_collocations_interaction_value=-1, identical_features_interaction_value=1)

        elif iap.identical_features_interaction_mode is IdenticalFeaturesInteractionMode.REPEL:
            if iap.different_features_interaction_mode is DifferentFeaturesInteractionMode.ATTRACT:
                self.features_instances_interaction = np.ones(shape=(self.features_instances_sum, self.features_instances_sum), dtype=np.int32)
                self.features_instances_interaction[self.features_ids[:, None] == self.features_ids[None, :]] = -1

            elif iap.different_features_interaction_mode is DifferentFeaturesInteractionMode.REPEL:
                self.features_instances_interaction = -1

            elif iap.different_features_interaction_mode is DifferentFeaturesInteractionMode.COLLOCATION_ATTRACT_OTHER_NEUTRAL:
                self.__define_features_instances_interaction(different_collocations_interaction_value=0, identical_features_interaction_value=-1)

            elif iap.different_features_interaction_mode is DifferentFeaturesInteractionMode.COLLOCATION_ATTRACT_OTHER_REPEL:
                self.__define_features_instances_interaction(different_collocations_interaction_value=-1, identical_features_interaction_value=-1)
