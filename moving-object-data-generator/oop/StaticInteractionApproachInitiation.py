import numpy as np

from oop.FeatureInteractionModes import IdenticalFeaturesInteractionMode, DifferentFeaturesInteractionMode
from oop.GravitationApproachParameters import GravitationApproachParameters
from oop.SpatialStandardPlacement import SpatialStandardPlacement
from oop.StandardInitiation import StandardInitiation
from oop.StaticInteractionApproachParameters import StaticInteractionApproachParameters


class StaticInteractionApproachInitiation(StandardInitiation):
    def __init__(self):
        super().__init__()

        self.static_interaction_approach_parameters: StaticInteractionApproachParameters = StaticInteractionApproachParameters()
        self.instances_coor: np.ndarray = np.empty(shape=(0, 2), dtype=np.float64)
        self.mass: np.ndarray = np.array([], dtype=np.float64)
        self.mass_sum: float = 0.0
        self.center: np.ndarray = np.zeros(shape=(1, 2), dtype=np.float64)
        self.velocity: np.ndarray = np.empty(shape=(0, 2), dtype=np.float64)
        self.time_interval: float = 1.0
        self.approx_step_time_interval: float = 1.0
        self.spatial_standard_placement: SpatialStandardPlacement = SpatialStandardPlacement()
        self.features_interaction: np.ndarray = np.empty(shape=(0, 0), dtype=np.float64)

    def __group_collocation_features(self):
        collocation_start_feature_id = 0
        for i_colloc in range(self.collocation_lengths.size):
            # get the features ids of current co-location
            collocation_features = np.arange(collocation_start_feature_id, collocation_start_feature_id + self.collocation_lengths[i_colloc])
            collocation_features[-1] += i_colloc % self.static_interaction_approach_parameters.m_overlap

            # todo
            self.features_interaction

            # change starting feature of next co-location according to the m_overlap parameter value
            if (i_colloc + 1) % self.static_interaction_approach_parameters.m_overlap == 0:
                collocation_start_feature_id += self.collocation_lengths[i_colloc] + self.standard_parameters.m_overlap - 1
        return

    def initiate(self, siap: StaticInteractionApproachParameters = StaticInteractionApproachParameters()):
        super().initiate(sp=siap)

        self.static_interaction_approach_parameters = siap

        # create class object, which holds all data of the objects starting placement
        self.spatial_standard_placement = SpatialStandardPlacement()

        # place all objects at starting position
        self.spatial_standard_placement.place(si=self)

        # keep instances coordinates in one array
        self.instances_coor = np.column_stack(tup=(self.spatial_standard_placement.x, self.spatial_standard_placement.y))

        if siap.mass_param < 0:
            # create array of instances mass all equals to -mass_param
            self.mass = np.full(shape=self.features_instances_sum, fill_value=-siap.mass_param, dtype=np.float64)
        else:
            # each type of feature has own mean mass value drawn from gamma distribution
            feature_mass_mu = np.random.gamma(shape=siap.mass_param, scale=1.0, size=self.collocation_features_sum + siap.ndf)
            print("feature_mass_mu=%s" % str(feature_mass_mu))
            # each instance of given type feature has own mass value drawn from normal distribution
            self.mass = np.random.normal(loc=feature_mass_mu[self.features_ids], scale=feature_mass_mu[self.features_ids] / 5, size=self.features_instances_sum)
            self.mass[self.mass < 0] *= -1

        self.mass_sum = self.mass.sum()
        self.center = np.sum(a=self.instances_coor * self.mass[:, None], axis=0) / self.mass_sum

        if siap.velocity_param == 0:
            # create array of instances velocity all equals to 0
            self.velocity = np.zeros_like(self.instances_coor)
        elif siap.velocity_param < 0:
            # create array of instances velocity all with constant value in random direction
            velocity_angle = np.random.uniform(high=2 * np.pi, size=self.features_instances_sum)
            self.velocity = np.column_stack(tup=(np.cos(velocity_angle), np.sin(velocity_angle)))
            self.velocity *= -siap.velocity_param
        else:
            # create array of instances velocity all with gamma distribution value in random direction
            velocity_angle = np.random.uniform(high=2 * np.pi, size=self.features_instances_sum)
            self.velocity = np.column_stack(tup=(np.cos(velocity_angle), np.sin(velocity_angle)))
            self.velocity *= np.random.gamma(shape=siap.velocity_param, scale=1.0, size=self.features_instances_sum)[:, None]

        # define time interval between time frames
        self.time_interval = 1 / siap.time_unit

        # divide time interval of single time frame into steps of equal duration
        self.approx_step_time_interval = self.time_interval / siap.approx_steps_number

        # define matrix
        if self.static_interaction_approach_parameters.identical_features_interaction_mode is IdenticalFeaturesInteractionMode.ATTRACT:
            if self.static_interaction_approach_parameters.different_features_interaction_mode is DifferentFeaturesInteractionMode.ATTRACT:
                self.features_interaction = 1
            elif self.static_interaction_approach_parameters.different_features_interaction_mode is DifferentFeaturesInteractionMode.REPEL:
                self.features_interaction = -np.ones(shape=(self.features_instances_sum, self.features_instances_sum), dtype=np.int32)
                self.features_interaction[self.features_ids[:, None] == self.features_ids[None, :]] = 1
            elif self.static_interaction_approach_parameters.different_features_interaction_mode is DifferentFeaturesInteractionMode.COLLOCATION_ATTRACT_OTHER_NEUTRAL:
                self.features_interaction = np.zeros(shape=(self.features_instances_sum, self.features_instances_sum), dtype=np.int32)
                # todo
                self.features_interaction[self.features_ids[:, None] == self.features_ids[None, :]] = 1
            elif self.static_interaction_approach_parameters.different_features_interaction_mode is DifferentFeaturesInteractionMode.COLLOCATION_ATTRACT_OTHER_REPEL:
                self.features_interaction = -np.ones(shape=(self.features_instances_sum, self.features_instances_sum), dtype=np.int32)
                # todo
                self.features_interaction[self.features_ids[:, None] == self.features_ids[None, :]] = 1
        elif self.static_interaction_approach_parameters.identical_features_interaction_mode is IdenticalFeaturesInteractionMode.REPEL:
            if self.static_interaction_approach_parameters.different_features_interaction_mode is DifferentFeaturesInteractionMode.ATTRACT:
                self.features_interaction = np.ones(shape=(self.features_instances_sum, self.features_instances_sum), dtype=np.int32)
                self.features_interaction[self.features_ids[:, None] == self.features_ids[None, :]] = -1
            elif self.static_interaction_approach_parameters.different_features_interaction_mode is DifferentFeaturesInteractionMode.REPEL:
                self.features_interaction = -1
            elif self.static_interaction_approach_parameters.different_features_interaction_mode is DifferentFeaturesInteractionMode.COLLOCATION_ATTRACT_OTHER_NEUTRAL:
                self.features_interaction = np.zeros(shape=(self.features_instances_sum, self.features_instances_sum), dtype=np.int32)
                # todo
                self.features_interaction[self.features_ids[:, None] == self.features_ids[None, :]] = -1
            elif self.static_interaction_approach_parameters.different_features_interaction_mode is DifferentFeaturesInteractionMode.COLLOCATION_ATTRACT_OTHER_REPEL:
                self.features_interaction = -np.ones(shape=(self.features_instances_sum, self.features_instances_sum), dtype=np.int32)
                # todo
                self.features_interaction[self.features_ids[:, None] == self.features_ids[None, :]] = -1
