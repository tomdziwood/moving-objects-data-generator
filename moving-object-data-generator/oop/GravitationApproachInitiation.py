import numpy as np

from oop.GravitationApproachParameters import GravitationApproachParameters
from oop.SpatialStandardPlacement import SpatialStandardPlacement
from oop.StandardInitiation import StandardInitiation


class GravitationApproachInitiation(StandardInitiation):
    def __init__(self):
        super().__init__()

        self.gravitation_approach_parameters: GravitationApproachParameters = GravitationApproachParameters()
        self.instances_coor: np.ndarray = np.empty(shape=(0, 2), dtype=np.float64)
        self.mass: np.ndarray = np.array([], dtype=np.float64)
        self.mass_sum: float = 0.0
        self.center: np.ndarray = np.zeros(shape=(1, 2), dtype=np.float64)
        self.velocity: np.ndarray = np.empty(shape=(0, 2), dtype=np.float64)
        self.time_interval: float = 1.0
        self.approx_step_time_interval: float = 1.0
        self.spatial_standard_placement: SpatialStandardPlacement = SpatialStandardPlacement()

    def initiate(self, gap: GravitationApproachParameters = GravitationApproachParameters()):
        super().initiate(sp=gap)

        self.gravitation_approach_parameters = gap

        # create class object, which holds all data of the objects starting placement
        self.spatial_standard_placement = SpatialStandardPlacement()

        # place all objects at starting position
        self.spatial_standard_placement.place(si=self)

        # keep instances coordinates in one array
        self.instances_coor = np.column_stack(tup=(self.spatial_standard_placement.x, self.spatial_standard_placement.y))

        if gap.mass_param < 0:
            # create array of instances mass all equals to -mass_param
            self.mass = np.full(shape=self.features_sum, fill_value=-gap.mass_param, dtype=np.float64)
        else:
            # each type of feature has own mean mass value drawn from gamma distribution
            feature_mass_mu = np.random.gamma(shape=gap.mass_param, scale=1.0, size=self.collocation_features_sum + gap.ndf)
            print("feature_mass_mu=%s" % str(feature_mass_mu))
            # each instance of given type feature has own mass value drawn from normal distribution
            self.mass = np.random.normal(loc=feature_mass_mu[self.features_ids], scale=feature_mass_mu[self.features_ids] / 5, size=self.features_sum)
            self.mass[self.mass < 0] *= -1

        self.mass_sum = self.mass.sum()
        self.center = np.sum(a=self.instances_coor * self.mass[:, None], axis=0) / self.mass_sum

        if gap.velocity_param == 0:
            # create array of instances velocity all equals to 0
            self.velocity = np.zeros_like(self.instances_coor)
        elif gap.velocity_param < 0:
            # create array of instances velocity all with constant value in random direction
            velocity_angle = np.random.uniform(high=2 * np.pi, size=self.features_sum)
            self.velocity = np.column_stack(tup=(np.cos(velocity_angle), np.sin(velocity_angle)))
            self.velocity *= -gap.velocity_param
        else:
            # create array of instances velocity all with gamma distribution value in random direction
            velocity_angle = np.random.uniform(high=2 * np.pi, size=self.features_sum)
            self.velocity = np.column_stack(tup=(np.cos(velocity_angle), np.sin(velocity_angle)))
            self.velocity *= np.random.gamma(shape=gap.velocity_param, scale=1.0, size=self.features_sum)[:, None]

        # define time interval between time frames
        self.time_interval = 1 / gap.time_unit

        # divide time interval of single time frame into steps of equal duration
        self.approx_step_time_interval = self.time_interval / gap.approx_steps_number
