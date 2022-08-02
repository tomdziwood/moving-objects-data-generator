import numpy as np

from oop.CircularMotionApproachParameters import CircularMotionApproachParameters
from oop.SpatialBasicPlacement import SpatialBasicPlacement
from oop.BasicInitiation import BasicInitiation


class CircularMotionApproachInitiation(BasicInitiation):
    def __init__(self):

        super().__init__()

        self.circular_motion_approach_parameters: CircularMotionApproachParameters = CircularMotionApproachParameters()
        self.radius_length: np.ndarray = np.array([], dtype=np.float64)
        self.angular_velocity: np.ndarray = np.array([], dtype=np.float64)
        self.start_angle: np.ndarray = np.array([], dtype=np.float64)
        self.start_orbit_center_coor: np.ndarray = np.empty(shape=(0, 2), dtype=np.float64)

    def initiate(self, cmap: CircularMotionApproachParameters = CircularMotionApproachParameters()):

        super().initiate(bp=cmap)

        self.circular_motion_approach_parameters = cmap

        # create class object, which holds all data of the objects starting placement
        spatial_basic_placement = SpatialBasicPlacement()

        # place all objects at starting position
        spatial_basic_placement.place(bi=self)

        # copy coordinates of features instances
        features_instances_start_coor = np.copy(spatial_basic_placement.features_instances_coor)
        print(features_instances_start_coor)

        # determine orbit centers of the co-locations' instances
        collocations_instances_orbit_center = np.random.uniform(high=self.area_in_cell_dim * cmap.cell_size, size=(self.collocations_instances_global_sum, 2))

        # determine orbit centers of the features which belong to the given co-location's instance
        features_instances_orbit_center = np.repeat(a=collocations_instances_orbit_center, repeats=self.collocations_instances_global_ids_repeats, axis=0)

        # remember starting position of calculating orbital position
        self.start_orbit_center_coor = features_instances_orbit_center

        # calculate coordinates difference between feature instance and its orbit center
        coor_diff = features_instances_start_coor - features_instances_orbit_center

        # calculate length of the orbit radius
        self.radius_length = np.sqrt(np.sum(a=coor_diff ** 2, axis=-1))

        # calculate angular velocity based on the given linear velocity parameter value
        self.angular_velocity = cmap.linear_velocity_mean / self.radius_length

        # determine direction of the rotation - features of the given co-location have the same direction
        angular_velocity_clockwise = np.repeat(
            a=np.random.randint(2, size=self.collocations_instances_global_sum, dtype=bool),
            repeats=self.collocations_instances_global_ids_repeats
        )
        self.angular_velocity[angular_velocity_clockwise] *= -1

        # calculate angle value at the beginning
        self.start_angle = np.arctan2(coor_diff[:, 1], coor_diff[:, 0])
