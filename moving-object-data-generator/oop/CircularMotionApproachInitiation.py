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

        # perform the initiation of the super class
        super().initiate(bp=cmap)

        # store parameters of the initiation
        self.circular_motion_approach_parameters = cmap

        # create class object, which holds all data of the objects starting placement
        spatial_basic_placement = SpatialBasicPlacement()

        # place all objects at starting position
        spatial_basic_placement.place(bi=self)

        # copy coordinates of features instances
        features_instances_start_coor = np.copy(spatial_basic_placement.features_instances_coor)

        # calculate mean position of each co-location's instance
        slice_ind = np.concatenate(([0], self.collocations_instances_global_ids_repeats.cumsum()[:-1]))
        features_instances_start_coor_sums = np.add.reduceat(features_instances_start_coor, indices=slice_ind, axis=0)
        features_instances_start_coor_mean = features_instances_start_coor_sums / self.collocations_instances_global_ids_repeats[:, None]

        # draw angle value at the starting time point of the every circular orbit of the every co-location's instance
        collocations_instances_start_angle = np.random.uniform(high=2 * np.pi, size=(cmap.circle_chain_size, self.collocations_instances_global_sum))

        # determine angle value at the starting time point of the every circular orbit of the feature instance which belong to the given co-location's instance
        self.start_angle = np.repeat(a=collocations_instances_start_angle, repeats=self.collocations_instances_global_ids_repeats, axis=1)

        # draw length of the orbit radius of the co-locations instances from uniform distribution according to boundaries values passed in parameters
        collocations_instances_radius_length = np.random.uniform(low=cmap.circle_r_min, high=cmap.circle_r_max, size=(cmap.circle_chain_size, self.collocations_instances_global_sum))

        # determine radius length of the every circular orbit of the feature instance which belong to the given co-location's instance
        self.radius_length = np.repeat(a=collocations_instances_radius_length, repeats=self.collocations_instances_global_ids_repeats, axis=1)

        # ---begin--- find the starting point of calculation orbital position - the center of the first circular orbit
        # calculate position determined by each of circular orbit - position calculated in reference system of the given circular orbit center (per co-location instance)
        circle_delta_x = collocations_instances_radius_length * np.cos(collocations_instances_start_angle)
        circle_delta_y = collocations_instances_radius_length * np.sin(collocations_instances_start_angle)

        # find starting point by subtracting summed deltas from mean position of co-locations' instances
        collocations_instances_start_orbit_center_coor = np.copy(features_instances_start_coor_mean)
        collocations_instances_start_orbit_center_coor[:, 0] -= np.sum(a=circle_delta_x, axis=0)
        collocations_instances_start_orbit_center_coor[:, 1] -= np.sum(a=circle_delta_y, axis=0)

        # determine starting point of every feature's instance
        self.start_orbit_center_coor = np.repeat(a=collocations_instances_start_orbit_center_coor, repeats=self.collocations_instances_global_ids_repeats, axis=0)
        # ----end---- find the starting point of calculation orbital position - the center of the first circular orbit

        # draw angular velocity values of the co-locations instances from uniform distribution according to boundaries values passed in parameters
        collocations_instances_angular_velocity = np.random.uniform(low=cmap.omega_min, high=cmap.omega_max, size=(cmap.circle_chain_size, self.collocations_instances_global_sum))

        # determine angular velocity values of the features which belong to the given co-location's instance
        self.angular_velocity = np.repeat(a=collocations_instances_angular_velocity, repeats=self.collocations_instances_global_ids_repeats, axis=1)

        # determine direction of the rotation - features of the given co-location have the same direction
        angular_velocity_clockwise = np.repeat(
            a=np.random.randint(2, size=(cmap.circle_chain_size, self.collocations_instances_global_sum), dtype=bool),
            repeats=self.collocations_instances_global_ids_repeats,
            axis=1
        )
        self.angular_velocity[angular_velocity_clockwise] *= -1
