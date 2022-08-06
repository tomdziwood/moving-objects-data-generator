import numpy as np

from algorithms.initiation.StandardTimeFrameInitiation import StandardTimeFrameInitiation
from algorithms.parameters.CircularMotionApproachParameters import CircularMotionApproachParameters


class CircularMotionApproachInitiation(StandardTimeFrameInitiation):
    def __init__(self):

        super().__init__()

        self.circular_motion_approach_parameters: CircularMotionApproachParameters = CircularMotionApproachParameters()
        self.radius_length: np.ndarray = np.array([], dtype=np.float64)
        self.angular_velocity: np.ndarray = np.array([], dtype=np.float64)
        self.start_angle: np.ndarray = np.array([], dtype=np.float64)
        self.start_orbit_center_coor: np.ndarray = np.empty(shape=(0, 2), dtype=np.float64)

    def initiate(self, cmap: CircularMotionApproachParameters = CircularMotionApproachParameters()):

        # perform the initiation of the super class
        super().initiate(stfp=cmap)

        # store parameters of the initiation
        self.circular_motion_approach_parameters = cmap

        # copy coordinates of features instances
        features_instances_start_coor = np.copy(self.spatial_standard_placement.features_instances_coor)

        # calculate mean position of each co-location's instance
        _, inverse, counts = np.unique(self.collocations_clumpy_instances_global_ids, return_inverse=True, return_counts=True)
        collocations_instances_start_coor_mean = np.empty(shape=(self.collocations_clumpy_instances_global_sum, 2), dtype=np.float64)
        collocations_instances_start_coor_mean[:, 0] = np.bincount(inverse, features_instances_start_coor[:, 0]) / counts
        collocations_instances_start_coor_mean[:, 1] = np.bincount(inverse, features_instances_start_coor[:, 1]) / counts

        # draw angle value at the starting time point of the every circular orbit of the every co-location's instance
        collocations_instances_start_angle = np.random.uniform(high=2 * np.pi, size=(cmap.circle_chain_size, self.collocations_clumpy_instances_global_sum))

        # draw length of the orbit radius of the co-locations instances from uniform distribution according to boundaries values passed in parameters
        collocations_instances_radius_length = np.random.uniform(low=cmap.circle_r_min, high=cmap.circle_r_max, size=(cmap.circle_chain_size, self.collocations_clumpy_instances_global_sum))

        # ---begin--- determine circular orbits' centers of all co-locations' instances
        # calculate position determined by each of circular orbit - position calculated in reference system of the given circular orbit center (per co-location instance)
        circle_delta_x = collocations_instances_radius_length * np.cos(collocations_instances_start_angle)
        circle_delta_y = collocations_instances_radius_length * np.sin(collocations_instances_start_angle)

        # calculate cumulative sum of inverted circle delta - "the position delta from i-th circular orbit"
        circle_delta_x_cumsum = np.cumsum(circle_delta_x[::-1], axis=0)[::-1]
        circle_delta_y_cumsum = np.cumsum(circle_delta_y[::-1], axis=0)[::-1]

        # find starting point of every circular orbit by subtracting cumulative deltas from mean position of co-locations' instances
        collocations_instances_orbit_centers_coor = np.empty(shape=(cmap.circle_chain_size, self.collocations_clumpy_instances_global_sum, 2), dtype=np.float64)
        collocations_instances_orbit_centers_coor[:, :, 0] = -circle_delta_x_cumsum
        collocations_instances_orbit_centers_coor[:, :, 1] = -circle_delta_y_cumsum
        collocations_instances_orbit_centers_coor += collocations_instances_start_coor_mean[None, :, :]
        # ----end---- determine circular orbits' centers of all co-locations' instances

        # ---begin--- determine circular orbits' centers of all features' instances
        # determine circular orbits' centers of the feature instance which belong to the given co-location's instance
        features_instances_orbit_centers_coor = collocations_instances_orbit_centers_coor[:, self.collocations_clumpy_instances_global_ids]

        # calculate random position displacement of every circular orbit of the given feature instance according to the 'center_noise_displacement' parameter value
        orbit_center_displacement_theta = np.random.uniform(high=2 * np.pi, size=(cmap.circle_chain_size, self.features_instances_sum))
        orbit_center_displacement_r = cmap.center_noise_displacement * np.sqrt(np.random.random(size=(cmap.circle_chain_size, self.features_instances_sum)))
        orbit_center_displacement_coor = np.empty(shape=(cmap.circle_chain_size, self.features_instances_sum, 2), dtype=np.float64)
        orbit_center_displacement_coor[:, :, 0] = orbit_center_displacement_r * np.cos(orbit_center_displacement_theta)
        orbit_center_displacement_coor[:, :, 1] = orbit_center_displacement_r * np.sin(orbit_center_displacement_theta)

        # displace every circular orbit with calculated noise
        features_instances_orbit_centers_coor += orbit_center_displacement_coor
        # ----end---- determine circular orbits' centers of all features' instances

        # remember starting position of calculating orbital position of every feature instance
        self.start_orbit_center_coor = features_instances_orbit_centers_coor[0]

        # ---begin--- recalculate radius length of the every circular orbit after noise displacement of centers
        # prepare calculation of coordinates difference by concatenating orbit centers coordinates and features instances coordinates
        before_coor_diff = np.append(arr=features_instances_orbit_centers_coor, values=features_instances_start_coor[None, :, :], axis=0)

        # calculate coordinates difference between consecutive circular orbits - with np.diff
        coor_diff = np.diff(a=before_coor_diff, axis=0)

        # calculate length of the orbit radius
        self.radius_length = np.sqrt(np.sum(a=coor_diff ** 2, axis=-1))
        # ----end---- recalculate radius length of the every circular orbit after noise displacement of centers

        # recalculate start angle of the every circular orbit after noise displacement of centers
        self.start_angle = np.arctan2(coor_diff[:, :, 1], coor_diff[:, :, 0])

        # draw angular velocity values of the co-locations instances from uniform distribution according to boundaries values passed in parameters
        collocations_instances_angular_velocity = np.random.uniform(low=cmap.omega_min, high=cmap.omega_max, size=(cmap.circle_chain_size, self.collocations_clumpy_instances_global_sum))

        # determine angular velocity values of the features which belong to the given co-location's instance
        self.angular_velocity = collocations_instances_angular_velocity[:, self.collocations_clumpy_instances_global_ids]

        # determine direction of the rotation - features of the given co-location have the same direction
        collocations_instances_angular_velocity_clockwise = np.random.randint(2, size=(cmap.circle_chain_size, self.collocations_clumpy_instances_global_sum), dtype=bool)
        features_instances_angular_velocity_clockwise = collocations_instances_angular_velocity_clockwise[:, self.collocations_clumpy_instances_global_ids]
        self.angular_velocity[features_instances_angular_velocity_clockwise] *= -1
