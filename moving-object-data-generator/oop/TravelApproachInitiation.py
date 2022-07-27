import numpy as np

from oop.StandardInitiation import StandardInitiation
from oop.TravelApproachEnums import StepLengthMethod, StepAngleMethod
from oop.TravelApproachParameters import TravelApproachParameters


class TravelApproachInitiation(StandardInitiation):
    def __init__(self):
        super().__init__()

        self.travel_approach_parameters: TravelApproachParameters = TravelApproachParameters()
        self.collocations_instances_global_ids: np.ndarray = np.array([], dtype=np.int32)
        self.collocations_instances_global_sum: int = 0
        self.collocations_instances_global_ids_repeats: np.ndarray = np.array([], dtype=np.int32)
        self.collocations_instances_destination_coor: np.ndarray = np.empty(shape=(0, 2), dtype=np.float64)
        self.features_instances_destination_coor: np.ndarray = np.empty(shape=(0, 2), dtype=np.float64)
        self.features_instances_destination_reached: np.ndarray = np.array([], dtype=bool)
        self.collocations_instances_waiting_countdown: np.ndarray = np.array([], dtype=np.int32)
        self.features_step_length_mean: np.ndarray = np.array([], dtype=np.float64)
        self.features_step_length_max: np.ndarray = np.array([], dtype=np.float64)
        self.features_step_length_std: np.ndarray = np.array([], dtype=np.float64)
        self.features_step_angle_range = np.array([], dtype=np.float64)
        self.features_step_angle_std = np.array([], dtype=np.float64)

    def initiate(self, tap: TravelApproachParameters = TravelApproachParameters()):
        super().initiate(sp=tap)

        self.travel_approach_parameters = tap

        # prepare array of co-locations instances global ids to which features belong
        self.collocations_instances_global_ids = np.array([], dtype=np.int32)
        last_collocation_instance_global_id = 0
        for i_colloc in range(tap.n_colloc * tap.m_overlap):
            i_colloc_collocations_instances_global_ids = np.repeat(
                a=np.arange(last_collocation_instance_global_id, last_collocation_instance_global_id + self.collocation_instances_counts[i_colloc]),
                repeats=self.collocation_lengths[i_colloc]
            )
            self.collocations_instances_global_ids = np.concatenate((self.collocations_instances_global_ids, i_colloc_collocations_instances_global_ids))
            last_collocation_instance_global_id += self.collocation_instances_counts[i_colloc]

        # sum of all specified collocation global instances
        self.collocations_instances_global_sum = last_collocation_instance_global_id + self.collocation_noise_features_instances_sum + tap.ndfn
        print("collocations_instances_global_sum=%d" % self.collocations_instances_global_sum)

        # every single noise feature instance is assigned to the unique individual co-location instance global id
        self.collocations_instances_global_ids = np.concatenate((
            self.collocations_instances_global_ids,
            np.arange(last_collocation_instance_global_id, self.collocations_instances_global_sum)
        ))

        # save number of repeats of the consecutive co-locations instances global ids - required data at detection of reached co-locations
        self.collocations_instances_global_ids_repeats = np.concatenate((
            np.repeat(a=self.collocation_lengths, repeats=self.collocation_instances_counts),
            np.ones(shape=self.collocation_noise_features_instances_sum + tap.ndfn, dtype=np.int32)
        ))

        # set destination point of every feature instance
        self.collocations_instances_destination_coor = np.random.randint(low=self.area_in_cell_dim, size=(self.collocations_instances_global_sum, 2))
        self.collocations_instances_destination_coor *= tap.cell_size
        self.collocations_instances_destination_coor = self.collocations_instances_destination_coor.astype(dtype=np.float64)
        self.features_instances_destination_coor = self.collocations_instances_destination_coor[self.collocations_instances_global_ids]
        self.features_instances_destination_coor += np.random.uniform(high=tap.cell_size, size=self.features_instances_destination_coor.shape)

        # create boolean array which tells if the given feature instance reached its own destination point
        self.features_instances_destination_reached = np.zeros(shape=self.features_instances_sum, dtype=bool)

        # create countdown array which tells how many time frames remain of waiting for the rest of features instances of the given co-location instances
        self.collocations_instances_waiting_countdown = np.full(shape=self.collocations_instances_global_sum, fill_value=-1, dtype=np.int32)

        # determine travel step length settings of each feature type
        self.features_step_length_mean = np.random.gamma(shape=tap.step_length_mean, scale=1.0, size=self.features_sum)
        print("features_step_length_mean=%s" % str(self.features_step_length_mean))
        if tap.step_length_method == StepLengthMethod.UNIFORM:
            self.features_step_length_max = self.features_step_length_mean * 2
            print("features_step_length_max=%s" % str(self.features_step_length_max))
        elif tap.step_length_method == StepLengthMethod.NORMAL:
            self.features_step_length_std = tap.step_length_std_ratio * self.features_step_length_mean
            print("features_step_length_std=%s" % str(self.features_step_length_std))

        # determine travel step angle settings of each feature type
        self.features_step_angle_range = np.random.gamma(shape=tap.step_angle_range_mean, scale=1.0, size=self.features_sum)
        self.features_step_angle_range[self.features_step_angle_range < -tap.step_angle_range_limit] = -tap.step_angle_range_limit
        self.features_step_angle_range[self.features_step_angle_range > tap.step_angle_range_limit] = tap.step_angle_range_limit
        print("features_step_angle_range=%s" % str(self.features_step_angle_range))
        if tap.step_angle_method == StepAngleMethod.NORMAL:
            self.features_step_angle_std = tap.step_angle_std_ratio * self.features_step_angle_range
            print("features_step_angle_std=%s" % str(self.features_step_angle_std))
