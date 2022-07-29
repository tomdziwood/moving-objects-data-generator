import numpy as np

from oop.BasicInitiation import BasicInitiation
from oop.TravelApproachEnums import StepLengthMethod, StepAngleMethod
from oop.TravelApproachParameters import TravelApproachParameters


class TravelApproachInitiation(BasicInitiation):
    """
    The class of a `SpatioTemporalTravelApproachGenerator` initiation. Object of this class stores all initial data, which is required to generate spatio-temporal data
    in each time frame.

    Attributes
    ----------
    travel_approach_parameters : TravelApproachParameters
        The object of class `TravelApproachParameters`, which holds all the required parameters of the `SpatioTemporalTravelApproachGenerator` generator.

    collocations_instances_global_ids : np.ndarray
        The array's size is equal to the number of features' instances. The i-th value represents the global id of the co-location instance,
        to which the i-th feature instance belongs.

    collocations_instances_global_sum : int
        The number of all specified collocation global instances.

    collocations_instances_global_ids_repeats : np.ndarray
        The array's size is equal to the number of co-locations instances. The i-th value represents the number of features' instances,
        which belong to the i-th co-location instance.

    collocations_instances_destination_coor : np.ndarray
        The array's size is equal to the number of co-locations instances. The i-th value represents the coordinates of the destination point of the i-th co-location instance.

    features_instances_destination_coor : np.ndarray
        The array's size is equal to the number of features' instances. The i-th value represents the coordinates of the destination point of the i-th feature instance.

    features_instances_destination_reached : np.ndarray
        The array's size is equal to the number of features' instances. The i-th value tells if i-th feature instance has reached its destination point.

    collocations_instances_waiting_countdown : np.ndarray
        The array's size is equal to the number of co-locations instances. The i-th value represents remaining time frames of waiting for the rest
        of the features' instances of the i-th co-location instance to reach their destination point. When counter is inactive, the value is equal ``-1``.

    features_step_length_mean : np.ndarray
        The array's size is equal to the number of features' types. The i-th value represents mean value of the step's length of a feature's instance
        of the i-th feature's type.

    features_step_length_uniform_min : np.ndarray
        The array's size is equal to the number of features' types. The array is initialized, when the generator's parameter ``step_length_method``
        is equal to ``StepLengthMethod.UNIFORM``. The i-th value represents lower boundary of the uniform distribution of the step's length of the i-th feature's type.

    features_step_length_uniform_max : np.ndarray
        The array's size is equal to the number of features' types. The array is initialized, when the generator's parameter ``step_length_method``
        is equal to ``StepLengthMethod.UNIFORM``. The i-th value represents upper boundary of the uniform distribution of the step's length of the i-th feature's type.

    features_step_length_normal_std : np.ndarray
        The array's size is equal to the number of features' types. The array is initialized, when the generator's parameter ``step_length_method``
        is equal to ``StepLengthMethod.NORMAL``. The i-th value represents standard deviation of the normal distribution of the step's length of the i-th feature's type.

    features_step_angle_range : np.ndarray
        The array's size is equal to the number of features' types. The i-th value represents the maximum step's angle to the direction of destination point
        of the i-th feature's type.

    features_step_angle_normal_std : np.ndarray
        The array's size is equal to the number of features' types. The array is initialized, when the generator's parameter ``step_angle_method``
        is equal to ``StepAngleMethod.NORMAL``. The i-th value represents the standard deviation of the normal distribution of the step's angle
        to the direction of destination point of the i-th feature's type.

    """

    def __init__(self):
        """
        Construct empty object of the `TravelApproachInitiation` class.
        """

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
        self.features_step_length_uniform_min: np.ndarray = np.array([], dtype=np.float64)
        self.features_step_length_uniform_max: np.ndarray = np.array([], dtype=np.float64)
        self.features_step_length_normal_std: np.ndarray = np.array([], dtype=np.float64)
        self.features_step_angle_range: np.ndarray = np.array([], dtype=np.float64)
        self.features_step_angle_normal_std: np.ndarray = np.array([], dtype=np.float64)

    def initiate(self, tap: TravelApproachParameters = TravelApproachParameters()):
        """
        Initiate required data to generate spatio-temporal data in each time frame.

        Parameters
        ----------
        tap: TravelApproachParameters
            The object of class `TravelApproachParameters`, which holds all the required parameters of the `SpatioTemporalTravelApproachGenerator` generator.
            Its attributes will be used to initialize required data.
        """

        super().initiate(bp=tap)

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
            self.features_step_length_uniform_min = self.features_step_length_mean * tap.step_length_uniform_low_to_mean_ratio
            print("features_step_length_uniform_min=%s" % str(self.features_step_length_uniform_min))
            self.features_step_length_uniform_max = 2 * self.features_step_length_mean - self.features_step_length_uniform_min
            print("features_step_length_uniform_max=%s" % str(self.features_step_length_uniform_max))
        elif tap.step_length_method == StepLengthMethod.NORMAL:
            self.features_step_length_normal_std = tap.step_length_normal_std_ratio * self.features_step_length_mean
            print("features_step_length_normal_std=%s" % str(self.features_step_length_normal_std))

        # determine travel step angle settings of each feature type
        self.features_step_angle_range = np.random.gamma(shape=tap.step_angle_range_mean, scale=1.0, size=self.features_sum)
        self.features_step_angle_range[self.features_step_angle_range < -tap.step_angle_range_limit] = -tap.step_angle_range_limit
        self.features_step_angle_range[self.features_step_angle_range > tap.step_angle_range_limit] = tap.step_angle_range_limit
        print("features_step_angle_range=%s" % str(self.features_step_angle_range))
        if tap.step_angle_method == StepAngleMethod.NORMAL:
            self.features_step_angle_normal_std = tap.step_angle_normal_std_ratio * self.features_step_angle_range
            print("features_step_angle_normal_std=%s" % str(self.features_step_angle_normal_std))
