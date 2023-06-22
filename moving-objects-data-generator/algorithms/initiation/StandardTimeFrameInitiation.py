import numpy as np

from algorithms.initiation.BasicInitiation import BasicInitiation
from algorithms.parameters.StandardTimeFrameParameters import StandardTimeFrameParameters
from algorithms.utils.SpatialStandardPlacement import SpatialStandardPlacement


class StandardTimeFrameInitiation(BasicInitiation):
    """
    The class of an initiation of many generator types. Object of this class stores all initial data, which is required to begin the spatio-temporal data generating process.

    Attributes
    ----------
    standard_time_frame_parameters : StandardTimeFrameParameters
        The object of class `StandardTimeFrameParameters`, which holds all the basic parameters used by the many classes of spatio-temporal data generators.

    collocations_instances_number_spatial_prevalence_threshold : np.ndarray
        The array's size is equal to the number of co-locations. The i-th value represents the minimal number of the i-th co-location instances occurrences,
        which makes the co-location becomes spatial prevalent.

    spatial_prevalent_collocations_sum : int
        The number of the co-locations which are chosen as spatial prevalent co-locations at the first time frame. The number is determined
        with a ''spatial_prevalent_ratio'' parameter value.

    spatial_prevalent_collocations_ids : np.ndarray
        The array's size is equal to the number of spatial prevalent co-locations. The array contains the ids of the co-locations,
        which has been chosen as spatial prevalent co-locations.

    collocations_spatial_prevalence_flags : np.ndarray
        The array's size is equal to the number of co-locations. The array is a boolean vector, which indicates if the i-th co-location is a spatial prevalent co-location
        at the first time frame.

    spatial_standard_placement : SpatialStandardPlacement
        The object, which holds all data of the objects placement at the first time frame.
    """

    def __init__(self):
        """
        Construct empty object of the `StandardTimeFrameInitiation` class.
        """

        super().__init__()

        self.standard_time_frame_parameters: StandardTimeFrameParameters = StandardTimeFrameParameters()
        self.collocations_instances_number_spatial_prevalence_threshold: np.ndarray = np.array([], dtype=np.int32)
        self.spatial_prevalent_collocations_sum: int = 0
        self.spatial_prevalent_collocations_ids: np.ndarray = np.array([], dtype=np.int32)
        self.collocations_spatial_prevalence_flags: np.ndarray = np.array([], dtype=bool)
        self.spatial_standard_placement: SpatialStandardPlacement = SpatialStandardPlacement()

    def initiate(self, stfp: StandardTimeFrameParameters = StandardTimeFrameParameters()):
        """
        Initiate required data to begin the spatio-temporal data generating process.

        Parameters
        ----------
        stfp: StandardTimeFrameParameters
            The object of class `StandardTimeFrameParameters`, which holds all the required parameters of the initiation.
            Its attributes will be used to start the spatio-temporal data generating process.
        """

        # perform the initiation of the super class
        super().initiate(bp=stfp)

        # store parameters of the initiation
        self.standard_time_frame_parameters = stfp

        # determine the minimal number of the given co-location instances occurrences, which makes the co-location becomes spatial prevalent
        self.collocations_instances_number_spatial_prevalence_threshold = np.ceil(stfp.spatial_prevalence_threshold * self.collocation_instances_counts).astype(np.int32)
        print("collocations_instances_number_spatial_prevalence_threshold=%s" % str(self.collocations_instances_number_spatial_prevalence_threshold))

        # calculate the number of spatial prevalent co-locations
        self.spatial_prevalent_collocations_sum = int(self.collocations_sum * stfp.spatial_prevalent_ratio)
        print("spatial_prevalent_collocations_sum=%d" % self.spatial_prevalent_collocations_sum)

        # chose spatial prevalent co-locations' ids
        self.spatial_prevalent_collocations_ids = np.random.choice(a=self.collocations_sum, size=self.spatial_prevalent_collocations_sum, replace=False)
        self.spatial_prevalent_collocations_ids.sort()
        print("spatial_prevalent_collocations_ids=%s" % str(self.spatial_prevalent_collocations_ids))

        # create boolean vector which tells if the given co-location pattern is spatial prevalent
        self.collocations_spatial_prevalence_flags = np.zeros(shape=self.collocations_sum, dtype=bool)
        self.collocations_spatial_prevalence_flags[self.spatial_prevalent_collocations_ids] = True

        # create class object, which holds all data of the objects placement
        self.spatial_standard_placement = SpatialStandardPlacement(bi=self, collocations_instances_number_spatial_prevalence_threshold=self.collocations_instances_number_spatial_prevalence_threshold)

        # perform placement of all the features instances
        self.spatial_standard_placement.place(collocations_spatial_prevalence_flags=self.collocations_spatial_prevalence_flags)

        # ---begin--- reindex global ids of co-locations' instances
        print("---begin--- reindex global ids of co-locations' instances")

        # create boolean vector which tells if the given co-locations instance do not need to be expanded
        collocations_instances_not_expanded_flag = np.concatenate((self.spatial_standard_placement.collocations_instances_spatial_prevalent_flags, np.ones(shape=self.collocation_noise_features_instances_sum + stfp.ndfn, dtype=bool)))

        # create array which contains repeats of expanding process
        expand_repeats = np.copy(self.collocations_instances_global_ids_repeats)
        expand_repeats[collocations_instances_not_expanded_flag] = 1

        # expand co-locations' instances' global ids repeats with correct values
        self.collocations_instances_global_ids_repeats[np.logical_not(collocations_instances_not_expanded_flag)] = 1
        self.collocations_instances_global_ids_repeats = np.repeat(a=self.collocations_instances_global_ids_repeats, repeats=expand_repeats)

        # recalculate sum of all specified collocation global instances
        self.collocations_instances_global_sum = self.collocations_instances_global_ids_repeats.size
        print("collocations_instances_global_sum=%d" % self.collocations_instances_global_sum)

        # prepare again array of co-locations instances global ids to which features belong
        self.collocations_instances_global_ids = np.repeat(a=np.arange(self.collocations_instances_global_sum), repeats=self.collocations_instances_global_ids_repeats)

        print("----end---- reindex global ids of co-locations' instances")
        # ----end---- reindex global ids of co-locations' instances

        # ---begin--- reindex global ids of co-locations' "clumpy" instances
        # important note: self.collocations_clumpy_instances_global_ids_repeats value is meaningless. It can not be used to recreate ids of feature instances, because they are no longer in not-decreasing order.
        print("---begin--- reindex global ids of co-locations' \"clumpy\" instances")

        if stfp.m_clumpy == 1:
            # m_clumpy parameter doesn't bring any changes in co-locations instances ids
            self.collocation_clumpy_instances_counts = self.collocation_instances_counts
            self.collocations_clumpy_instances_global_sum = self.collocations_instances_global_sum
            self.collocations_clumpy_instances_global_ids_repeats = self.collocations_instances_global_ids_repeats
            self.collocations_clumpy_instances_global_ids = self.collocations_instances_global_ids
        else:
            # create boolean vector which tells if the given feature instance do not need to have a new co-location's id
            features_instances_spatial_prevalent_flags = np.concatenate((self.spatial_standard_placement.features_instances_spatial_prevalent_flags, np.ones(shape=self.collocation_noise_features_instances_sum + stfp.ndfn, dtype=bool)))

            # create boolean vector which tells if the given feature instance need to have a new co-location's id
            features_instances_not_spatial_prevalent_flags = np.logical_not(features_instances_spatial_prevalent_flags)

            # calculate number of new co-locations' ids
            new_collocation_indices_sum = features_instances_not_spatial_prevalent_flags.sum()

            # assign new co-locations' ids to the requested features instances
            self.collocations_clumpy_instances_global_ids[features_instances_not_spatial_prevalent_flags] = np.arange(self.collocations_clumpy_instances_global_sum, self.collocations_clumpy_instances_global_sum + new_collocation_indices_sum)

            # ---begin--- enumerate co-locations' ids again with consecutive integer values
            # call unique function on currently reassigned ids
            unique, inverse_ids = np.unique(self.collocations_clumpy_instances_global_ids, return_inverse=True)

            # actual sum of co-locations' clumpy instances is determined by the number of unique values
            self.collocations_clumpy_instances_global_sum = unique.size
            print("collocations_clumpy_instances_global_sum=%d" % self.collocations_clumpy_instances_global_sum)

            # indices of the unique array, which can be used to reconstruct original array, becomes actual co-locations' clumpy instances ids of consecutive features instances
            self.collocations_clumpy_instances_global_ids = inverse_ids

            # ----end---- enumerate co-locations' ids again with consecutive integer values

        print("----end---- reindex global ids of co-locations' \"clumpy\" instances")
        # ----end---- reindex global ids of co-locations' instances
