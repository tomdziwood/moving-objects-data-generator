import numpy as np

from algorithms.initiation.BasicInitiation import BasicInitiation
from algorithms.parameters.StandardTimeFrameParameters import StandardTimeFrameParameters
from algorithms.utils.SpatialStandardPlacement import SpatialStandardPlacement


class StandardTimeFrameInitiation(BasicInitiation):
    """
    The class of an initiation of many generator types. Object of this class stores all initial data, which is required to begin the spatio-temporal data generating proces.
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
        Initiate required data to begin the spatio-temporal data generating proces.

        Parameters
        ----------
        stfp: StandardTimeFrameParameters
            The object of class `StandardTimeFrameParameters`, which holds all the required parameters of the initiation.
            Its attributes will be used to start the spatio-temporal data generating proces.
        """

        # perform the initiation of the super class
        super().initiate(bp=stfp)

        # store parameters of the initiation
        self.standard_time_frame_parameters = stfp

        # determine the minimal number of the given co-location instances occurrence, which makes the co-location becomes spatial prevalent
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

        self.spatial_standard_placement = SpatialStandardPlacement(bi=self, collocations_instances_number_spatial_prevalence_threshold=self.collocations_instances_number_spatial_prevalence_threshold)

        self.spatial_standard_placement.place(collocations_spatial_prevalence_flags=self.collocations_spatial_prevalence_flags)
