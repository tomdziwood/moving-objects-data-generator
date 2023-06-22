import numpy as np

from algorithms.initiation.BasicInitiation import BasicInitiation
from algorithms.parameters.StandardParameters import StandardParameters


class StandardInitiation(BasicInitiation):
    """
    The class of a `SpatioTemporalStandardGenerator` initiation. Object of this class stores all initial data, which is required to generate spatio-temporal data
    in each time frame.

    Attributes
    ----------
    standard_parameters : StandardParameters
        The object of class `StandardParameters`, which holds all the required parameters of the `SpatioTemporalStandardGenerator` generator.

    collocations_instances_number_spatial_prevalence_threshold : np.ndarray
        The array's size is equal to the number of co-locations. The i-th value represents the minimal number of the i-th co-location instances occurrences,
        which makes the co-location becomes spatial prevalent.

    persistent_collocations_sum : int
        The number of persistent co-locations. A persistent co-location is a co-location, whose time prevalence measure exceeds the time prevalence threshold.

    persistent_collocations_ids : np.ndarray
        The array's size is equal to the number of persistent co-locations. The array contains the ids of the co-locations, which has been chosen as persistent co-locations.

    collocations_persistence_flags : np.ndarray
        The array's size is equal to the number of co-locations. The array is a boolean vector, which indicates if the i-th co-location is persistent.

    transient_collocations_sum : int
        The number of transient co-locations. A transient co-location is a co-location, whose time prevalence measure does not exceed the time prevalence threshold.

    transient_collocations_ids : np.ndarray
        The array's size is equal to the number of transient co-locations. The array contains the ids of the co-locations, which has been chosen as transient co-locations.

    collocations_transience_flags : np.ndarray
        The array's size is equal to the number of co-locations. The array is a boolean vector, which indicates if the i-th co-location is transient.
    """

    def __init__(self):
        """
        Construct empty object of the `StandardInitiation` class.
        """

        super().__init__()

        self.standard_parameters: StandardParameters = StandardParameters()
        self.collocations_instances_number_spatial_prevalence_threshold: np.ndarray = np.array([], dtype=np.int32)
        self.persistent_collocations_sum: int = 0
        self.persistent_collocations_ids: np.ndarray = np.array([], dtype=np.int32)
        self.collocations_persistence_flags: np.ndarray = np.array([], dtype=bool)
        self.transient_collocations_sum: int = 0
        self.transient_collocations_ids: np.ndarray = np.array([], dtype=np.int32)
        self.collocations_transience_flags: np.ndarray = np.array([], dtype=bool)

    def initiate(self, sp: StandardParameters = StandardParameters()):
        """
        Initiate required data to generate spatio-temporal data in each time frame.

        Parameters
        ----------
        sp : StandardParameters
            The object of class `StandardParameters`, which holds all the required parameters of the `SpatioTemporalStandardGenerator` generator.
            Its attributes will be used to initialize required data.
        """

        # perform the initiation of the super class
        super().initiate(bp=sp)

        # store parameters of the initiation
        self.standard_parameters = sp

        # determine the minimal number of the given co-location instances occurrences, which makes the co-location becomes spatial prevalent
        self.collocations_instances_number_spatial_prevalence_threshold = np.ceil(sp.spatial_prevalence_threshold * self.collocation_instances_counts).astype(np.int32)
        print("collocations_instances_number_spatial_prevalence_threshold=%s" % str(self.collocations_instances_number_spatial_prevalence_threshold))

        # calculate the number of persistent co-locations
        self.persistent_collocations_sum = int(self.collocations_sum * sp.persistent_ratio)
        print("persistent_collocations_sum=%d" % self.persistent_collocations_sum)

        # chose persistent co-locations' ids
        self.persistent_collocations_ids = np.random.choice(a=self.collocations_sum, size=self.persistent_collocations_sum, replace=False)
        self.persistent_collocations_ids.sort()
        print("persistent_collocations_ids=%s" % str(self.persistent_collocations_ids))

        # create boolean vector which tells if the given co-location pattern is persistent
        self.collocations_persistence_flags = np.zeros(shape=self.collocations_sum, dtype=bool)
        self.collocations_persistence_flags[self.persistent_collocations_ids] = True

        # calculate the number of transient co-locations
        self.transient_collocations_sum = self.collocations_sum - self.persistent_collocations_sum
        print("transient_collocations_sum=%d" % self.transient_collocations_sum)

        # create boolean vector which tells if the given co-location pattern is transient
        self.collocations_transience_flags = np.logical_not(self.collocations_persistence_flags)

        # chose transient co-locations' ids
        self.transient_collocations_ids = np.flatnonzero(self.collocations_transience_flags)
        print("transient_collocations_ids=%s" % str(self.transient_collocations_ids))
