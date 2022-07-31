import numpy as np

from oop.BasicInitiation import BasicInitiation
from oop.StandardParameters import StandardParameters
from oop.TravelApproachParameters import TravelApproachParameters


class StandardInitiation(BasicInitiation):
    """
    The class of a `SpatioTemporalStandardGenerator` initiation. Object of this class stores all initial data, which is required to generate spatio-temporal data
    in each time frame.
    """

    def __init__(self):
        """
        Construct empty object of the `StandardInitiation` class.
        """

        super().__init__()

        self.standard_parameters: StandardParameters = StandardParameters()
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
        sp: TravelApproachParameters
            The object of class `StandardParameters`, which holds all the required parameters of the `SpatioTemporalStandardGenerator` generator.
            Its attributes will be used to initialize required data.
        """

        # perform the initiation of the super class
        super().initiate(bp=sp)

        # store parameters of the initiation
        self.standard_parameters = sp

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
