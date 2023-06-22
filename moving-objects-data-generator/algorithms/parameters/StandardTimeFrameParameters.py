from algorithms.parameters.BasicParameters import BasicParameters


class StandardTimeFrameParameters(BasicParameters):
    """
    The class represents a set of basic parameters used by the many classes of spatio-temporal data generators. These parameters are mainly used at the early stage
    of the data generation, when the localization of the objects needs to be defined at the first time frame. This class is overridden by other classes of parameters,
    which are used later by the specific versions of spatio-temporal data generators.
    """

    def __init__(
            self,
            spatial_prevalent_ratio: float = 1.0,
            spatial_prevalence_threshold: float = 1.0,
            **kwargs):
        """
        Construct object which holds all the basic parameters of the many classes of spatio-temporal data generators. These parameters are required while calculating
        the localization of the objects at the first time frame.

        Parameters
        ----------
        spatial_prevalent_ratio : float
            The ratio of the co-location patterns which are chosen as a spatial prevalent co-location pattern at the first time frame. This ratio is used to determine
            the exact number of all spatial prevalent co-location patterns.

        spatial_prevalence_threshold : float
            The spatial prevalence threshold. It is used to determine the minimal number of co-location pattern instances occurrences, which makes the given co-location
            pattern becomes a spatial prevalent co-location at the first time frame.

        kwargs
            Other parameters passed to the super constructor of the derived class `BasicParameters`.
        """

        super().__init__(**kwargs)

        # check 'spatial_prevalent_ratio' value
        if spatial_prevalent_ratio < 0.0:
            spatial_prevalent_ratio = 0.0
        if spatial_prevalent_ratio > 1.0:
            spatial_prevalent_ratio = 1.0

        # check 'spatial_prevalence_threshold' value
        if spatial_prevalence_threshold <= 0.0 or spatial_prevalence_threshold > 1.0:
            spatial_prevalence_threshold = 1.0

        self.spatial_prevalent_ratio = spatial_prevalent_ratio
        self.spatial_prevalence_threshold = spatial_prevalence_threshold
