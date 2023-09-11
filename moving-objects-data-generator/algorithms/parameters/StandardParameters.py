from algorithms.parameters.BasicParameters import BasicParameters


class StandardParameters(BasicParameters):
    """
    The class represents set of parameters used by the `SpatioTemporalStandardGenerator` class of a spatio-temporal data generator.
    """

    def __init__(
            self,
            persistent_ratio: float = 0.5,
            spatial_prevalence_threshold: float = 0.5,
            time_prevalence_threshold: float = 0.5,
            **kwargs):
        """
        Construct an object which holds all the required parameters of the `SpatioTemporalStandardGenerator` class of a spatio-temporal data generator.

        Parameters
        ----------
        persistent_ratio : float
            The ratio of the co-location patterns which are chosen as a persistent co-location pattern. This ratio is used to determine the exact number
            of all persistent co-location patterns. The parameter is also known as the "l_persistent".

        spatial_prevalence_threshold : float
            The spatial prevalence threshold. It is used to determine the minimal number of co-location pattern instances occurrences, which makes the given co-location
            pattern becomes a spatial prevalent co-location. The parameter is also known as the "θ_p".

        time_prevalence_threshold : float
            The time prevalence threshold. It is used to determine the minimal number of time frames, when the given co-location pattern is spatial prevalent,
            so the co-location pattern could be time prevalent. The parameter is also known as the "θ_t".

        kwargs
            Other parameters passed to the super constructor of the derived class `BasicParameters`.
        """

        super().__init__(**kwargs)

        # check 'persistent_ratio' value
        if persistent_ratio < 0.0:
            persistent_ratio = 0.0
        if persistent_ratio > 1.0:
            persistent_ratio = 1.0

        # check 'spatial_prevalence_threshold' value
        if spatial_prevalence_threshold <= 0.0 or spatial_prevalence_threshold > 1.0:
            spatial_prevalence_threshold = 1.0

        # check 'time_prevalence_threshold' value
        if time_prevalence_threshold < 0.0:
            time_prevalence_threshold = 0.0
        if time_prevalence_threshold > 1.0:
            time_prevalence_threshold = 1.0

        self.persistent_ratio = persistent_ratio
        self.spatial_prevalence_threshold = spatial_prevalence_threshold
        self.time_prevalence_threshold = time_prevalence_threshold
