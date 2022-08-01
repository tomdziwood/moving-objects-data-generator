from oop.BasicParameters import BasicParameters


class StandardParameters(BasicParameters):
    def __init__(
            self,
            persistent_ratio: float = 0.5,
            spatial_prevalence_threshold: float = 0.5,
            time_prevalence_threshold: float = 0.5,
            **kwargs):

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
