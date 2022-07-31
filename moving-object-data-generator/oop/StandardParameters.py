from oop.BasicParameters import BasicParameters


class StandardParameters(BasicParameters):
    def __init__(
            self,
            persistent_ratio: float = 0.5,
            spatial_prevalence_threshold: float = 0.5,
            time_prevalence_threshold: float = 0.5,
            **kwargs):

        super().__init__(**kwargs)

        self.persistent_ratio = persistent_ratio
        self.spatial_prevalence_threshold = spatial_prevalence_threshold
        self.time_prevalence_threshold = time_prevalence_threshold
