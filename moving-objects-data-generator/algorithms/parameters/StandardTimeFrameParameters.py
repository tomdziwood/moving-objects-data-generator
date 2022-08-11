from algorithms.parameters.BasicParameters import BasicParameters


class StandardTimeFrameParameters(BasicParameters):
    def __init__(
            self,
            spatial_prevalent_ratio: float = 1.0,
            spatial_prevalence_threshold: float = 1.0,
            **kwargs):

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
