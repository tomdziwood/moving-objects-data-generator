import numpy as np

from oop.FeatureInteractionModes import IdenticalFeaturesInteractionMode, DifferentFeaturesInteractionMode
from oop.BasicParameters import BasicParameters


class StaticInteractionApproachParameters(BasicParameters):
    def __init__(
            self,
            time_unit: float = 1.0,
            distance_unit: float = 1.0,
            approx_steps_number: int = 10,
            min_dist: float = 0.1,
            max_force: float = np.inf,
            k_force: float = 1.0,
            mass_param: float = 1.0,
            velocity_param: float = 0.0,
            faraway_limit: float = np.inf,
            identical_features_interaction_mode: IdenticalFeaturesInteractionMode = IdenticalFeaturesInteractionMode.ATTRACT,
            different_features_interaction_mode: DifferentFeaturesInteractionMode = DifferentFeaturesInteractionMode.ATTRACT,
            **kwargs):
        super().__init__(**kwargs)
        self.time_unit = time_unit
        self.distance_unit = distance_unit
        self.approx_steps_number = approx_steps_number
        self.min_dist = min_dist
        self.max_force = max_force
        self.k_force = k_force
        self.mass_param = mass_param
        self.velocity_param = velocity_param
        self.faraway_limit = faraway_limit
        self.identical_features_interaction_mode = identical_features_interaction_mode
        self.different_features_interaction_mode = different_features_interaction_mode
