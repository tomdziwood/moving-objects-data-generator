import numpy as np

from oop.StaticInteractionApproachEnums import IdenticalFeaturesInteractionMode, DifferentFeaturesInteractionMode, MassMode, VelocityMode
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
            mass_mode: MassMode = MassMode.CONSTANT,
            mass_mean: float = 1.0,
            mass_normal_std_ratio: float = 1 / 5,
            velocity_mode: VelocityMode = VelocityMode.CONSTANT,
            velocity_mean: float = 0.0,
            faraway_limit: float = np.inf,
            identical_features_interaction_mode: IdenticalFeaturesInteractionMode = IdenticalFeaturesInteractionMode.ATTRACT,
            different_features_interaction_mode: DifferentFeaturesInteractionMode = DifferentFeaturesInteractionMode.ATTRACT,
            **kwargs):

        super().__init__(**kwargs)

        # check 'time_unit' value
        if time_unit <= 0.0:
            time_unit = 1.0

        # check 'distance_unit' value
        if distance_unit <= 0.0:
            distance_unit = 1.0

        # check 'approx_steps_number' value
        if approx_steps_number <= 0:
            approx_steps_number = 1

        # check 'min_dist' value
        if min_dist < 0.0:
            min_dist = 0.0

        # check 'max_force' value
        if max_force <= 0.0:
            max_force = np.inf

        # check 'k_force' value
        if k_force <= 0.0:
            k_force = 1.0

        # check 'mass_mean' value
        if mass_mean <= 0.0:
            mass_mean = 1.0

        # check 'mass_normal_std_ratio' value
        if mass_normal_std_ratio < 0.0:
            mass_normal_std_ratio = 0.0

        # check 'velocity_mean' value
        if velocity_mean < 0.0:
            velocity_mean = 0.0

        # check 'faraway_limit' value
        if faraway_limit <= 0:
            faraway_limit = np.inf

        self.time_unit = time_unit
        self.distance_unit = distance_unit
        self.approx_steps_number = approx_steps_number
        self.min_dist = min_dist
        self.max_force = max_force
        self.k_force = k_force
        self.mass_mode = mass_mode
        self.mass_mean = mass_mean
        self.mass_normal_std_ratio = mass_normal_std_ratio
        self.velocity_mode = velocity_mode
        self.velocity_mean = velocity_mean
        self.faraway_limit = faraway_limit
        self.identical_features_interaction_mode = identical_features_interaction_mode
        self.different_features_interaction_mode = different_features_interaction_mode
