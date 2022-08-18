import numpy as np

from algorithms.enums.OptimalDistanceApproachEnums import MassMethod, VelocityMethod
from algorithms.parameters.StandardTimeFrameParameters import StandardTimeFrameParameters


class OptimalDistanceApproachParameters(StandardTimeFrameParameters):
    def __init__(
            self,
            time_unit: float = 1.0,
            approx_steps_number: int = 10,
            k_optimal_distance: float = 1.0,
            k_force: float = 1.0,
            force_limit: float = 5.0,
            velocity_limit: float = 5.0,
            faraway_limit_ratio: float = np.sqrt(2) / 2,
            mass_method: MassMethod = MassMethod.CONSTANT,
            mass_mean: float = 1.0,
            mass_normal_std_ratio: float = 1 / 5,
            velocity_method: VelocityMethod = VelocityMethod.CONSTANT,
            velocity_mean: float = 0.0,
            **kwargs):

        super().__init__(**kwargs)

        # check 'time_unit' value
        if time_unit <= 0.0:
            time_unit = 1.0

        # check 'approx_steps_number' value
        if approx_steps_number <= 0:
            approx_steps_number = 1

        # check 'k_optimal_distance' value
        if k_optimal_distance <= 0.0:
            k_optimal_distance = 1.0

        # check 'k_force' value
        if k_force <= 0.0:
            k_force = 1.0

        # check 'force_limit' value
        if force_limit <= 0.0:
            force_limit = 5.0

        # check 'velocity_limit' value
        if velocity_limit <= 0.0:
            velocity_limit = 5.0

        # check 'faraway_limit_ratio' value
        if faraway_limit_ratio <= 0:
            faraway_limit_ratio = np.sqrt(2) / 2

        # check 'mass_mean' value
        if mass_mean <= 0.0:
            mass_mean = 1.0

        # check 'mass_normal_std_ratio' value
        if mass_normal_std_ratio < 0.0:
            mass_normal_std_ratio = 0.0

        # check 'velocity_mean' value
        if velocity_mean < 0.0:
            velocity_mean = 0.0

        self.time_unit = time_unit
        self.approx_steps_number = approx_steps_number
        self.k_optimal_distance = k_optimal_distance
        self.k_force = k_force
        self.force_limit = force_limit
        self.velocity_limit = velocity_limit
        self.faraway_limit_ratio = faraway_limit_ratio
        self.mass_method = mass_method
        self.mass_mean = mass_mean
        self.mass_normal_std_ratio = mass_normal_std_ratio
        self.velocity_method = velocity_method
        self.velocity_mean = velocity_mean