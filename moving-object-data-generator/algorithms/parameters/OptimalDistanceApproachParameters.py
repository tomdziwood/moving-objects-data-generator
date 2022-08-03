import numpy as np

from algorithms.enums.OptimalDistanceApproachEnums import MassMode, VelocityMode
from algorithms.parameters.BasicParameters import BasicParameters


class OptimalDistanceApproachParameters(BasicParameters):
    def __init__(
            self,
            time_unit: float = 1.0,
            approx_steps_number: int = 10,
            k_optimal_distance: float = 1.0,
            k_force: float = 1.0,
            force_limit: float = 5.0,
            mass_mode: MassMode = MassMode.CONSTANT,
            mass_mean: float = 1.0,
            mass_normal_std_ratio: float = 1 / 5,
            velocity_mode: VelocityMode = VelocityMode.CONSTANT,
            velocity_mean: float = 0.0,
            faraway_limit: float = np.inf,
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
            force_limit = np.inf

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
        self.approx_steps_number = approx_steps_number
        self.k_optimal_distance = k_optimal_distance
        self.k_force = k_force
        self.force_limit = force_limit
        self.mass_mode = mass_mode
        self.mass_mean = mass_mean
        self.mass_normal_std_ratio = mass_normal_std_ratio
        self.velocity_mode = velocity_mode
        self.velocity_mean = velocity_mean
        self.faraway_limit = faraway_limit
