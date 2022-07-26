import numpy as np

from oop.StandardParameters import StandardParameters
from oop.TravelApproachEnums import StepLengthMethod, StepAngleMethod


class TravelApproachParameters(StandardParameters):
    def __init__(
            self,
            step_length_mean: float = 10.0,
            step_length_method: StepLengthMethod = StepLengthMethod.UNIFORM,
            step_length_std_ratio: float = 0.5,
            step_angle_range: float = np.pi / 18,
            step_angle_method: StepAngleMethod = StepAngleMethod.UNIFORM,
            step_angle_std_ratio: float = 1 / 3,
            **kwargs):
        super().__init__(**kwargs)
        self.step_length_mean = step_length_mean
        self.step_length_method = step_length_method
        self.step_length_std_ratio = step_length_std_ratio
        self.step_angle_range = step_angle_range
        self.step_angle_method = step_angle_method
        self.step_angle_std_ratio = step_angle_std_ratio
