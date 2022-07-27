import numpy as np

from oop.StandardParameters import StandardParameters
from oop.TravelApproachEnums import StepLengthMethod, StepAngleMethod


class TravelApproachParameters(StandardParameters):
    def __init__(
            self,
            step_length_mean: float = 10.0,
            step_length_method: StepLengthMethod = StepLengthMethod.UNIFORM,
            step_length_std_ratio: float = 0.5,
            step_angle_range_mean: float = np.pi / 9,
            step_angle_range_limit: float = np.pi / 2,
            step_angle_method: StepAngleMethod = StepAngleMethod.UNIFORM,
            step_angle_std_ratio: float = 1 / 3,
            waiting_time_frames: int = np.inf,
            **kwargs):
        super().__init__(**kwargs)
        self.step_length_mean = step_length_mean
        self.step_length_method = step_length_method
        self.step_length_std_ratio = step_length_std_ratio
        self.step_angle_range_mean = step_angle_range_mean
        self.step_angle_range_limit = step_angle_range_limit
        self.step_angle_method = step_angle_method
        self.step_angle_std_ratio = step_angle_std_ratio
        self.waiting_time_frames = waiting_time_frames
