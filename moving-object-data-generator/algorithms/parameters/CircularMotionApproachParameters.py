import numpy as np

from algorithms.parameters.BasicParameters import BasicParameters


class CircularMotionApproachParameters(BasicParameters):
    def __init__(
            self,
            circle_chain_size: int = 2,
            omega_min: float = 2 * np.pi / 200,
            omega_max: float = 2 * np.pi / 25,
            circle_r_min: float = 20.0,
            circle_r_max: float = 200.0,
            center_noise_displacement: float = 5.0,
            **kwargs):

        super().__init__(**kwargs)

        # check 'circle_chain_size' value
        if circle_chain_size < 1:
            circle_chain_size = 1

        # check 'omega_min' value
        if omega_min < 0.0:
            omega_min = 0.0

        # check 'omega_max' value
        if omega_max < omega_min:
            omega_max = omega_min

        # check 'circle_r_min' value
        if circle_r_min <= 0.0:
            circle_r_min = 20.0

        # check 'omega_max' value
        if circle_r_max < circle_r_min:
            circle_r_max = circle_r_min

        # check 'center_noise_displacement' value
        if center_noise_displacement < 0.0:
            center_noise_displacement = 0.0

        self.circle_chain_size = circle_chain_size
        self.omega_min = omega_min
        self.omega_max = omega_max
        self.circle_r_min = circle_r_min
        self.circle_r_max = circle_r_max
        self.center_noise_displacement = center_noise_displacement
