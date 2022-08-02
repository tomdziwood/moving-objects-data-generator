import numpy as np

from oop.BasicParameters import BasicParameters


class CircularMotionApproachParameters(BasicParameters):
    def __init__(
            self,
            circle_chain_size: int = 2,
            omega_min: float = 2 * np.pi / 200,
            omega_max: float = 2 * np.pi / 25,
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

        self.circle_chain_size = circle_chain_size
        self.omega_min = omega_min
        self.omega_max = omega_max
