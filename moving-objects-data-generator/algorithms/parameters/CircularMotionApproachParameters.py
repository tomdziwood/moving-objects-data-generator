import numpy as np

from algorithms.parameters.StandardTimeFrameParameters import StandardTimeFrameParameters


class CircularMotionApproachParameters(StandardTimeFrameParameters):
    """
    The class represents set of parameters used by the `SpatioTemporalCircularMotionApproachGenerator` class of a spatio-temporal data generator.
    """

    def __init__(
            self,
            circle_chain_size: int = 2,
            omega_min: float = 2 * np.pi / 200,
            omega_max: float = 2 * np.pi / 25,
            circle_r_min: float = 20.0,
            circle_r_max: float = 200.0,
            center_noise_displacement: float = 5.0,
            **kwargs):
        """
        Construct an object which holds all the required parameters of the `SpatioTemporalCircularMotionApproachGenerator` class of a spatio-temporal data generator.

        Parameters
        ----------
        circle_chain_size : int
            The number of defined circular motions that make up the trajectory of a single feature instance. The parameter is also known as the "n_circle".

        omega_min : float
            The lower boundary of the uniform distribution of the angular velocity of the circular orbit. The angular velocity is expressed in radians per time frame.
             The parameter is also known as the "ω_min".

        omega_max : float
            The upper boundary of the uniform distribution of the angular velocity of the circular orbit. The angular velocity is expressed in radians per time frame.
             The parameter is also known as the "ω_max".

        circle_r_min : float
            The lower boundary of the uniform distribution of the radius length of the circular orbit. The parameter is also known as the "r_circle_min".

        circle_r_max : float
            The upper boundary of the uniform distribution of the radius length of the circular orbit. The parameter is also known as the "r_circle_max".

        center_noise_displacement : float
            The radius length of the area within which the initially fixed center of the circular orbit is displaced. The parameter is also known as the "r_displacement".

        kwargs
            Other parameters passed to the super constructor of the derived class `StandardTimeFrameParameters`.
        """

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
