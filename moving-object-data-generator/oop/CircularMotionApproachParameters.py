from oop.BasicParameters import BasicParameters


class CircularMotionApproachParameters(BasicParameters):
    def __init__(
            self,
            linear_velocity_mean: float = 1.0,
            **kwargs):

        super().__init__(**kwargs)

        # check 'linear_velocity_mean' value
        if linear_velocity_mean < 0.0:
            linear_velocity_mean = 0.0

        self.linear_velocity_mean = linear_velocity_mean
