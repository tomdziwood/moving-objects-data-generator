import numpy as np

from oop.StandardInitiation import StandardInitiation
from oop.TravelApproachParameters import TravelApproachParameters


class TravelApproachInitiation(StandardInitiation):
    def __init__(self):
        super().__init__()

        self.travel_approach_parameters: TravelApproachParameters = TravelApproachParameters()

    def initiate(self, tap: TravelApproachParameters = TravelApproachParameters()):
        super().initiate(sp=tap)

        self.travel_approach_parameters = tap
