import numpy as np

from algorithms.initiation.CircularMotionApproachInitiation import CircularMotionApproachInitiation
from algorithms.parameters.CircularMotionApproachParameters import CircularMotionApproachParameters
from algorithms.utils.SpatioTemporalWriters import SpatioTemporalCircularMotionApproachWriter


class SpatioTemporalCircularMotionApproachGenerator:
    """
    The class of a spatio-temporal data generator. The generated spatio-temporal data is determined through specific feature instance trajectory formulas that describe
    the positional relationship of features instances over consecutive time frames. In the case of this generator, trajectories are defined using equations
    describing the position-time relationship in uniform circular motion.

    The position of a given feature instance is described by many dedicated equations of uniform circular motion with a specified radius length. The exact number of equations
    used to determine the position of a given feature instance is defined by the parameter ``circle_chain_size``. The position of the feature instance at a given time frame
    is calculated through successive displacements (starting from a fixed reference point) along directed radii according to consecutive assigned equations of circular motion.
    This approach can be described as if a certain number of distinct radii, corresponding to various circular movements, were connected into a single polygonal line,
    into a single chain. One end of the polyline is located at a fixed point, while the other end of the polyline determines the feature instance's position in space.
    The resulting movement of the feature instance can be alternatively described as if the feature instance moves around a point, which subsequently revolves
    around another point, ..., and so on, until it finally encircles the last fixed point in space.

    The equations describing the position dependence over time in uniform circular motion are determined by the angular velocity and the radius length of the circle.
    These values are randomly drawn from a uniform distribution within the range specified by the generator parameters. The radius length is drawn from the uniform
    distribution within the range from ``circle_r_min`` to ``circle_r_max``, while the angular velocity is drawn from the uniform distribution within the range
    from ``omega_min`` to ``omega_max``.

    Features instances, which are intended to co-occur within a given co-location instance, they have very closely related sequences of circular motion equations.
    For features instances that are initially located close to each other within a small spatial cell at the first time frame, a shared sequence of points
    representing the positions of consecutive circular motion centers is determined. Next, the parameter ``center_noise_displacement`` is utilized to distort
    the initially similar trajectories of features instances movement. The parameter ``center_noise_displacement`` describes the radius length of the area
    within which the initially fixed center of the every circular orbit is randomly displaced. The applied modification of the sequence of equations allows
    influencing the actual occurrence of co-locations among a given group of features instances.
    """

    def __init__(
            self,
            cmap: CircularMotionApproachParameters = CircularMotionApproachParameters()):
        """
        Create object of a spatio-temporal data generator with given set of parameters.

        Parameters
        ----------
        cmap : CircularMotionApproachParameters
            The object which represents set of parameters used by the generator. For detailed description of available parameters, see documentation
            of the `CircularMotionApproachParameters` class.
        """

        # store parameters of the generator
        self.cmap = cmap

        # prepare all variables and vectors required to generate data at every time frame
        self.cmai = CircularMotionApproachInitiation()
        self.cmai.initiate(cmap=self.cmap)

    def generate(
            self,
            time_frames_number: int = 10,
            output_filename: str = "output\\SpatioTemporalCircularMotionApproachGenerator_output_file.txt",
            output_filename_timestamp: bool = True):
        """
        Generate spatio-temporal data.

        Parameters
        ----------
        time_frames_number : int
            The number of time frames in which spatio-temporal data will be generated.

        output_filename : str
            The file name to which output will be written.

        output_filename_timestamp : bool
            When ``True``, the filename has added unique string which is created based on the current timestamp.
            It helps to automatically recognize different output of generator.
        """

        print("SpatioTemporalCircularMotionApproachGenerator.generate()")

        # open file to which output will be written
        stoa_writer = SpatioTemporalCircularMotionApproachWriter(output_filename=output_filename, output_filename_timestamp=output_filename_timestamp)

        # write comment to output file about chosen configuration
        stoa_writer.write_comment(cmai=self.cmai)

        # generate data for next time frames
        for time_frame in range(0, time_frames_number):
            print("time_frame %d of %d" % (time_frame + 1, time_frames_number))

            # start features instances coordinates at the center of first circular orbit in circles chain of the given feature instance
            instances_coor = np.copy(self.cmai.start_orbit_center_coor)

            # calculate position determined by each of circular orbit - position calculated in reference system of the given circular orbit center
            circle_delta_x = self.cmai.radius_length * np.cos(self.cmai.angular_velocity * time_frame + self.cmai.start_angle)
            circle_delta_y = self.cmai.radius_length * np.sin(self.cmai.angular_velocity * time_frame + self.cmai.start_angle)

            # calculate final coordinates by summing all circles coordinates along axis 'x' and 'y'
            instances_coor[:, 0] += np.sum(a=circle_delta_x, axis=0)
            instances_coor[:, 1] += np.sum(a=circle_delta_y, axis=0)

            # generate vector of time frame ids of starting time frame
            time_frame_ids = np.full(shape=self.cmai.features_instances_sum, fill_value=time_frame, dtype=np.int32)

            # write data of all features instances to the output file
            stoa_writer.write(
                time_frame_ids=time_frame_ids,
                features_ids=self.cmai.features_ids,
                features_instances_ids=self.cmai.features_instances_ids,
                x=instances_coor[:, 0],
                y=instances_coor[:, 1]
            )

        # end of file writing
        stoa_writer.close()


if __name__ == "__main__":
    print("SpatioTemporalCircularMotionApproachGenerator main()")

    cmap = CircularMotionApproachParameters(
        area=1000,
        cell_size=5,
        n_base=2,
        lambda_1=5,
        lambda_2=3,
        m_clumpy=3,
        m_overlap=2,
        ncfr=0,
        ncfn=0,
        ncf_proportional=False,
        ndf=1,
        ndfn=5,
        random_seed=0,
        spatial_prevalent_ratio=1.0,
        spatial_prevalence_threshold=1.0,
        circle_chain_size=5,
        omega_min=2 * np.pi / 200,
        omega_max=2 * np.pi / 50,
        circle_r_min=20.0,
        circle_r_max=200.0,
        center_noise_displacement=5.0
    )

    stcmag = SpatioTemporalCircularMotionApproachGenerator(cmap=cmap)
    stcmag.generate(
        time_frames_number=500,
        output_filename="output\\SpatioTemporalCircularMotionApproachGenerator_output_file.txt",
        output_filename_timestamp=False
    )
