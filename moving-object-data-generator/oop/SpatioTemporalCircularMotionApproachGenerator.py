import numpy as np

from oop.CircularMotionApproachInitiation import CircularMotionApproachInitiation
from oop.CircularMotionApproachParameters import CircularMotionApproachParameters
from oop.SpatioTemporalWriters import SpatioTemporalCircularMotionApproachWriter


class SpatioTemporalCircularMotionApproachGenerator:
    def __init__(
            self,
            cmap: CircularMotionApproachParameters = CircularMotionApproachParameters()):

        # store parameters of the generator
        self.cmap = cmap

        # prepare all variables and vectors required to generate data at every time frame
        self.cmai = CircularMotionApproachInitiation()
        self.cmai.initiate(cmap=self.cmap)

    def generate(
            self,
            time_frames_number: int = 10,
            output_filename: str = "SpatioTemporalCircularMotionApproachGenerator_output_file.txt",
            output_filename_timestamp: bool = True):

        print("SpatioTemporalCircularMotionApproachGenerator.generate()")

        # open file to which output will be written
        stoa_writer = SpatioTemporalCircularMotionApproachWriter(output_filename=output_filename, output_filename_timestamp=output_filename_timestamp)

        # write comment to output file about chosen configuration
        stoa_writer.write_comment(cmai=self.cmai)

        # generate data for next time frames
        for time_frame in range(0, time_frames_number):
            print("time_frame %d of %d" % (time_frame + 1, time_frames_number))

            instances_coor = np.copy(self.cmai.start_orbit_center_coor)

            x = self.cmai.radius_length * np.cos(self.cmai.angular_velocity * time_frame + self.cmai.start_angle)
            y = self.cmai.radius_length * np.sin(self.cmai.angular_velocity * time_frame + self.cmai.start_angle)

            instances_coor[:, 0] += x
            instances_coor[:, 1] += y

            # generate vector of time frame ids of starting time frame
            time_frame_ids = np.full(shape=self.cmai.features_instances_sum, fill_value=time_frame, dtype=np.int32)

            # write starting data of all the features to the output file
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
        n_colloc=2,
        lambda_1=4,
        lambda_2=3,
        m_clumpy=3,
        m_overlap=2,
        ncfr=0,
        ncfn=0,
        ncf_proportional=False,
        ndf=0,
        ndfn=0,
        random_seed=0,
        linear_velocity_mean=50
    )

    stoag = SpatioTemporalCircularMotionApproachGenerator(cmap=cmap)
    stoag.generate(
        time_frames_number=500,
        output_filename="SpatioTemporalCircularMotionApproachGenerator_output_file.txt",
        output_filename_timestamp=False
    )
