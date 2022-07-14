import numpy as np

from oop.SpatialStandardPlacement import SpatialStandardPlacement
from oop.SpatioTemporalWriters import SpatioTemporalStandardWriter
from oop.StandardInitiation import StandardInitiation
from oop.StandardParameters import StandardParameters


class SpatioTemporalStandardGenerator:
    def __init__(
            self,
            sp: StandardParameters = StandardParameters()):
        # store parameters of the generator
        self.sp = sp

        # prepare all variables and vectors required to generate data at every time frame
        self.si = StandardInitiation()
        self.si.initiate(sp=self.sp)

    def generate(
            self,
            time_frames_number: int = 10,
            output_filename: str = "SpatialStandardGenerator_output_file.txt",
            output_filename_timestamp: bool = True):
        print("SpatioTemporalStandardGenerator.generate()")

        # open file to which output will be written
        sts_writer = SpatioTemporalStandardWriter(output_filename=output_filename, output_filename_timestamp=output_filename_timestamp)

        # write comment to output file about chosen configuration
        sts_writer.write_comment(si=self.si)

        # create class object, which holds all data of the objects placement
        ssp = SpatialStandardPlacement()

        # generate data for each time frame
        for time_frame in range(time_frames_number):
            print("time_frame %d of %d" % (time_frame + 1, time_frames_number))

            # perform placement of all the features
            ssp.place(self.si)

            # generate vector of time frame ids of current time frame
            time_frame_ids = np.full(shape=self.si.features_instances_sum, fill_value=time_frame, dtype=np.int32)

            # write data of all the features to the output file
            sts_writer.write(
                    time_frame_ids=time_frame_ids,
                    features_ids=self.si.features_ids,
                    features_instances_ids=self.si.features_instances_ids,
                    x=ssp.x,
                    y=ssp.y
            )

        # end of file writing
        sts_writer.close()


if __name__ == "__main__":
    print("SpatioTemporalStandardGenerator main()")

    sp = StandardParameters(
        area=1000,
        cell_size=5,
        n_colloc=2,
        lambda_1=3,
        lambda_2=3,
        m_clumpy=1,
        m_overlap=1,
        ncfr=0.5,
        ncfn=0.5,
        ncf_proportional=False,
        ndf=1,
        ndfn=5,
        random_seed=0
    )

    stsg = SpatioTemporalStandardGenerator(sp=sp)
    stsg.generate(
        time_frames_number=10,
        output_filename="SpatioTemporalStandardGenerator_output_file.txt",
        output_filename_timestamp=False
    )
