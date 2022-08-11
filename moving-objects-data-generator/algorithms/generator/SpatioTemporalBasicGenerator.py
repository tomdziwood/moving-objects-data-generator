import numpy as np

from algorithms.utils.SpatialBasicPlacement import SpatialBasicPlacement
from algorithms.utils.SpatioTemporalWriters import SpatioTemporalBasicWriter
from algorithms.initiation.BasicInitiation import BasicInitiation
from algorithms.parameters.BasicParameters import BasicParameters


class SpatioTemporalBasicGenerator:
    def __init__(
            self,
            bp: BasicParameters = BasicParameters()):
        # store parameters of the generator
        self.bp = bp

        # prepare all variables and vectors required to generate data at every time frame
        self.bi = BasicInitiation()
        self.bi.initiate(bp=self.bp)

    def generate(
            self,
            time_frames_number: int = 10,
            output_filename: str = "output\\SpatioTemporalBasicGenerator_output_file.txt",
            output_filename_timestamp: bool = True):
        print("SpatioTemporalBasicGenerator.generate()")

        # open file to which output will be written
        stb_writer = SpatioTemporalBasicWriter(output_filename=output_filename, output_filename_timestamp=output_filename_timestamp)

        # write comment to output file about chosen configuration
        stb_writer.write_comment(bi=self.bi)

        # create class object, which holds all data of the objects placement
        sbp = SpatialBasicPlacement()

        # generate data for each time frame
        for time_frame in range(time_frames_number):
            print("time_frame %d of %d" % (time_frame + 1, time_frames_number))

            # perform placement of all the features
            sbp.place(self.bi)

            # generate vector of time frame ids of current time frame
            time_frame_ids = np.full(shape=self.bi.features_instances_sum, fill_value=time_frame, dtype=np.int32)

            # write data of all the features to the output file
            stb_writer.write(
                    time_frame_ids=time_frame_ids,
                    features_ids=self.bi.features_ids,
                    features_instances_ids=self.bi.features_instances_ids,
                    x=sbp.features_instances_coor[:, 0],
                    y=sbp.features_instances_coor[:, 1]
            )

        # end of file writing
        stb_writer.close()


if __name__ == "__main__":
    print("SpatioTemporalBasicGenerator main()")

    bp = BasicParameters(
        area=1000,
        cell_size=5,
        n_colloc=2,
        lambda_1=3,
        lambda_2=6,
        m_clumpy=2,
        m_overlap=2,
        ncfr=0.5,
        ncfn=0.5,
        ncf_proportional=False,
        ndf=1,
        ndfn=5,
        random_seed=0
    )

    stbg = SpatioTemporalBasicGenerator(bp=bp)
    stbg.generate(
        time_frames_number=10,
        output_filename="output\\SpatioTemporalBasicGenerator_output_file.txt",
        output_filename_timestamp=False
    )
