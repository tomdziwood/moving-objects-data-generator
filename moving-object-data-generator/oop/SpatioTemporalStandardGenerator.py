import numpy as np

from oop.SpatialStandardPlacement import SpatialStandardPlacement
from oop.SpatioTemporalWriter import SpatioTemporalWriter
from oop.StandardInitiation import StandardInitiation
from oop.StandardParameters import StandardParameters


class SpatioTemporalStandardGenerator:
    def generate(
            self,
            output_file: str = "SpatialStandardGenerator_output_file.txt",
            time_frames_number: int = 10,
            sp: StandardParameters = StandardParameters()
    ):

        print("generate()")

        # open file to which output will be written
        st_writer = SpatioTemporalWriter(output_file=output_file)

        # prepare all variables and vectors required to generate data at every time frame
        si = StandardInitiation()
        si.initiate(sp=sp)

        # create class object, which holds all data of the objects placement
        ssp = SpatialStandardPlacement()

        # generate data for each time frame
        for i_time_frame in range(time_frames_number):
            print("i_time_frame=%d" % i_time_frame)

            # perform placement of all the feature
            ssp.place(si)

            # generate vector of time frame ids of current time frame
            time_frame_ids = np.full(shape=ssp.features_ids.size, fill_value=i_time_frame, dtype=np.int32)

            # write data of all the features to the output file
            st_writer.write(
                    time_frame_ids=time_frame_ids,
                    features_ids=ssp.features_ids,
                    features_instances_ids=ssp.features_instances_ids,
                    x=ssp.x,
                    y=ssp.y
            )

        # end of file writing
        st_writer.close()


if __name__ == "__main__":
    print("main()")

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

    stsg = SpatioTemporalStandardGenerator()
    stsg.generate(
        output_file="SpatioTemporalStandardGenerator_output_file.txt",
        time_frames_number=10,
        sp=sp
    )
