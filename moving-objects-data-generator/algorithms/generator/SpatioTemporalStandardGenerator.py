import numpy as np

from algorithms.utils.SpatialStandardPlacement import SpatialStandardPlacement
from algorithms.utils.SpatioTemporalWriters import SpatioTemporalStandardWriter
from algorithms.initiation.StandardInitiation import StandardInitiation
from algorithms.parameters.StandardParameters import StandardParameters


class SpatioTemporalStandardGenerator:
    """
    The class of a spatio-temporal data generator. The generator replicates previously encountered concepts of synthetic data generation. Data generation generally involves
    determining new random locations for individual objects in successive time frames.

    In this generator, a fixed set of maximal co-locations is determined, consisting of specific types of objects. Instances of these objects, along with additional instances
    that contribute to collocation noise and additional noise, have their random locations defined within a bounded space for each successive time frame.
    The defined space is divided into a grid of small square areas. Object instances intended to co-occur within a given co-location instance are then jointly placed
    in a randomly selected small cell of the space.

    Through the appropriate parameter values, data is generated to ensure the occurrence of the desired number of maximal co-locations. The exact number of persistent
    and transient co-locations is determined by the ''persistent_ratio'' parameter. For each specific type of co-location, a specific number of time frames is established
    in which the co-location will or will not be persistent, based on the ''time_prevalence_threshold'' parameter. At a given time frame, the number of occurring
    co-location instances is determined, taking into account the ''spatial_prevalence_threshold'' parameter.
    """

    def __init__(
            self,
            sp: StandardParameters = StandardParameters()):
        """
        Create object of a spatio-temporal data generator with given set of parameters.

        Parameters
        ----------
        sp : StandardParameters
            The object which represents set of parameters used by the generator. For detailed description of available parameters, see documentation
            of the `StandardParameters` class.
        """

        # store parameters of the generator
        self.sp = sp

        # prepare all variables and vectors required to generate data at every time frame
        self.si = StandardInitiation()
        self.si.initiate(sp=self.sp)

    def generate(
            self,
            time_frames_number: int = 10,
            output_filename: str = "output\\SpatioTemporalStandardGenerator_output_file.txt",
            output_filename_timestamp: bool = True):
        """
        Generate spatio-temporal data.

        Parameters
        ----------
        time_frames_number : int
            The number of time frames in which spatio-temporal data will be generated.

        output_filename : str
            The filename to which output will be written.

        output_filename_timestamp : bool
            When ``True``, the filename has added unique string which is created based on the current timestamp.
            It helps to automatically recognize different output of generator.
        """

        print("SpatioTemporalStandardGenerator.generate()")

        # open file to which output will be written
        sts_writer = SpatioTemporalStandardWriter(output_filename=output_filename, output_filename_timestamp=output_filename_timestamp)

        # write comment to output file about chosen configuration
        sts_writer.write_comment(si=self.si)

        # create class object, which holds all data of the objects placement
        ssp = SpatialStandardPlacement(bi=self.si, collocations_instances_number_spatial_prevalence_threshold=self.si.collocations_instances_number_spatial_prevalence_threshold)

        # calculate the minimal number of time frames, which makes a co-location becomes time prevalent
        time_frames_threshold = np.ceil(self.sp.time_prevalence_threshold * time_frames_number).astype(np.int32)

        # determine the number of time frames when the given co-location pattern is spatial prevalent
        collocations_time_frames_numbers_of_spatial_prevalence = np.zeros(shape=self.si.collocations_sum, dtype=np.int32)
        collocations_time_frames_numbers_of_spatial_prevalence[self.si.transient_collocations_ids] = np.random.randint(time_frames_threshold, size=self.si.transient_collocations_sum)
        collocations_time_frames_numbers_of_spatial_prevalence[self.si.persistent_collocations_ids] = np.random.randint(low=time_frames_threshold, high=time_frames_number + 1, size=self.si.persistent_collocations_sum)
        print("collocations_time_frames_numbers_of_spatial_prevalence=%s" % str(collocations_time_frames_numbers_of_spatial_prevalence))

        # copy the defined numbers of time frames of spatial prevalence into vector, where values will be modified
        # values will be decreased and used to calculate probability that the given co-location becomes spatial prevalent at the current time frame
        collocations_remaining_time_frames_numbers_of_spatial_prevalence = np.copy(collocations_time_frames_numbers_of_spatial_prevalence)

        # generate data for each time frame
        for time_frame in range(time_frames_number):
            print("time_frame %d of %d" % (time_frame + 1, time_frames_number))

            # decide which of the co-location patterns are spatial prevalent at the current time frame
            random_value = np.random.randint(low=1, high=time_frames_number - time_frame + 1, size=self.si.collocations_sum)
            collocations_spatial_prevalence_flags = random_value <= collocations_remaining_time_frames_numbers_of_spatial_prevalence

            # perform placement of all the features instances
            ssp.place(collocations_spatial_prevalence_flags=collocations_spatial_prevalence_flags)

            # generate vector of time frame ids of the current time frame
            time_frame_ids = np.full(shape=self.si.features_instances_sum, fill_value=time_frame, dtype=np.int32)

            # write data of all the features instances to the output file
            sts_writer.write(
                    time_frame_ids=time_frame_ids,
                    features_ids=self.si.features_ids,
                    features_instances_ids=self.si.features_instances_ids,
                    x=ssp.features_instances_coor[:, 0],
                    y=ssp.features_instances_coor[:, 1]
            )

            # actualize the remaining number of the time frames when the given co-location pattern is spatial prevalent
            collocations_remaining_time_frames_numbers_of_spatial_prevalence -= collocations_spatial_prevalence_flags

        # end of file writing
        sts_writer.close()


if __name__ == "__main__":
    print("SpatioTemporalStandardGenerator main()")

    sp = StandardParameters(
        area=50,
        cell_size=5,
        n_base=10,
        lambda_1=5,
        lambda_2=2,
        m_clumpy=1,
        m_overlap=1,
        ncfr=0,
        ncfn=0,
        ncf_proportional=False,
        ndf=1,
        ndfn=5,
        random_seed=0,
        persistent_ratio=1.0,
        spatial_prevalence_threshold=1.0,
        time_prevalence_threshold=1.0
    )

    stsg = SpatioTemporalStandardGenerator(sp=sp)
    stsg.generate(
        time_frames_number=10,
        output_filename="output\\SpatioTemporalStandardGenerator_output_file.txt",
        output_filename_timestamp=False
    )
