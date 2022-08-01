import math

import numpy as np

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
            output_filename: str = "SpatioTemporalStandardGenerator_output_file.txt",
            output_filename_timestamp: bool = True):

        print("SpatioTemporalStandardGenerator.generate()")

        # open file to which output will be written
        sts_writer = SpatioTemporalStandardWriter(output_filename=output_filename, output_filename_timestamp=output_filename_timestamp)

        # write comment to output file about chosen configuration
        sts_writer.write_comment(si=self.si)

        # calculate the minimal number of time frames which makes a co-location becomes persistent
        time_frames_threshold = math.ceil(self.sp.time_prevalence_threshold * time_frames_number)

        # determine the number of time frames when the given co-location pattern is spatial prevalent
        collocations_time_frames_numbers_of_spatial_prevalence = np.zeros(shape=self.si.collocations_sum, dtype=np.int32)
        collocations_time_frames_numbers_of_spatial_prevalence[self.si.transient_collocations_ids] = np.random.randint(time_frames_threshold, size=self.si.transient_collocations_sum)
        collocations_time_frames_numbers_of_spatial_prevalence[self.si.persistent_collocations_ids] = np.random.randint(low=time_frames_threshold, high=time_frames_number + 1, size=self.si.persistent_collocations_sum)
        print("collocations_time_frames_numbers_of_spatial_prevalence=%s" % str(collocations_time_frames_numbers_of_spatial_prevalence))

        # copy the defined numbers of time frames of spatial prevalence into vector, where values will be modified
        # values will be decreased and used to calculate probability that the given co-location become spatial prevalent in the current time frame
        collocations_remaining_time_frames_numbers_of_spatial_prevalence = np.copy(collocations_time_frames_numbers_of_spatial_prevalence)

        # determine the minimal number of the given co-location instances occurrence, which makes the co-location becomes spatial prevalent
        collocations_instances_number_spatial_prevalence_threshold = np.ceil(self.sp.spatial_prevalence_threshold * self.si.collocation_instances_counts).astype(np.int32)
        print("collocations_instances_number_spatial_prevalence_threshold=%s" % str(collocations_instances_number_spatial_prevalence_threshold))

        # generate data for each time frame
        for time_frame in range(time_frames_number):
            print("time_frame %d of %d" % (time_frame + 1, time_frames_number))

            # decide which of the co-locations patterns are spatial prevalent in the current time frame
            random_value = np.random.randint(low=1, high=time_frames_number - time_frame + 1, size=self.si.collocations_sum)
            collocations_spatial_prevalence_flags = random_value <= collocations_remaining_time_frames_numbers_of_spatial_prevalence

            # determine the number of the co-locations instances which actually creates co-location in the current time frame
            collocations_spatial_prevalent_instances_number = np.zeros(shape=self.si.collocations_sum, dtype=np.int32)
            collocations_spatial_prevalent_instances_number[np.logical_not(collocations_spatial_prevalence_flags)] = np.random.randint(
                low=0,
                high=collocations_instances_number_spatial_prevalence_threshold[np.logical_not(collocations_spatial_prevalence_flags)]
            )
            collocations_spatial_prevalent_instances_number[collocations_spatial_prevalence_flags] = np.random.randint(
                low=collocations_instances_number_spatial_prevalence_threshold[collocations_spatial_prevalence_flags],
                high=self.si.collocation_instances_counts[collocations_spatial_prevalence_flags] + 1
            )

            # ---begin--- create boolean vector which tells if the given co-locations instance occurs in the current time frame
            collocation_instances_counts_cumsum = np.cumsum(self.si.collocation_instances_counts)

            shuffled_values = np.repeat(
                a=self.si.collocation_instances_counts - collocation_instances_counts_cumsum,
                repeats=self.si.collocation_instances_counts
            ) + np.arange(1, self.si.collocations_instances_sum + 1)

            ind_begin = np.concatenate(([0], collocation_instances_counts_cumsum[: -1]))

            [np.random.shuffle(shuffled_values[ind_begin[i]: collocation_instances_counts_cumsum[i]]) for i in range(self.si.collocations_sum)]

            collocations_instances_spatial_prevalent_flags = shuffled_values <= np.repeat(a=collocations_spatial_prevalent_instances_number, repeats=self.si.collocation_instances_counts)
            # ----end---- create boolean vector which tells if the given co-locations instance occurs in the current time frame

            # expand co-locations' instances' flags into features' instances' flags
            features_instances_spatial_prevalent_flags = np.repeat(
                a=collocations_instances_spatial_prevalent_flags,
                repeats=self.si.collocations_instances_global_ids_repeats[:self.si.collocations_instances_sum]
            )

            # initialize features' instances' coordinates as if there were no co-locations' instances occurrences at all
            features_instances_coor = np.random.uniform(high=self.si.area_in_cell_dim * self.sp.cell_size, size=(self.si.features_instances_sum, 2))

            # initialize features' instances' coordinates as if there occurred every co-locations' instance
            collocations_instances_coor_all_collocations_instances_occured = np.random.randint(low=self.si.area_in_cell_dim, size=(self.si.collocations_instances_global_sum, 2))
            collocations_instances_coor_all_collocations_instances_occured *= self.sp.cell_size
            collocations_instances_coor_all_collocations_instances_occured = collocations_instances_coor_all_collocations_instances_occured.astype(dtype=np.float64)
            features_instances_coor_all_collocations_instances_occured = collocations_instances_coor_all_collocations_instances_occured[self.si.collocations_instances_global_ids]
            features_instances_coor_all_collocations_instances_occured += np.random.uniform(high=self.sp.cell_size, size=features_instances_coor.shape)

            # mix features' instances' coordinates according to the 'features_instances_spatial_prevalent_flags'
            features_instances_coor[:self.si.collocation_features_instances_sum][features_instances_spatial_prevalent_flags] =\
                features_instances_coor_all_collocations_instances_occured[:self.si.collocation_features_instances_sum][features_instances_spatial_prevalent_flags]

            # generate vector of time frame ids of the current time frame
            time_frame_ids = np.full(shape=self.si.features_instances_sum, fill_value=time_frame, dtype=np.int32)

            # write data of all the features to the output file
            sts_writer.write(
                    time_frame_ids=time_frame_ids,
                    features_ids=self.si.features_ids,
                    features_instances_ids=self.si.features_instances_ids,
                    x=features_instances_coor[:, 0],
                    y=features_instances_coor[:, 1]
            )

            # actualize the remaining number of the time frames when the given co-location pattern is spatial prevalent
            collocations_remaining_time_frames_numbers_of_spatial_prevalence -= collocations_spatial_prevalence_flags

        # end of file writing
        sts_writer.close()


if __name__ == "__main__":
    print("SpatioTemporalStandardGenerator main()")

    sp = StandardParameters(
        area=1000,
        cell_size=5,
        n_colloc=2,
        lambda_1=3,
        lambda_2=2,
        m_clumpy=1,
        m_overlap=1,
        ncfr=0,
        ncfn=0,
        ncf_proportional=False,
        ndf=0,
        ndfn=0,
        random_seed=0,
        persistent_ratio=0.5,
        spatial_prevalence_threshold=0.5,
        time_prevalence_threshold=0.5
    )

    stsg = SpatioTemporalStandardGenerator(sp=sp)
    stsg.generate(
        time_frames_number=10,
        output_filename="SpatioTemporalStandardGenerator_output_file.txt",
        output_filename_timestamp=False
    )
