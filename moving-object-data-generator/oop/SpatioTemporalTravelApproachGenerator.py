import numpy as np

from oop.SpatialStandardPlacement import SpatialStandardPlacement
from oop.SpatioTemporalWriters import SpatioTemporalTravelApproachWriter
from oop.TravelApproachInitiation import TravelApproachInitiation
from oop.TravelApproachParameters import TravelApproachParameters


class SpatioTemporalTravelApproachGenerator:
    def __init__(
            self,
            tap: TravelApproachParameters = TravelApproachParameters()):
        # store parameters of the generator
        self.tap = tap

        # prepare all variables and vectors required to generate data at every time frame
        self.tai = TravelApproachInitiation()
        self.tai.initiate(tap=self.tap)

    def generate(
            self,
            time_frames_number: int = 10,
            output_filename: str = "SpatioTemporalTravelApproachGenerator_output_file.txt",
            output_filename_timestamp: bool = True):
        print("SpatioTemporalTravelApproachGenerator.generate()")

        # open file to which output will be written
        stta_writer = SpatioTemporalTravelApproachWriter(output_filename=output_filename, output_filename_timestamp=output_filename_timestamp)

        # write comment to output file about chosen configuration
        stta_writer.write_comment(tai=self.tai)

        # create class object, which holds all data of the objects placement
        ssp = SpatialStandardPlacement()

        # place all objects at starting position
        ssp.place(si=self.tai)

        # generate vector of time frame ids of starting time frame
        time_frame_ids = np.full(shape=self.tai.features_instances_sum, fill_value=0, dtype=np.int32)

        # write starting data of all the features to the output file
        stta_writer.write(
            time_frame_ids=time_frame_ids,
            features_ids=self.tai.features_ids,
            features_instances_ids=self.tai.features_instances_ids,
            x=ssp.x,
            y=ssp.y
        )

        # prepare array of co-locations instances global ids to which features belong
        collocations_instances_global_ids = np.array([], dtype=np.int32)
        last_collocation_instance_global_id = 0
        for i_colloc in range(self.tap.n_colloc * self.tap.m_overlap):
            i_colloc_collocations_instances_global_ids = np.repeat(
                a=np.arange(last_collocation_instance_global_id, last_collocation_instance_global_id + self.tai.collocation_instances_counts[i_colloc]),
                repeats=self.tai.collocation_lengths[i_colloc]
            )
            collocations_instances_global_ids = np.concatenate((collocations_instances_global_ids, i_colloc_collocations_instances_global_ids))
            last_collocation_instance_global_id += self.tai.collocation_instances_counts[i_colloc]

        # every single noise feature is assigned to the unique individual co-location instance global id
        collocations_instances_global_ids = np.concatenate((
            collocations_instances_global_ids,
            np.arange(last_collocation_instance_global_id, last_collocation_instance_global_id + self.tai.collocation_noise_features_instances_sum + self.tap.ndfn)
        ))

        # sum of all specified collocation global instances
        collocation_instances_global_sum = last_collocation_instance_global_id + self.tai.collocation_noise_features_instances_sum + self.tap.ndfn

        # set destination point of every feature instance
        collocation_instances_destination_coor = np.random.randint(low=self.tai.area_in_cell_dim, size=(collocation_instances_global_sum, 2))
        collocation_instances_destination_coor *= self.tap.cell_size
        collocation_instances_destination_coor = collocation_instances_destination_coor.astype(dtype=np.float64)
        features_instances_destination_coor = collocation_instances_destination_coor[collocations_instances_global_ids]
        features_instances_destination_coor += np.random.uniform(high=self.tap.cell_size, size=features_instances_destination_coor.shape)

        # create boolean array which tells if given feature reached its own destination point
        features_instances_destination_reached = np.zeros(shape=self.tai.features_sum, dtype=bool)

        # get coordinates of features into single array
        instances_coor = np.column_stack(tup=(ssp.x, ssp.y))

        # generate data for each time frame
        for time_frame in range(1, time_frames_number):
            print("time_frame %d of %d" % (time_frame + 1, time_frames_number))

            # calculate coordinates difference between destination and current position of each feature
            coor_diff = features_instances_destination_coor - instances_coor

            # calculate distance between destination and current position of each feature
            dist = np.sqrt(np.sum(a=coor_diff ** 2, axis=-1))


            # generate vector of time frame ids of current time frame
            time_frame_ids = np.full(shape=self.si.features_instances_sum, fill_value=time_frame, dtype=np.int32)

            # write data of all the features to the output file
            stta_writer.write(
                    time_frame_ids=time_frame_ids,
                    features_ids=self.si.features_ids,
                    features_instances_ids=self.si.features_instances_ids,
                    x=ssp.x,
                    y=ssp.y
            )

        # end of file writing
        stta_writer.close()


if __name__ == "__main__":
    print("SpatioTemporalTravelApproachGenerator main()")

    tap = TravelApproachParameters(
        area=1000,
        cell_size=5,
        n_colloc=5,
        lambda_1=5,
        lambda_2=10,
        m_clumpy=1,
        m_overlap=1,
        ncfr=0.5,
        ncfn=0.5,
        ncf_proportional=False,
        ndf=5,
        ndfn=50,
        random_seed=0
    )

    sttag = SpatioTemporalTravelApproachGenerator(tap=tap)
    sttag.generate(
        time_frames_number=10,
        output_filename="SpatioTemporalTravelApproachGenerator_output_file.txt",
        output_filename_timestamp=False
    )
