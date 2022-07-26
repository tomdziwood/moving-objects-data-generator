import numpy as np

from oop.SpatialStandardPlacement import SpatialStandardPlacement
from oop.SpatioTemporalWriters import SpatioTemporalTravelApproachWriter
from oop.TravelApproachInitiation import TravelApproachInitiation
from oop.TravelApproachEnums import StepLengthMethod, StepAngleMethod
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

        # every single noise feature instance is assigned to the unique individual co-location instance global id
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

        # create boolean array which tells if the given feature instance reached its own destination point
        features_instances_destination_reached = np.zeros(shape=self.tai.features_instances_sum, dtype=bool)

        # determine travel step length settings of each feature type
        features_step_length_mean = np.random.gamma(shape=self.tap.step_length_mean, scale=1.0, size=self.tai.features_sum)
        features_step_length_max = np.array([], dtype=np.float64)
        features_step_length_std = np.array([], dtype=np.float64)
        if self.tap.step_length_method == StepLengthMethod.UNIFORM:
            features_step_length_max = features_step_length_mean * 2
        elif self.tap.step_length_method == StepLengthMethod.NORMAL:
            features_step_length_std = self.tap.step_length_std_ratio * features_step_length_mean

        # determine travel step angle settings of each feature type
        features_step_angle_range = np.random.gamma(shape=self.tap.step_angle_range, scale=1.0, size=self.tai.features_sum)
        features_step_angle_std = np.array([], dtype=np.float64)
        if self.tap.step_angle_method == StepAngleMethod.NORMAL:
            features_step_angle_std = self.tap.step_angle_std_ratio * features_step_angle_range

        # get coordinates of features instances into single array
        instances_coor = np.column_stack(tup=(ssp.x, ssp.y))

        # generate data for each time frame
        for time_frame in range(1, time_frames_number):
            print("time_frame %d of %d" % (time_frame + 1, time_frames_number))

            # calculate coordinates difference between destination and current position of each feature instance
            coor_diff = features_instances_destination_coor - instances_coor

            # calculate distance between destination and current position of each feature instance
            dist = np.sqrt(np.sum(a=coor_diff ** 2, axis=-1))

            # determine travel step length of each feature instance
            features_instances_step_length = np.array([], dtype=np.float64)
            if self.tap.step_length_method == StepLengthMethod.CONSTANT:
                features_instances_step_length = features_step_length_mean[self.tai.features_ids]
            elif self.tap.step_length_method == StepLengthMethod.UNIFORM:
                features_instances_step_length = np.random.uniform(high=features_step_length_max[self.tai.features_ids], size=self.tai.features_instances_sum)
            elif self.tap.step_length_method == StepLengthMethod.GAUSS:
                features_instances_step_length = np.random.gamma(shape=features_step_length_mean[self.tai.features_ids], scale=1.0, size=self.tai.features_instances_sum)
            elif self.tap.step_length_method == StepLengthMethod.NORMAL:
                features_instances_step_length = np.random.normal(loc=features_step_length_mean[self.tai.features_ids], scale=features_step_length_std[self.tai.features_ids], size=self.tai.features_instances_sum)

            # calculate coordinates change when each feature instance move directly to the destination point in straight line
            instances_coor_delta_direct = np.divide(features_instances_step_length, dist, out=np.zeros_like(features_instances_step_length), where=dist != 0)
            instances_coor_delta_direct = instances_coor_delta_direct[:, None] * coor_diff

            # determine travel step angle of each feature instance
            features_instances_step_angle = np.array([], dtype=np.float64)
            if self.tap.step_angle_method == StepAngleMethod.UNIFORM:
                features_instances_step_angle = np.random.uniform(low=-features_step_angle_range[self.tai.features_ids], high=features_step_angle_range[self.tai.features_ids], size=self.tai.features_instances_sum)
            elif self.tap.step_angle_method == StepAngleMethod.NORMAL:
                features_instances_step_angle = np.random.normal(loc=0.0, scale=features_step_angle_std[self.tai.features_ids], size=self.tai.features_instances_sum)

                # if the angle of feature instance has been drawn outside of feature type range, the angle is drawn again with uniform distribution within its feature type range
                angle_out_of_range_indices = np.flatnonzero(np.logical_or(
                    features_instances_step_angle < -features_step_angle_range[self.tai.features_ids],
                    features_instances_step_angle > features_step_angle_range[self.tai.features_ids]
                ))
                features_instances_step_angle[angle_out_of_range_indices] = np.random.uniform(
                    low=-features_step_angle_range[self.tai.features_ids[angle_out_of_range_indices]],
                    high=features_step_angle_range[self.tai.features_ids[angle_out_of_range_indices]],
                    size=angle_out_of_range_indices.size
                )

            # rotate each feature instance movement according to the given angle of step
            instances_coor_delta_rotated = np.empty_like(instances_coor_delta_direct)
            cos_angle = np.cos(features_instances_step_angle)
            sin_angle = np.sin(features_instances_step_angle)
            instances_coor_delta_rotated[:, 0] = cos_angle * instances_coor_delta_direct[:, 0] - sin_angle * instances_coor_delta_direct[:, 1]
            instances_coor_delta_rotated[:, 1] = sin_angle * instances_coor_delta_direct[:, 0] + cos_angle * instances_coor_delta_direct[:, 1]

            # calculate location of instances in next time_frame
            instances_coor += instances_coor_delta_rotated

            # check if the given feature instance reached its own destination point
            # todo

            # set new destination points of features instances of the given co-location if all of these features have reached their own destination points
            # todo

            # generate vector of time frame ids of current time frame
            time_frame_ids = np.full(shape=self.tai.features_instances_sum, fill_value=time_frame, dtype=np.int32)

            # write data of all the features to the output file
            stta_writer.write(
                time_frame_ids=time_frame_ids,
                features_ids=self.tai.features_ids,
                features_instances_ids=self.tai.features_instances_ids,
                x=instances_coor[:, 0],
                y=instances_coor[:, 1]
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
        random_seed=0,
        step_length_mean=10.0,
        step_length_method=StepLengthMethod.UNIFORM,
        step_length_std_ratio=0.5,
        step_angle_range=50,
        step_angle_method=StepAngleMethod.UNIFORM,
        step_angle_std_ratio=1/3
    )

    sttag = SpatioTemporalTravelApproachGenerator(tap=tap)
    sttag.generate(
        time_frames_number=10,
        output_filename="SpatioTemporalTravelApproachGenerator_output_file.txt",
        output_filename_timestamp=False
    )
