import numpy as np

from oop.SpatialBasicPlacement import SpatialBasicPlacement
from oop.SpatioTemporalWriters import SpatioTemporalTravelApproachWriter
from oop.TravelApproachInitiation import TravelApproachInitiation
from oop.TravelApproachEnums import StepLengthMethod, StepAngleMethod
from oop.TravelApproachParameters import TravelApproachParameters


class SpatioTemporalTravelApproachGenerator:
    """
    The class of a spatio-temporal data generator. The main idea of the generator is that features' instances are changing their positions
    by making steps into direction of the individually defined destination points. When a current destination point is achieved, then a new destination point is defined.
    There are allowed different values of step's length and different values of step's angle to the direction of destination point of the given feature's instance.

    Feature's instances, which tend to occur together as a co-location instance, they have closely located their destination points - similar like when
    these feature's instance positions are defined at the starting position according to `SpatialBasicPlacement` procedure. In case of the given co-location instance,
    only when all the features' instances reach the current destination points, the new destination points are defined. It is possible to get new destination
    thanks to the elapsed time frames according to the `waiting_time_frames` parameter value.
    """

    def __init__(
            self,
            tap: TravelApproachParameters = TravelApproachParameters()):
        """
        Create object of a spatio-temporal data generator with given set of parameters.

        Parameters
        ----------
        tap: TravelApproachParameters
            The object which represents set of parameters used by the generator. For detailed description of available parameters, see documentation
            of the `TravelApproachParameters` class.
        """

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

        print("SpatioTemporalTravelApproachGenerator.generate()")

        # open file to which output will be written
        stta_writer = SpatioTemporalTravelApproachWriter(output_filename=output_filename, output_filename_timestamp=output_filename_timestamp)

        # write comment to output file about chosen configuration
        stta_writer.write_comment(tai=self.tai)

        # create class object, which holds all data of the objects placement
        sbp = SpatialBasicPlacement()

        # place all objects at starting position
        sbp.place(bi=self.tai)

        # generate vector of time frame ids of starting time frame
        time_frame_ids = np.full(shape=self.tai.features_instances_sum, fill_value=0, dtype=np.int32)

        # write starting data of all the features to the output file
        stta_writer.write(
            time_frame_ids=time_frame_ids,
            features_ids=self.tai.features_ids,
            features_instances_ids=self.tai.features_instances_ids,
            x=sbp.features_instances_coor[:, 0],
            y=sbp.features_instances_coor[:, 1]
        )

        # copy coordinates of features instances
        instances_coor = np.copy(sbp.features_instances_coor)

        # generate data for each time frame
        for time_frame in range(1, time_frames_number):
            print("time_frame %d of %d" % (time_frame + 1, time_frames_number))

            # calculate coordinates difference between destination and current position of each feature instance
            coor_diff = self.tai.features_instances_destination_coor - instances_coor

            # calculate distance between destination and current position of each feature instance
            dist = np.sqrt(np.sum(a=coor_diff ** 2, axis=-1))

            # determine travel step length of each feature instance
            features_instances_step_length = np.array([], dtype=np.float64)
            if self.tap.step_length_method == StepLengthMethod.UNIFORM:
                features_instances_step_length = np.random.uniform(low=self.tai.features_step_length_uniform_min[self.tai.features_ids], high=self.tai.features_step_length_uniform_max[self.tai.features_ids], size=self.tai.features_instances_sum)
            elif self.tap.step_length_method == StepLengthMethod.GAMMA:
                features_instances_step_length = np.random.gamma(shape=self.tai.features_step_length_mean[self.tai.features_ids], scale=1.0, size=self.tai.features_instances_sum)
            elif self.tap.step_length_method == StepLengthMethod.NORMAL:
                features_instances_step_length = np.random.normal(loc=self.tai.features_step_length_mean[self.tai.features_ids], scale=self.tai.features_step_length_normal_std[self.tai.features_ids], size=self.tai.features_instances_sum)

            # calculate coordinates change when each feature instance move directly to the destination point in straight line
            instances_coor_delta_direct = np.divide(features_instances_step_length, dist, out=np.zeros_like(features_instances_step_length), where=dist != 0)
            instances_coor_delta_direct = instances_coor_delta_direct[:, None] * coor_diff

            # determine travel step angle of each feature instance
            features_instances_step_angle = np.array([], dtype=np.float64)
            if self.tap.step_angle_method == StepAngleMethod.UNIFORM:
                features_instances_step_angle = np.random.uniform(low=-self.tai.features_step_angle_range[self.tai.features_ids], high=self.tai.features_step_angle_range[self.tai.features_ids], size=self.tai.features_instances_sum)
            elif self.tap.step_angle_method == StepAngleMethod.NORMAL:
                features_instances_step_angle = np.random.normal(loc=0.0, scale=self.tai.features_step_angle_normal_std[self.tai.features_ids], size=self.tai.features_instances_sum)

                # if the angle of feature instance has been drawn outside of feature type range, the angle is drawn again with uniform distribution within its feature type range
                angle_out_of_range_indices = np.flatnonzero(np.logical_or(
                    features_instances_step_angle < -self.tai.features_step_angle_range[self.tai.features_ids],
                    features_instances_step_angle > self.tai.features_step_angle_range[self.tai.features_ids]
                ))
                features_instances_step_angle[angle_out_of_range_indices] = np.random.uniform(
                    low=-self.tai.features_step_angle_range[self.tai.features_ids[angle_out_of_range_indices]],
                    high=self.tai.features_step_angle_range[self.tai.features_ids[angle_out_of_range_indices]],
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

            # check if the given feature instance reached its own destination point - distance to a destination lower than a half of mean step length
            dist_squared = np.sum(a=(self.tai.features_instances_destination_coor - instances_coor) ** 2, axis=-1)
            self.tai.features_instances_destination_reached = np.logical_or(
                self.tai.features_instances_destination_reached,
                dist_squared <= ((self.tai.features_step_length_mean[self.tai.features_ids] / 2) ** 2)
            )

            # create boolean array which tells if the given co-location instance has at least one feature instance with reached destination
            collocations_instances_global_ids_reached_at_least_one_feature_ids = self.tai.collocations_instances_global_ids[self.tai.features_instances_destination_reached]
            collocations_instances_global_ids_reached_at_least_one_feature_ids = np.unique(ar=collocations_instances_global_ids_reached_at_least_one_feature_ids)
            collocations_instances_global_ids_reached_at_least_one_feature = np.zeros(shape=self.tai.collocations_instances_global_sum, dtype=bool)
            collocations_instances_global_ids_reached_at_least_one_feature[collocations_instances_global_ids_reached_at_least_one_feature_ids] = True

            # turn on countdown of the given co-location instance whose first of its features instances has reached the destination in current time frame
            self.tai.collocations_instances_waiting_countdown[np.logical_and(
                collocations_instances_global_ids_reached_at_least_one_feature,
                self.tai.collocations_instances_waiting_countdown == -1
            )] = self.tap.waiting_time_frames

            # create boolean array which tells if the countdown of waiting in given co-location instance is finished
            collocations_instances_waiting_countdown_finished = self.tai.collocations_instances_waiting_countdown == 0

            # check which of co-locations instances has all of its feature instances with reached destination point
            collocations_instances_global_ids_not_reached_ids = self.tai.collocations_instances_global_ids[np.logical_not(self.tai.features_instances_destination_reached)]
            collocations_instances_global_ids_not_reached_ids = np.unique(ar=collocations_instances_global_ids_not_reached_ids)
            collocations_instances_global_ids_reached = np.ones(shape=self.tai.collocations_instances_global_sum, dtype=bool)
            collocations_instances_global_ids_reached[collocations_instances_global_ids_not_reached_ids] = False

            # indicate features instances whose destinations need to be changed based on finished countdowns and reached previous destinations
            collocations_instances_new_destination_needed = np.logical_or(collocations_instances_waiting_countdown_finished, collocations_instances_global_ids_reached)
            features_instances_new_destination_needed = np.repeat(a=collocations_instances_new_destination_needed, repeats=self.tai.collocations_instances_global_ids_repeats)

            # set new destination points of features instances of the given co-location if new destination points are needed
            self.tai.collocations_instances_destination_coor[collocations_instances_new_destination_needed] = np.random.randint(low=self.tai.area_in_cell_dim, size=(collocations_instances_new_destination_needed.sum(), 2))
            self.tai.collocations_instances_destination_coor[collocations_instances_new_destination_needed] *= self.tap.cell_size
            self.tai.features_instances_destination_coor[features_instances_new_destination_needed] = self.tai.collocations_instances_destination_coor[self.tai.collocations_instances_global_ids[features_instances_new_destination_needed]]
            self.tai.features_instances_destination_coor[features_instances_new_destination_needed] += np.random.uniform(high=self.tap.cell_size, size=(features_instances_new_destination_needed.sum(), 2))

            # mark new destination points as not reached
            self.tai.features_instances_destination_reached[features_instances_new_destination_needed] = False

            # turn off countdowns of co-locations instances with new destination points
            self.tai.collocations_instances_waiting_countdown[collocations_instances_new_destination_needed] = -1

            # decrement countdown of active co-locations instances
            self.tai.collocations_instances_waiting_countdown[self.tai.collocations_instances_waiting_countdown != -1] -= 1

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
        n_colloc=3,
        lambda_1=5,
        lambda_2=20,
        m_clumpy=1,
        m_overlap=1,
        ncfr=0,
        ncfn=0,
        ncf_proportional=False,
        ndf=3,
        ndfn=30,
        random_seed=0,
        step_length_mean=10.0,
        step_length_method=StepLengthMethod.UNIFORM,
        step_length_uniform_low_to_mean_ratio=1,
        step_length_normal_std_ratio=1 / 3,
        step_angle_range_mean=np.pi / 4,
        step_angle_range_limit=np.pi / 2,
        step_angle_method=StepAngleMethod.UNIFORM,
        step_angle_normal_std_ratio=1 / 3,
        waiting_time_frames=20
    )

    sttag = SpatioTemporalTravelApproachGenerator(tap=tap)
    sttag.generate(
        time_frames_number=10,
        output_filename="SpatioTemporalTravelApproachGenerator_output_file.txt",
        output_filename_timestamp=False
    )
