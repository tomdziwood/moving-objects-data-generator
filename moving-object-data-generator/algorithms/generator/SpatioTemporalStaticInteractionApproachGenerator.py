import numpy as np

from algorithms.enums.StaticInteractionApproachEnums import IdenticalFeaturesInteractionMode, DifferentFeaturesInteractionMode, MassMode, VelocityMode
from algorithms.utils.SpatioTemporalWriters import SpatioTemporalStaticInteractionApproachWriter
from algorithms.initiation.StaticInteractionApproachInitiation import StaticInteractionApproachInitiation
from algorithms.parameters.StaticInteractionApproachParameters import StaticInteractionApproachParameters


def view_statistics_of_absolute_values(array, array_name):
    array_abs = np.sqrt(np.sum(a=array ** 2, axis=-1))
    print("%s:\n\tmin:\t%.12f\n\tavg:\t%.12f\n\tmax:\t%.12f\n" % (array_name, array_abs.min(), array_abs.mean(), array_abs.max()))


class SpatioTemporalStaticInteractionApproachGenerator:
    def __init__(
            self,
            siap: StaticInteractionApproachParameters = StaticInteractionApproachParameters()):
        # store parameters of the generator
        self.siap = siap

        # prepare all variables and vectors required to generate data at every time frame
        self.siai = StaticInteractionApproachInitiation()
        self.siai.initiate(siap=self.siap)

    def generate(
            self,
            time_frames_number: int = 10,
            output_filename: str = "output\\SpatioTemporalStaticInteractionApproachGenerator_output_file.txt",
            output_filename_timestamp: bool = True):
        print("SpatioTemporalStaticInteractionApproachGenerator.generate()")

        # open file to which output will be written
        stsia_writer = SpatioTemporalStaticInteractionApproachWriter(output_filename=output_filename, output_filename_timestamp=output_filename_timestamp)

        # generate vector of time frame ids of starting time frame
        time_frame_ids = np.full(shape=self.siai.features_instances_sum, fill_value=0, dtype=np.int32)

        # write comment to output file about chosen configuration
        stsia_writer.write_comment(siai=self.siai)

        # write starting data of all the features to the output file
        stsia_writer.write(
            time_frame_ids=time_frame_ids,
            features_ids=self.siai.features_ids,
            features_instances_ids=self.siai.features_instances_ids,
            x=self.siai.spatial_basic_placement.features_instances_coor[:, 0],
            y=self.siai.spatial_basic_placement.features_instances_coor[:, 1]
        )

        # get arrays where new coordinates and velocities of instances will be calculated
        instances_coor = np.copy(self.siai.instances_coor)
        velocity = np.copy(self.siai.velocity)

        # generate data for next time frames
        for time_frame in range(1, time_frames_number):
            print("time_frame %d of %d" % (time_frame + 1, time_frames_number))

            # calculate positions in next time frame by making 'approx_steps_number' steps
            for approx_step in range(self.siap.approx_steps_number):
                # print("\tapprox_step %d of %d" % (approx_step + 1, self.siap.approx_steps_number))

                # calculate coordinates difference of each pair of instances
                coor_diff = instances_coor[None, :, :] - instances_coor[:, None, :]

                # translate coordinates difference into distance unit
                coor_diff /= self.siap.distance_unit

                # calculate squared distance between each pair of instances
                dist_squared = np.sum(a=coor_diff ** 2, axis=-1)

                # calculate distance between each pair of instances
                dist = np.sqrt(dist_squared)

                # calculate absolute value of attraction force between each pair of instances
                force_abs = np.divide(self.siai.force_multiplier_constant, dist_squared, out=np.zeros_like(dist_squared), where=dist_squared != 0)

                # scale forces according to the rules of interaction between given features instances pair
                force_abs *= self.siai.features_instances_interaction

                # calculate components of force between each pair of instances
                force_div = np.divide(force_abs, dist, out=np.zeros_like(force_abs), where=dist != 0)
                force = force_div[:, :, None] * coor_diff
                # force = (force_abs / dist)[:, :, None] * coor_diff

                # calculate resultant force for each instance
                force_resultant = np.sum(a=force, axis=1)

                # ---begin--- limit resultant forces which are greater than given 'force_limit' parameter value
                # calculate absolute value of resultant force for each instance
                force_resultant_abs = np.sqrt(np.sum(a=force_resultant ** 2, axis=-1))

                # flag resultant forces which are greater than given 'force_limit' parameter value
                force_resultant_abs_exceeded = force_resultant_abs > self.siap.force_limit

                # rescale components of resultant forces which exceeded limit
                force_resultant[force_resultant_abs_exceeded] *= self.siap.force_limit / force_resultant_abs[force_resultant_abs_exceeded][:, None]
                # ----end---- limit resultant forces which are greater than given 'force_limit' parameter value

                # ---begin--- force escaping objects to stay close to mass center within range of 'faraway_limit'
                # calculate absolute value of distance from mass center
                dist_center = (self.siai.center - instances_coor) / self.siap.distance_unit
                dist_center_abs = np.sqrt(np.sum(a=dist_center ** 2, axis=-1))

                # calculate absolute value of force from mass center
                force_center_abs = np.divide(dist_center_abs - self.siai.faraway_limit, self.siai.faraway_limit, out=np.zeros_like(dist_center_abs), where=dist_center_abs > self.siai.faraway_limit)
                force_center_abs = force_center_abs ** 2 * self.siai.force_center_multiplier_constant

                # calculate components of force from mass center
                force_center = np.divide(force_center_abs, dist_center_abs, out=np.zeros_like(force_center_abs), where=dist_center_abs != 0)
                force_center = force_center[:, None] * dist_center

                # modify the applied force to the given feature instance with force from mass center
                force_resultant += force_center
                # ----end---- force escaping objects to stay close to mass center within range of 'faraway_limit'

                # calculate acceleration
                acceleration = force_resultant / self.siai.mass[:, None]

                # calculate change of velocity
                velocity_delta = acceleration * self.siai.approx_step_time_interval

                # calculate velocity of instances at the end of time interval
                velocity_end = velocity + velocity_delta

                # limit velocity according to the 'velocity_limit' parameter value
                velocity_limited = np.copy(velocity_end)
                velocity_limited[velocity_limited > self.siap.velocity_limit] = self.siap.velocity_limit
                velocity_limited[velocity_limited < -self.siap.velocity_limit] = -self.siap.velocity_limit

                # calculate time of achieving limited velocity
                time_of_velocity_limit_achieving = np.divide(self.siai.approx_step_time_interval * (velocity_limited - velocity), velocity_delta, out=np.zeros_like(velocity_delta), where=velocity_delta != 0)

                # calculate distance change of instances
                instances_coor_delta = self.siai.approx_step_time_interval * (velocity + velocity_delta / 2)
                instances_coor_delta -= (self.siai.approx_step_time_interval - time_of_velocity_limit_achieving) * (velocity_end - velocity_limited) / 2

                # translate distance change into coordinates change
                instances_coor_delta *= self.siap.distance_unit

                # remember velocity of instances at the end of time interval
                velocity = velocity_limited

                # calculate location of instances in next time_frame
                instances_coor += instances_coor_delta

            # generate vector of time frame ids of starting time frame
            time_frame_ids = np.full(shape=self.siai.features_instances_sum, fill_value=time_frame, dtype=np.int32)

            # write starting data of all the features to the output file
            stsia_writer.write(
                time_frame_ids=time_frame_ids,
                features_ids=self.siai.features_ids,
                features_instances_ids=self.siai.features_instances_ids,
                x=instances_coor[:, 0],
                y=instances_coor[:, 1]
            )

        # end of file writing
        stsia_writer.close()


if __name__ == "__main__":
    print("SpatioTemporalStaticInteractionApproachGenerator main()")

    siap = StaticInteractionApproachParameters(
        area=1000,
        cell_size=5,
        n_colloc=2,
        lambda_1=4,
        lambda_2=5,
        m_clumpy=1,
        m_overlap=1,
        ncfr=0.5,
        ncfn=0.3,
        ncf_proportional=False,
        ndf=3,
        ndfn=30,
        random_seed=0,
        time_unit=1,
        distance_unit=1.0,
        approx_steps_number=10,
        k_force=10,
        force_limit=20.0,
        velocity_limit=20.0,
        faraway_limit_ratio=np.sqrt(2) / 2,
        mass_mode=MassMode.CONSTANT,
        mass_mean=1.0,
        mass_normal_std_ratio=1 / 5,
        velocity_mode=VelocityMode.CONSTANT,
        velocity_mean=0.0,
        identical_features_interaction_mode=IdenticalFeaturesInteractionMode.ATTRACT,
        different_features_interaction_mode=DifferentFeaturesInteractionMode.ATTRACT
    )

    stsiag = SpatioTemporalStaticInteractionApproachGenerator(siap=siap)
    stsiag.generate(
        time_frames_number=500,
        output_filename="output\\SpatioTemporalStaticInteractionApproachGenerator_output_file.txt",
        output_filename_timestamp=True
    )
