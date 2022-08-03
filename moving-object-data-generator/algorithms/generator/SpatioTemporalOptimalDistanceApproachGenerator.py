import numpy as np

from algorithms.enums.OptimalDistanceApproachEnums import MassMode, VelocityMode
from algorithms.initiation.OptimalDistanceApproachInitiation import OptimalDistanceApproachInitiation
from algorithms.parameters.OptimalDistanceApproachParameters import OptimalDistanceApproachParameters
from algorithms.utils.SpatioTemporalWriters import SpatioTemporalOptimalDistanceApproachWriter


class SpatioTemporalOptimalDistanceApproachGenerator:
    def __init__(
            self,
            odap: OptimalDistanceApproachParameters = OptimalDistanceApproachParameters()):

        # store parameters of the generator
        self.odap = odap

        # prepare all variables and vectors required to generate data at every time frame
        self.odai = OptimalDistanceApproachInitiation()
        self.odai.initiate(odap=self.odap)

    def generate(
            self,
            time_frames_number: int = 10,
            output_filename: str = "output\\SpatioTemporalOptimalDistanceApproachGenerator_output_file.txt",
            output_filename_timestamp: bool = True):
        print("SpatioTemporalOptimalDistanceApproachGenerator.generate()")

        # open file to which output will be written
        stoda_writer = SpatioTemporalOptimalDistanceApproachWriter(output_filename=output_filename, output_filename_timestamp=output_filename_timestamp)

        # generate vector of time frame ids of starting time frame
        time_frame_ids = np.full(shape=self.odai.features_instances_sum, fill_value=0, dtype=np.int32)

        # write comment to output file about chosen configuration
        stoda_writer.write_comment(odai=self.odai)

        # write starting data of all the features to the output file
        stoda_writer.write(
            time_frame_ids=time_frame_ids,
            features_ids=self.odai.features_ids,
            features_instances_ids=self.odai.features_instances_ids,
            x=self.odai.spatial_basic_placement.features_instances_coor[:, 0],
            y=self.odai.spatial_basic_placement.features_instances_coor[:, 1]
        )

        # get arrays where new coordinates and velocities of instances will be calculated
        instances_coor = np.copy(self.odai.instances_coor)
        velocity = np.copy(self.odai.velocity)

        # generate data for next time frames
        for time_frame in range(1, time_frames_number):
            print("time_frame %d of %d" % (time_frame + 1, time_frames_number))

            # calculate positions in next time frame by making 'approx_steps_number' steps
            for approx_step in range(self.odap.approx_steps_number):
                # print("\tapprox_step %d of %d" % (approx_step + 1, self.odap.approx_steps_number))

                # calculate coordinates difference of each pair of instances
                coor_diff = instances_coor[None, :, :] - instances_coor[:, None, :]

                # calculate squared distance between each pair of instances
                dist_squared = np.sum(a=coor_diff ** 2, axis=-1)

                # calculate distance between each pair of instances
                dist = np.sqrt(dist_squared)

                # ---begin--- calculate resultant repulsion force for each instance
                # calculate absolute value of repulsion force between each pair of instances
                force_repulsion_abs = -np.divide(odap.k_optimal_distance ** 2 * self.odai.force_multiplier_constant, dist, out=np.zeros_like(dist), where=dist != 0)

                # calculate components of repulsion force between each pair of instances
                force_repulsion_div = np.divide(force_repulsion_abs, dist, out=np.zeros_like(force_repulsion_abs), where=dist != 0)
                force_repulsion = force_repulsion_div[:, :, None] * coor_diff
                # force_repulsion = (force_repulsion_abs / dist)[:, :, None] * coor_diff

                # calculate resultant repulsion force for each instance
                force_repulsion_resultant = np.sum(a=force_repulsion, axis=1)
                # ----end---- calculate resultant repulsion force for each instance

                # ---begin--- calculate resultant attraction force for each instance
                # calculate absolute value of attraction force between each pair of instances
                force_attraction_abs = dist_squared * self.odai.force_multiplier_constant / odap.k_optimal_distance

                # reset force attraction where instances are not in the same co-location instance
                force_attraction_abs *= self.odai.common_collocation_instance_flag

                # calculate components of attraction force between each pair of instances
                force_attraction_div = np.divide(force_attraction_abs, dist, out=np.zeros_like(force_attraction_abs), where=dist != 0)
                force_attraction = force_attraction_div[:, :, None] * coor_diff
                # force_attraction = (force_attraction_abs / dist)[:, :, None] * coor_diff

                # calculate resultant attraction force for each instance
                force_attraction_resultant = np.sum(a=force_attraction, axis=1)
                # ----end---- calculate resultant attraction force for each instance

                # calculate resultant force of the repulsion and attraction forces for each instance
                force_resultant = force_attraction_resultant + force_repulsion_resultant

                # ---begin--- limit resultant forces which are greater than given 'force_limit' parameter value
                # calculate absolute value of resultant force for each instance
                force_resultant_abs = np.sqrt(np.sum(a=force_resultant ** 2, axis=-1))

                # flag resultant forces which are greater than given 'force_limit' parameter value
                force_resultant_abs_exceeded = force_resultant_abs > self.odap.force_limit

                # rescale components of resultant forces which exceeded limit
                force_resultant[force_resultant_abs_exceeded] *= self.odap.force_limit / force_resultant_abs[force_resultant_abs_exceeded][:, None]
                # ----end---- limit resultant forces which are greater than given 'force_limit' parameter value

                # ---begin--- force escaping objects to stay close to mass center within range of 'faraway_limit'
                # calculate absolute value of distance from mass center
                dist_center_diff = (self.odai.center - instances_coor)
                dist_center_abs = np.sqrt(np.sum(a=dist_center_diff ** 2, axis=-1))

                # calculate absolute value of force from mass center
                force_center_abs = np.divide(dist_center_abs - self.odap.faraway_limit, self.odap.faraway_limit, out=np.zeros_like(dist_center_abs), where=dist_center_abs > self.odap.faraway_limit)
                force_center_abs = force_center_abs ** 2 * self.odap.k_force * self.odai.mass_sum * self.odai.mass

                # calculate components of force from mass center
                force_center = np.divide(force_center_abs, dist_center_abs, out=np.zeros_like(force_center_abs), where=dist_center_abs != 0)
                force_center = force_center[:, None] * dist_center_diff

                force_resultant += force_center
                # ----end---- force escaping objects to stay close to mass center within range of 'faraway_limit'

                # calculate acceleration
                acceleration = force_resultant / self.odai.mass[:, None]

                # calculate change of velocity
                velocity_delta = acceleration * self.odai.approx_step_time_interval

                # calculate distance change of instances
                instances_coor_delta = self.odai.approx_step_time_interval * (velocity + velocity_delta / 2)

                # calculate velocity of instances
                velocity += velocity_delta

                # calculate location of instances in next time_frame
                instances_coor += instances_coor_delta

            # generate vector of time frame ids of starting time frame
            time_frame_ids = np.full(shape=self.odai.features_instances_sum, fill_value=time_frame, dtype=np.int32)

            # write starting data of all the features to the output file
            stoda_writer.write(
                time_frame_ids=time_frame_ids,
                features_ids=self.odai.features_ids,
                features_instances_ids=self.odai.features_instances_ids,
                x=instances_coor[:, 0],
                y=instances_coor[:, 1]
            )

        # end of file writing
        stoda_writer.close()


if __name__ == "__main__":
    print("SpatioTemporalOptimalDistanceApproachGenerator main()")

    odap = OptimalDistanceApproachParameters(
        area=1000,
        cell_size=5,
        n_colloc=2,
        lambda_1=5,
        lambda_2=1,
        m_clumpy=1,
        m_overlap=1,
        ncfr=0,
        ncfn=0,
        ncf_proportional=False,
        ndf=1,
        ndfn=5,
        random_seed=0,
        time_unit=1,
        approx_steps_number=1,
        k_optimal_distance=10.0,
        k_force=1.0,
        force_limit=5.0,
        mass_mode=MassMode.CONSTANT,
        mass_mean=1.0,
        mass_normal_std_ratio=1 / 5,
        velocity_mode=VelocityMode.CONSTANT,
        velocity_mean=0.0,
        faraway_limit=1000
    )

    stodag = SpatioTemporalOptimalDistanceApproachGenerator(odap=odap)
    stodag.generate(
        time_frames_number=100,
        output_filename="output\\SpatioTemporalOptimalDistanceApproachGenerator_output_file.txt",
        output_filename_timestamp=False
    )
