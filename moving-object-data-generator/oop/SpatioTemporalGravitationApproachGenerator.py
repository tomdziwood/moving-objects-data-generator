import numpy as np

from oop.GravitationApproachInitiation import GravitationApproachInitiation
from oop.GravitationApproachParameters import GravitationApproachParameters
from oop.SpatioTemporalWriters import SpatioTemporalGravitationApproachWriter


def view_statistics_of_absolute_values(array, array_name):
    array_abs = np.sqrt(np.sum(a=array ** 2, axis=-1))
    print("%s:\n\tmin:\t%.12f\n\tavg:\t%.12f\n\tmax:\t%.12f\n" % (array_name, array_abs.min(), array_abs.mean(), array_abs.max()))


class SpatioTemporalGravitationApproachGenerator:
    def __init__(
            self,
            gap: GravitationApproachParameters = GravitationApproachParameters()):
        # store parameters of the generator
        self.gap = gap

        # prepare all variables and vectors required to generate data at every time frame
        self.gai = GravitationApproachInitiation()
        self.gai.initiate(gap=self.gap)

    def generate(
            self,
            time_frames_number: int = 10,
            output_filename: str = "SpatialStandardGenerator_output_file.txt",
            output_filename_timestamp: bool = True):
        print("SpatioTemporalGravitationApproachGenerator.generate()")

        # open file to which output will be written
        stga_writer = SpatioTemporalGravitationApproachWriter(output_filename=output_filename, output_filename_timestamp=output_filename_timestamp)

        # generate vector of time frame ids of starting time frame
        time_frame_ids = np.full(shape=self.gai.features_instances_sum, fill_value=0, dtype=np.int32)

        # write comment to output file about chosen configuration
        stga_writer.write_comment(gai=self.gai)

        # write starting data of all the features to the output file
        stga_writer.write(
            time_frame_ids=time_frame_ids,
            features_ids=self.gai.features_ids,
            features_instances_ids=self.gai.features_instances_ids,
            x=self.gai.spatial_standard_placement.x,
            y=self.gai.spatial_standard_placement.y
        )

        # get arrays where new coordinates and velocities of instances will be calculated
        instances_coor = np.copy(self.gai.instances_coor)
        velocity = np.copy(self.gai.velocity)

        # generate data for next time frames
        for time_frame in range(1, time_frames_number):
            print("time_frame %d of %d" % (time_frame + 1, time_frames_number))

            # calculate positions in next time frame by making 'approx_steps_number' steps
            for approx_step in range(self.gap.approx_steps_number):
                # print("\tapprox_step %d of %d" % (approx_step + 1, self.gap.approx_steps_number))

                # calculate coordinates difference of each pair of instances
                coor_diff = instances_coor[None, :, :] - instances_coor[:, None, :]
                # print("\ncoor_diff:")
                # print(coor_diff)

                # translate coordinates difference into distance unit
                coor_diff /= self.gap.distance_unit
                # print("\ncoor_diff:")
                # print(coor_diff)

                # calculate squared distance between each pair of instances
                dist_squared = np.sum(a=coor_diff ** 2, axis=-1)
                # print("\ndist_squared:")
                # print(dist_squared)

                # calculate distance between each pair of instances
                dist = np.sqrt(dist_squared)
                # print("\ndist:")
                # print(dist)

                # calculate absolute value of attraction force between each pair of instances
                force_abs = np.divide(self.gap.k_force * self.gai.mass[:, None] * self.gai.mass[None, :], dist_squared, out=np.zeros_like(dist_squared), where=dist_squared != 0)
                # print("\nforce_abs:")
                # print(force_abs)

                # delete forces which comes from too close distances
                force_abs[dist < self.gap.min_dist] = 0
                # print("\nforce_abs:")
                # print(force_abs)

                # limit forces which are too big in current generator model
                force_abs[force_abs > self.gap.max_force] = self.gap.max_force
                # print("\nforce_abs:")
                # print(force_abs)

                # calculate components of attraction force between each pair of instances
                force_div = np.divide(force_abs, dist, out=np.zeros_like(force_abs), where=dist != 0)
                force = force_div[:, :, None] * coor_diff
                # force = (force_abs / dist)[:, :, None] * coor_diff
                # print("\nforce:")
                # print(force)

                # calculate resultant force for each instance
                force_resultant = np.sum(a=force, axis=1)
                # print("\nforce_resultant:")
                # print(force_resultant)

                # force escaping objects to stay close to mass center within range of 'faraway_limit'
                # center = np.sum(a=instances_coor * mass[:, None], axis=0) / mass_sum
                # print("\ncenter=%s" % center)
                dist_center = (self.gai.center - instances_coor) / self.gap.distance_unit
                # print("\ndist_center:")
                # print(dist_center)
                dist_center_abs = np.sqrt(np.sum(a=dist_center ** 2, axis=-1))
                # print("\ndist_center_abs:")
                # print(dist_center_abs)
                force_center_abs = np.divide(dist_center_abs - self.gap.faraway_limit, self.gap.faraway_limit, out=np.zeros_like(dist_center_abs), where=dist_center_abs > self.gap.faraway_limit)
                force_center_abs = force_center_abs ** 2 * self.gap.k_force * self.gai.mass_sum * self.gai.mass
                # print("\nforce_center_abs:")
                # print(force_center_abs)
                force_center = np.divide(force_center_abs, dist_center_abs, out=np.zeros_like(force_center_abs), where=dist_center_abs != 0)
                force_center = force_center[:, None] * dist_center
                # print("\nforce_center:")
                # print(force_center)
                force_resultant += force_center
                # print("\nforce_resultant:")
                # print(force_resultant)

                # calculate acceleration
                acceleration = force_resultant / self.gai.mass[:, None]
                # print("\nacceleration:")
                # print(acceleration)

                # calculate change of velocity
                velocity_delta = acceleration * self.gai.approx_step_time_interval
                # print("\nvelocity_delta:")
                # print(velocity_delta)

                # calculate distance change of instances
                instances_coor_delta = self.gai.approx_step_time_interval * (velocity + velocity_delta / 2)
                # print("\ninstances_coor_delta:")
                # print(instances_coor_delta)

                # translate distance change into coordinates change
                instances_coor_delta *= self.gap.distance_unit
                # print("\ninstances_coor_delta:")
                # print(instances_coor_delta)

                # calculate velocity of instances
                velocity += velocity_delta
                # print("\nvelocity:")
                # print(velocity)

                # calculate location of instances in next time_frame
                instances_coor += instances_coor_delta
                # print("\ninstances_coor:")
                # print(instances_coor)

                # view statistics of current time frame
                # view_statistics_of_absolute_values(force_resultant, "force_resultant")
                # view_statistics_of_absolute_values(acceleration, "acceleration")
                # view_statistics_of_absolute_values(velocity_delta, "velocity_delta")
                # view_statistics_of_absolute_values(velocity, "velocity")
                # view_statistics_of_absolute_values(instances_coor_delta, "instances_coor_delta")

            # generate vector of time frame ids of starting time frame
            time_frame_ids = np.full(shape=self.gai.features_instances_sum, fill_value=time_frame, dtype=np.int32)

            # write starting data of all the features to the output file
            stga_writer.write(
                time_frame_ids=time_frame_ids,
                features_ids=self.gai.features_ids,
                features_instances_ids=self.gai.features_instances_ids,
                x=instances_coor[:, 0],
                y=instances_coor[:, 1]
            )

        # end of file writing
        stga_writer.close()


if __name__ == "__main__":
    print("SpatioTemporalGravitationApproachGenerator main()")

    gap = GravitationApproachParameters(
        area=1000,
        cell_size=5,
        n_colloc=5,
        lambda_1=3,
        lambda_2=30,
        m_clumpy=1,
        m_overlap=1,
        ncfr=0.5,
        ncfn=0.5,
        ncf_proportional=False,
        ndf=3,
        ndfn=50,
        random_seed=0,
        time_unit=1.0,
        distance_unit=1.0,
        approx_steps_number=10,
        min_dist=0.5,
        max_force=np.inf,
        k_force=10,
        mass_param=1.0,
        velocity_param=0.0,
        faraway_limit=1000
    )

    stgag = SpatioTemporalGravitationApproachGenerator(gap=gap)
    stgag.generate(
        time_frames_number=50,
        output_filename="SpatioTemporalGravitationApproachGenerator_output_file.txt"
    )
