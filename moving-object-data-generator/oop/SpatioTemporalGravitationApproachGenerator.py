import numpy as np

from oop.SpatialStandardPlacement import SpatialStandardPlacement
from oop.SpatioTemporalWriter import SpatioTemporalWriter
from oop.StandardInitiation import StandardInitiation
from oop.StandardParameters import StandardParameters


def view_statistics_of_absolute_values(array, array_name):
    array_abs = np.sqrt(np.sum(a=array ** 2, axis=-1))
    print("%s:\n\tmin:\t%.12f\n\tavg:\t%.12f\n\tmax:\t%.12f\n" % (array_name, array_abs.min(), array_abs.mean(), array_abs.max()))


class SpatioTemporalGravitationApproachGenerator:

    def generate(
            self,
            output_file: str = "SpatioTemporalParticlesInteractionsApproachGenerator_output_file.txt",
            time_frames_number: int = 10,
            sp: StandardParameters = StandardParameters(),
            time_unit: float = 1.0,
            distance_unit: float = 1.0,
            approx_steps_number: int = 10,
            min_dist: float = 0.01,
            max_force: float = np.inf,
            k_force: float = 1.0,
            mass_lambda: float = 1.0
    ):

        print("generate()")

        # open file to which output will be written
        st_writer = SpatioTemporalWriter(output_file=output_file)

        # prepare all variables and vectors required to generate data at every time frame
        si = StandardInitiation()
        si.initiate(sp=sp)

        # create class object, which holds all data of the objects starting placement
        ssp = SpatialStandardPlacement()

        # place all objects at starting position
        ssp.place(si)

        # generate vector of time frame ids of starting time frame
        time_frame_ids = np.full(shape=ssp.features_ids.size, fill_value=0, dtype=np.int32)

        # write starting data of all the features to the output file
        st_writer.write(
            time_frame_ids=time_frame_ids,
            features_ids=ssp.features_ids,
            features_instances_ids=ssp.features_instances_ids,
            x=ssp.x,
            y=ssp.y
        )

        # keep instances coordinates in one array
        instances_coor = np.column_stack(tup=(ssp.x, ssp.y))
        # print("instances_coor.shape=%s" % str(instances_coor.shape))
        # print("\ninstances_coor:")
        # print(instances_coor)

        # create array of instances mass (all equals to 1)
        # each type of feature has own mean mass value drawn from gamma distribution
        feature_mass_mu = np.random.gamma(shape=mass_lambda, scale=1.0, size=si.collocation_features_sum + sp.ndf)
        print("feature_mass_mu=%s" % str(feature_mass_mu))
        # each instance of given type feature has own mass value drawn from normal distribution
        mass = np.random.normal(loc=feature_mass_mu[ssp.features_ids], scale=feature_mass_mu[ssp.features_ids] / 5, size=ssp.features_ids.size)
        mass[mass < 0] *= -1

        # create array of instances velocity (all equals to 0)
        velocity = np.zeros_like(instances_coor)

        # define time interval between time frames
        time_interval = 1 / time_unit

        # divide time interval of single time frame into steps of equal duration
        approx_step_time_interval = time_interval / approx_steps_number

        # generate data for next time frames
        for time_frame in range(1, time_frames_number):
            print("time_frame %d of %d" % (time_frame + 1, time_frames_number))

            # calculate positions in next time frame by making 'approx_steps_number' steps
            for approx_step in range(approx_steps_number):
                # print("\tapprox_step %d of %d" % (approx_step + 1, approx_steps_number))

                # calculate coordinates difference of each pair of instances
                coor_diff = instances_coor[None, :, :] - instances_coor[:, None, :]
                # print("\ncoor_diff:")
                # print(coor_diff)

                # translate coordinates difference into distance unit
                coor_diff /= distance_unit
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
                force_abs = np.divide(k_force * mass[:, None] * mass[None, :], dist_squared, out=np.zeros_like(dist_squared), where=dist_squared != 0)
                # print("\nforce_abs:")
                # print(force_abs)

                # delete forces which comes from to low distances
                force_abs[dist < min_dist] = 0
                # print("\nforce_abs:")
                # print(force_abs)

                # limit forces which are too big in current generator model
                force_abs[force_abs > max_force] = max_force
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

                # calculate acceleration
                acceleration = force_resultant / mass[:, None]
                # print("\nacceleration:")
                # print(acceleration)

                # calculate change of velocity
                velocity_delta = acceleration * approx_step_time_interval
                # print("\nvelocity_delta:")
                # print(velocity_delta)

                # calculate distance change of instances
                instances_coor_delta = approx_step_time_interval * (velocity + velocity_delta / 2)
                # print("\ninstances_coor_delta:")
                # print(instances_coor_delta)

                # translate distance change into coordinates change
                instances_coor_delta *= distance_unit
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
            time_frame_ids = np.full(shape=ssp.features_ids.size, fill_value=time_frame, dtype=np.int32)

            # write starting data of all the features to the output file
            st_writer.write(
                time_frame_ids=time_frame_ids,
                features_ids=ssp.features_ids,
                features_instances_ids=ssp.features_instances_ids,
                x=instances_coor[:, 0],
                y=instances_coor[:, 1]
            )

        # end of file writing
        st_writer.close()


if __name__ == "__main__":
    print("main()")

    sp = StandardParameters(
        area=1000,
        cell_size=5,
        n_colloc=1,
        lambda_1=10,
        lambda_2=1,
        m_clumpy=1,
        m_overlap=1,
        ncfr=0.0,
        ncfn=0.0,
        ncf_proportional=False,
        ndf=0,
        ndfn=0,
        random_seed=2
    )

    stgag = SpatioTemporalGravitationApproachGenerator()
    stgag.generate(
        output_file="SpatioTemporalGravitationApproachGenerator_output_file.txt",
        time_frames_number=500,
        sp=sp,
        approx_steps_number=100,
        min_dist=0.5,
        k_force=10
    )
