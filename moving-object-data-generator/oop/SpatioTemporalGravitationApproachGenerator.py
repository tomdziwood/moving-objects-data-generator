import numpy as np

from oop.SpatialStandardPlacement import SpatialStandardPlacement
from oop.SpatioTemporalWriter import SpatioTemporalWriter
from oop.StandardInitiation import StandardInitiation
from oop.StandardParameters import StandardParameters
from timeit import default_timer as timer


class SpatioTemporalGravitationApproachGenerator:
    def generate(
            self,
            output_file: str = "SpatioTemporalParticlesInteractionsApproachGenerator_output_file.txt",
            time_frames_number: int = 10,
            sp: StandardParameters = StandardParameters()
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

        # keep instances coordinations in one array
        instances_coor = np.column_stack(tup=(ssp.x, ssp.y))
        # print("instances_coor.shape=%s", str(instances_coor.shape))
        # print("\ninstances_coor:")
        # print(instances_coor)

        # create array of instances mass (all equals to 1)
        mass = np.ones_like(ssp.features_ids)

        # create array of instances velocity (all equals to 0)
        velocity = np.zeros_like(instances_coor)

        # define time interval between time frames
        time_interval = 1

        # generate data for next time frames
        for i_time_frame in range(1, time_frames_number):
            print("i_time_frame=%d" % i_time_frame)

            # calculate coordinates difference of each pair of instances
            coor_diff = instances_coor[None, :, :] - instances_coor[:, None, :]
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
            force_abs = np.divide(mass[:, None] * mass[None, :], dist_squared, out=np.zeros_like(dist_squared), where=dist_squared != 0)
            # print("\nforce_abs:")
            # print(force_abs)

            # calculate components of attraction force between each pair of instances
            force_div = np.divide(force_abs, dist, out=np.zeros_like(force_abs), where=dist != 0)
            force = force_div[:, :, None] * coor_diff
            # force = (force_abs / dist)[:, :, None] * coor_diff
            # print("\nforce:")
            # print(force)

            # calculate resultant force for each instance
            force_sum = np.sum(a=force, axis=1)
            # print("\nforce_sum:")
            # print(force_sum)

            # calculate acceleration
            acceleration = force_sum / mass[:, None]
            # print("\nacceleration:")
            # print(acceleration)

            # calculate change of velocity
            velocity_delta = acceleration * time_interval
            # print("\nvelocity_delta:")
            # print(velocity_delta)

            # calculate location of instances in next time_frame
            instances_coor += time_interval * (velocity + velocity_delta / 2)
            # print("\ninstances_coor:")
            # print(instances_coor)

            # calculate velocity of instances
            velocity += velocity_delta
            # print("\nvelocity:")
            # print(velocity)

            # generate vector of time frame ids of starting time frame
            time_frame_ids = np.full(shape=ssp.features_ids.size, fill_value=i_time_frame, dtype=np.int32)

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
        n_colloc=0,
        lambda_1=5,
        lambda_2=100,
        m_clumpy=2,
        m_overlap=3,
        ncfr=0.4,
        ncfn=1,
        ncf_proportional=False,
        ndf=5,
        ndfn=5,
        random_seed=0
    )

    stgag = SpatioTemporalGravitationApproachGenerator()
    stgag.generate(
        output_file="SpatioTemporalStandardGenerator_output_file.txt",
        time_frames_number=10,
        sp=sp
    )
