import numpy as np
from StandardInitiation import StandardInitiation
from SpatioTemporalWriter import SpatioTemporalWriter


class SpatioTemporalStandardGenerator:
    def generate(
            self,
            output_file="SpatialStandardGenerator_output_file.txt",
            time_frames_number=10,
            area=1000,
            cell_size=5,
            n_colloc=3,
            lambda_1=5,
            lambda_2=100,
            m_clumpy=1,
            m_overlap=1,
            ncfr=1.0,
            ncfn=1.0,
            ncf_proportional=False,
            ndf=2,
            ndfn=5000,
            random_seed=None):

        print("generate()")

        # open file to which output will be written
        st_writer = SpatioTemporalWriter(output_file=output_file)

        # prepare all variables and vectors required to generate data at every time frame
        si = StandardInitiation(
            area=area,
            cell_size=cell_size,
            n_colloc=n_colloc,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            m_overlap=m_overlap,
            ncfr=ncfr,
            ncfn=ncfn,
            ncf_proportional=ncf_proportional,
            ndf=ndf,
            ndfn=ndfn,
            random_seed=random_seed
        )

        # generate data for each time frame
        for i_time_frame in range(time_frames_number):
            print("i_time_frame=%d" % i_time_frame)

            collocation_start_feature_id = 0
            collocation_features_instances_counters = np.zeros(shape=si.collocation_features_sum, dtype=np.int32)

            # generate data of every co-location in given time frame
            for i_colloc in range(n_colloc * m_overlap):

                # get the features ids of current co-location
                collocation_features = np.arange(collocation_start_feature_id, collocation_start_feature_id + si.collocation_lengths[i_colloc])
                collocation_features[-1] += i_colloc % m_overlap
                print("collocation_features=%s" % str(collocation_features))

                # calculate total number of all co-location feature instances
                collocation_features_instances_sum = si.collocations_instances_counts[i_colloc] * si.collocation_lengths[i_colloc]

                # generate vector of x coordinates of all the consecutive instances
                collocation_features_instances_x = np.random.randint(low=si.area_in_cell_dim, size=(si.collocations_instances_counts[i_colloc] - 1) // m_clumpy + 1)
                collocation_features_instances_x *= cell_size
                collocation_features_instances_x = collocation_features_instances_x.astype(dtype=np.float64)
                collocation_features_instances_x = np.repeat(a=collocation_features_instances_x, repeats=m_clumpy)[:si.collocations_instances_counts[i_colloc]]
                collocation_features_instances_x = np.repeat(a=collocation_features_instances_x, repeats=si.collocation_lengths[i_colloc])
                collocation_features_instances_x += np.random.uniform(high=cell_size, size=collocation_features_instances_sum)

                # generate vector of y coordinates of all the consecutive instances
                collocation_features_instances_y = np.random.randint(low=si.area_in_cell_dim, size=(si.collocations_instances_counts[i_colloc] - 1) // m_clumpy + 1)
                collocation_features_instances_y *= cell_size
                collocation_features_instances_y = collocation_features_instances_y.astype(dtype=np.float64)
                collocation_features_instances_y = np.repeat(a=collocation_features_instances_y, repeats=m_clumpy)[:si.collocations_instances_counts[i_colloc]]
                collocation_features_instances_y = np.repeat(a=collocation_features_instances_y, repeats=si.collocation_lengths[i_colloc])
                collocation_features_instances_y += np.random.uniform(high=cell_size, size=collocation_features_instances_sum)

                # generate vector of features ids of all the consecutive instances
                collocation_features_ids = np.tile(A=collocation_features, reps=si.collocations_instances_counts[i_colloc])

                # generate vector of features instances ids of all the consecutive instances
                collocation_features_instances_ids = np.arange(
                    start=collocation_features_instances_counters[collocation_start_feature_id],
                    stop=collocation_features_instances_counters[collocation_start_feature_id] + si.collocations_instances_counts[i_colloc]
                )
                collocation_features_instances_ids = np.tile(A=collocation_features_instances_ids, reps=(si.collocation_lengths[i_colloc] - 1, 1))
                collocation_features_instances_ids = np.concatenate((
                    collocation_features_instances_ids,
                    np.arange(si.collocations_instances_counts[i_colloc]).reshape((1, si.collocations_instances_counts[i_colloc]))
                ))
                collocation_features_instances_ids = collocation_features_instances_ids.T.flatten()

                # generate vector of time frame ids of current time frame
                time_frame_ids = np.full(shape=collocation_features_instances_sum, fill_value=i_time_frame, dtype=np.int32)

                # write data of all the co-location features to the output file
                st_writer.write(
                    features_instances_sum=collocation_features_instances_sum,
                    time_frame_ids=time_frame_ids,
                    features_ids=collocation_features_ids,
                    features_instances_ids=collocation_features_instances_x,
                    x=collocation_features_instances_x,
                    y=collocation_features_instances_y
                )

                # increase counts of created instances of the co-location features which occurred in current co-location
                collocation_features_instances_counters[collocation_features] += si.collocations_instances_counts[i_colloc]

                # change starting feature of next co-location according to the m_overlap parameter value
                if (i_colloc + 1) % m_overlap == 0:
                    collocation_start_feature_id += si.collocation_lengths[i_colloc] + m_overlap - 1

            # generate data of every co-location noise feature in given time frame
            # generate vectors of x and y coordinates of all the consecutive instances of co-location noise features
            collocation_noise_features_instances_x = np.random.uniform(high=area, size=si.collocation_noise_features_instances_sum)
            collocation_noise_features_instances_y = np.random.uniform(high=area, size=si.collocation_noise_features_instances_sum)

            # generate vector of time frame ids of current time frame
            time_frame_ids = np.full(shape=si.collocation_noise_features_instances_sum, fill_value=i_time_frame, dtype=np.int32)

            # write data of all the co-location noise features to the output file
            st_writer.write(
                features_instances_sum=si.collocation_noise_features_instances_sum,
                time_frame_ids=time_frame_ids,
                features_ids=si.collocation_noise_features_ids,
                features_instances_ids=si.collocation_noise_features_instances_ids,
                x=collocation_noise_features_instances_x,
                y=collocation_noise_features_instances_y
            )

            # generate additional noise features if they are requested in given time frame
            if ndf > 0:
                # generate vectors of x and y coordinates of all the consecutive instances of additional noise features
                additional_noise_features_instances_x = np.random.uniform(high=area, size=ndfn)
                additional_noise_features_instances_y = np.random.uniform(high=area, size=ndfn)

                # generate vector of time frame ids of current time frame
                time_frame_ids = np.full(shape=ndfn, fill_value=i_time_frame, dtype=np.int32)

                # write data of all the additional noise features to the output file
                st_writer.write(
                    features_instances_sum=ndfn,
                    time_frame_ids=time_frame_ids,
                    features_ids=si.additional_noise_features_ids,
                    features_instances_ids=si.additional_noise_features_instances_ids,
                    x=additional_noise_features_instances_x,
                    y=additional_noise_features_instances_y
                )

        # end of file writing
        st_writer.close()


if __name__ == "__main__":
    print("main()")

    stsg = SpatioTemporalStandardGenerator()
    stsg.generate(
        output_file="SpatioTemporalStandardGenerator_output_file.txt",
        time_frames_number=10,
        area=1000,
        cell_size=5,
        n_colloc=2,
        lambda_1=5,
        lambda_2=100,
        m_clumpy=2,
        m_overlap=3,
        ncfr=0.4,
        ncfn=1,
        ncf_proportional=False,
        ndf=5,
        ndfn=200,
        random_seed=0
    )
