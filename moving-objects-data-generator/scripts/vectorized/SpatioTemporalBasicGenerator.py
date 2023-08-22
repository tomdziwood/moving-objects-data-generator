import numpy as np


def generate(output_file="SpatialBasicGenerator_output_file.txt", time_frames_number=10, area=1000, cell_size=5, n_base=3, lambda_1=5, lambda_2=100, m_clumpy=1, m_overlap=1, ncfr=1.0, ncfn=1.0, ncf_proportional=False, ndf=2, ndfn=5000, random_seed=None):
    print("generate()")

    # set random seed value
    if random_seed is not None:
        np.random.seed(random_seed)

    # open file to which output will be written
    f = open(file=output_file, mode="w")

    # determine length to each of the n_base base co-locations with poisson distribution (lam=lambda_1)
    base_collocation_lengths = np.random.poisson(lam=lambda_1, size=n_base)
    base_collocation_lengths[base_collocation_lengths < 2] = 2
    print("base_collocation_lengths=%s" % str(base_collocation_lengths))

    # set final length to each of the co-locations according to the m_overlap parameter value
    if m_overlap > 1:
        collocation_lengths = np.repeat(a=base_collocation_lengths + 1, repeats=m_overlap)
    else:
        collocation_lengths = base_collocation_lengths
    print("collocation_lengths=%s" % str(collocation_lengths))

    # determine number of instances to each of the co-locations with poisson distribution (lam=lambda_2)
    collocations_instances_counts = np.random.poisson(lam=lambda_2, size=n_base * m_overlap)
    print("collocations_instances_counts=%s" % str(collocations_instances_counts))

    # determine the total number of features, which take part in co-locations
    collocation_features_sum = np.sum(base_collocation_lengths)
    if m_overlap > 1:
        collocation_features_sum += n_base * m_overlap
    print("collocation_features_sum=%d" % collocation_features_sum)

    # count all instances of every i'th co-location feature
    collocation_features_instances_counts = np.zeros(shape=collocation_features_sum, dtype=np.int32)
    collocation_start_feature_id = 0
    for i_colloc in range(n_base * m_overlap):
        collocation_features = np.arange(collocation_start_feature_id, collocation_start_feature_id + collocation_lengths[i_colloc])
        collocation_features[-1] += i_colloc % m_overlap
        collocation_features_instances_counts[collocation_features] += collocations_instances_counts[i_colloc]
        if (i_colloc + 1) % m_overlap == 0:
            collocation_start_feature_id += collocation_lengths[i_colloc] + m_overlap - 1
    print("collocation_features_instances_counts=%s" % str(collocation_features_instances_counts))

    # express area dimension in spatial cell unit
    area_in_cell_dim = area // cell_size
    print("area_in_cell_dim: ", area_in_cell_dim)

    # determine the total number of features, which take part in co-locations and also are used to generate noise
    collocation_noise_features_sum = round(ncfr * collocation_features_sum)
    print("collocation_noise_features_sum=%d" % collocation_noise_features_sum)

    # chose co-location noise features from co-location features
    collocation_noise_features = np.random.choice(a=collocation_features_sum, size=collocation_noise_features_sum, replace=False)
    collocation_noise_features.sort()
    print("collocation_noise_features=%s" % str(collocation_noise_features))

    # prepare array which holds counts of created instances of the co-location noise feature
    collocation_noise_features_instances_sum = round(ncfn * collocation_features_instances_counts.sum())
    if ncf_proportional:
        # number of the instances of given co-location noise feature is proportional to the number of instances of given feature, which are participating in co-locations
        collocation_noise_features_instances_counts = collocation_noise_features_instances_sum * collocation_features_instances_counts[collocation_noise_features] / collocation_features_instances_counts[collocation_noise_features].sum()
        collocation_noise_features_instances_counts = collocation_noise_features_instances_counts.astype(np.int32)
    else:
        # number of the instances of every co-location noise feature is similar, because co-location noise feature id is chosen randomly with uniform distribution.
        collocation_noise_feature_random_choices = np.random.randint(low=collocation_noise_features_sum, size=collocation_noise_features_instances_sum)
        (_, collocation_noise_features_instances_counts) = np.unique(ar=collocation_noise_feature_random_choices, return_counts=True)
    print("collocation_noise_features_instances_counts=%s" % str(collocation_noise_features_instances_counts))

    # generate vector of features ids of all consecutive instances of co-location noise features
    collocation_noise_features_ids = np.repeat(a=collocation_noise_features, repeats=collocation_noise_features_instances_counts)

    # generate vector of features instances ids of all consecutive instances of co-location noise features
    start = collocation_features_instances_counts[collocation_noise_features]
    length = collocation_noise_features_instances_counts
    collocation_noise_features_instances_ids = np.repeat(a=(start + length - length.cumsum()), repeats=length) + np.arange(collocation_noise_features_instances_sum)
    # print("collocation_noise_features_instances_ids=%s" % collocation_noise_features_instances_ids)

    # initiate basic data of the additional noise features if they are requested
    additional_noise_features_ids = np.array([])
    additional_noise_features_instances_ids = np.array([])
    if ndf > 0:
        # determine number of each of the additional noise features
        additional_noise_features_ids = np.random.randint(low=ndf, size=ndfn) + collocation_features_sum
        (additional_noise_features, additional_noise_features_instances_counts) = np.unique(ar=additional_noise_features_ids, return_counts=True)
        print("additional_noise_features=%s" % str(additional_noise_features))
        print("additional_noise_features_instances_counts=%s" % str(additional_noise_features_instances_counts))

        # generate vector of feature ids of all consecutive instances of additional noise features
        additional_noise_features_ids = np.repeat(a=additional_noise_features, repeats=additional_noise_features_instances_counts)

        # generate vector of features instances ids of all consecutive instances of additional noise features
        additional_noise_features_instances_ids = np.repeat(
            a=(additional_noise_features_instances_counts - additional_noise_features_instances_counts.cumsum()),
            repeats=additional_noise_features_instances_counts
        ) + np.arange(ndfn)

    # generate data for each time frame
    for i_time_frame in range(time_frames_number):
        print("i_time_frame=%d" % i_time_frame)

        collocation_start_feature_id = 0
        collocation_features_instances_counters = np.zeros(shape=collocation_features_sum, dtype=np.int32)

        # generate data of every co-location in given time frame
        for i_colloc in range(n_base * m_overlap):

            # get the features ids of current co-location
            collocation_features = np.arange(collocation_start_feature_id, collocation_start_feature_id + collocation_lengths[i_colloc])
            collocation_features[-1] += i_colloc % m_overlap
            print("collocation_features=%s" % str(collocation_features))

            # calculate total number of all co-location feature instances
            collocation_features_instances_sum = collocations_instances_counts[i_colloc] * collocation_lengths[i_colloc]

            # generate vector of x coordinates of all consecutive instances
            collocation_features_instances_x = np.random.randint(low=area_in_cell_dim, size=(collocations_instances_counts[i_colloc] - 1) // m_clumpy + 1)
            collocation_features_instances_x *= cell_size
            collocation_features_instances_x = collocation_features_instances_x.astype(dtype=np.float64)
            collocation_features_instances_x = np.repeat(a=collocation_features_instances_x, repeats=m_clumpy)[:collocations_instances_counts[i_colloc]]
            collocation_features_instances_x = np.repeat(a=collocation_features_instances_x, repeats=collocation_lengths[i_colloc])
            collocation_features_instances_x += np.random.uniform(high=cell_size, size=collocation_features_instances_sum)

            # generate vector of y coordinates of all consecutive instances
            collocation_features_instances_y = np.random.randint(low=area_in_cell_dim, size=(collocations_instances_counts[i_colloc] - 1) // m_clumpy + 1)
            collocation_features_instances_y *= cell_size
            collocation_features_instances_y = collocation_features_instances_y.astype(dtype=np.float64)
            collocation_features_instances_y = np.repeat(a=collocation_features_instances_y, repeats=m_clumpy)[:collocations_instances_counts[i_colloc]]
            collocation_features_instances_y = np.repeat(a=collocation_features_instances_y, repeats=collocation_lengths[i_colloc])
            collocation_features_instances_y += np.random.uniform(high=cell_size, size=collocation_features_instances_sum)

            # generate vector of features ids of all consecutive instances
            collocation_features_ids = np.tile(A=collocation_features, reps=collocations_instances_counts[i_colloc])

            # generate vector of features instances ids of all consecutive instances
            collocation_features_instances_ids = np.arange(
                start=collocation_features_instances_counters[collocation_start_feature_id],
                stop=collocation_features_instances_counters[collocation_start_feature_id] + collocations_instances_counts[i_colloc]
            )
            collocation_features_instances_ids = np.tile(A=collocation_features_instances_ids, reps=(collocation_lengths[i_colloc] - 1, 1))
            collocation_features_instances_ids = np.concatenate((
                collocation_features_instances_ids,
                np.arange(collocations_instances_counts[i_colloc]).reshape((1, collocations_instances_counts[i_colloc]))
            ))
            collocation_features_instances_ids = collocation_features_instances_ids.T.flatten()

            # generate vector of time frame ids of current time frame
            time_frame_ids = np.full(shape=collocation_features_instances_sum, fill_value=i_time_frame, dtype=np.int32)

            # write data of all co-location features to the output file
            fmt = '%d;%d;%d;%.6f;%.6f\n' * collocation_features_instances_sum
            data = fmt % tuple(np.column_stack(tup=(time_frame_ids, collocation_features_ids, collocation_features_instances_ids, collocation_features_instances_x, collocation_features_instances_y)).ravel())
            f.write(data)

            # increase counts of created instances of the co-location features which occurred in current co-location
            collocation_features_instances_counters[collocation_features] += collocations_instances_counts[i_colloc]

            # change starting feature of next co-location according to the m_overlap parameter value
            if (i_colloc + 1) % m_overlap == 0:
                collocation_start_feature_id += collocation_lengths[i_colloc] + m_overlap - 1

        # generate data of every co-location noise feature in given time frame
        # generate vectors of x and y coordinates of all consecutive instances of co-location noise features
        collocation_noise_features_instances_x = np.random.uniform(high=area, size=collocation_noise_features_instances_sum)
        collocation_noise_features_instances_y = np.random.uniform(high=area, size=collocation_noise_features_instances_sum)

        # generate vector of time frame ids of current time frame
        time_frame_ids = np.full(shape=collocation_noise_features_instances_sum, fill_value=i_time_frame, dtype=np.int32)

        # write data of all co-location noise features to the output file
        fmt = '%d;%d;%d;%.6f;%.6f\n' * collocation_noise_features_instances_sum
        data = fmt % tuple(np.column_stack(tup=(time_frame_ids, collocation_noise_features_ids, collocation_noise_features_instances_ids, collocation_noise_features_instances_x, collocation_noise_features_instances_y)).ravel())
        f.write(data)

        # generate additional noise features if they are requested in given time frame
        if ndf > 0:
            # generate vectors of x and y coordinates of all consecutive instances of additional noise features
            additional_noise_features_instances_x = np.random.uniform(high=area, size=ndfn)
            additional_noise_features_instances_y = np.random.uniform(high=area, size=ndfn)

            # generate vector of time frame ids of current time frame
            time_frame_ids = np.full(shape=ndfn, fill_value=i_time_frame, dtype=np.int32)

            # write data of all additional noise features to the output file
            fmt = '%d;%d;%d;%.6f;%.6f\n' * ndfn
            data = fmt % tuple(np.column_stack(tup=(time_frame_ids, additional_noise_features_ids, additional_noise_features_instances_ids, additional_noise_features_instances_x, additional_noise_features_instances_y)).ravel())
            f.write(data)

    # end of file writing
    f.close()


def main():
    print("main()")

    generate(output_file="SpatioTemporalBasicGenerator_output_file.txt", time_frames_number=10, area=1000, cell_size=5, n_base=2, lambda_1=5, lambda_2=100, m_clumpy=2, m_overlap=3, ncfr=0.4, ncfn=1, ncf_proportional=False, ndf=5, ndfn=200, random_seed=0)


if __name__ == "__main__":
    main()
