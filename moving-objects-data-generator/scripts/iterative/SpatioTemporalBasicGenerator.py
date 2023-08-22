import numpy as np


def generate(output_file="SpatioTemporalBasicGenerator_output_file.txt", time_frames_number=10, area=1000, cell_size=5, n_base=3, lambda_1=5, lambda_2=100, m_clumpy=1, m_overlap=1, ncfr=1.0, ncfn=1.0, ncf_proportional=False, ndf=2, ndfn=5000, random_seed=None):
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
    collocation_instances_counts = np.random.poisson(lam=lambda_2, size=n_base * m_overlap)
    print("collocation_instances_counts=%s" % str(collocation_instances_counts))

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
        collocation_features_instances_counts[collocation_features] += collocation_instances_counts[i_colloc]
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

    # determine number of each of the additional noise features if they are requested
    (additional_noise_features, additional_noise_features_instances_counts) = (np.array([]), np.array([]))
    if ndf > 0:
        additional_noise_features_ids = np.random.randint(low=ndf, size=ndfn) + collocation_features_sum
        (additional_noise_features, additional_noise_features_instances_counts) = np.unique(ar=additional_noise_features_ids, return_counts=True)
        print("additional_noise_features=%s" % str(additional_noise_features))
        print("additional_noise_features_instances_counts=%s" % str(additional_noise_features_instances_counts))

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

            # generate data of every instance of current co-location
            collocation_instance_id = 0
            while collocation_instance_id < collocation_instances_counts[i_colloc]:

                # determine spatial cell with uniform distribution
                cell_x_id = np.random.randint(low=area_in_cell_dim)
                cell_y_id = np.random.randint(low=area_in_cell_dim)
                # print("ids:\t(%d, %d)" % (cell_x_id, cell_y_id))

                # get starting coordinations of chosen cell
                cell_x = cell_x_id * cell_size
                cell_y = cell_y_id * cell_size
                # print("cell coor:\t(%d, %d)" % (cell_x, cell_y))

                # determine number of instances placed in the same spatial cell according to the m_clumpy parameter value
                m_clumpy_repeats = min(m_clumpy, collocation_instances_counts[i_colloc] - collocation_instance_id)

                # place m_clumpy_repeats number of co-location instances in the chosen cell
                for _ in range(m_clumpy_repeats):

                    # determine coordinations of every co-location feature
                    for collocation_feature in collocation_features:
                        instance_x = cell_x + np.random.uniform() * cell_size
                        instance_y = cell_y + np.random.uniform() * cell_size
                        # print("collocation_feature: %d\tinst coor:\t(%f, %f)" % (collocation_feature, instance_x, instance_y))
                        f.write("%d;%d;%d;%.6f;%.6f\n" % (i_time_frame, collocation_feature, collocation_features_instances_counters[collocation_feature], instance_x, instance_y))
                        collocation_features_instances_counters[collocation_feature] += 1

                    collocation_instance_id += 1

            # change starting feature of next co-location according to the m_overlap parameter value
            if (i_colloc + 1) % m_overlap == 0:
                collocation_start_feature_id += collocation_lengths[i_colloc] + m_overlap - 1

        # generate data of every co-location noise feature in given time frame
        for i_feature in range(collocation_noise_features_sum):
            collocation_noise_feature_id = collocation_noise_features[i_feature]
            collocation_noise_feature_instances_count = collocation_noise_features_instances_counts[i_feature]
            # print("collocation_noise_feature_instances_count=%d" % collocation_noise_feature_instances_count)
            collocation_noise_feature_instance_id = collocation_features_instances_counts[collocation_noise_feature_id]

            # generate data of every instance of current co-location noise feature
            for _ in range(collocation_noise_feature_instances_count):
                instance_x = np.random.uniform(high=area)
                instance_y = np.random.uniform(high=area)
                # print("noise feature %d \t feature instance %d \t coor:\t(%f, %f)" % (collocation_noise_feature_id, collocation_noise_feature_instance_id, instance_x, instance_y))
                f.write("%d;%d;%d;%.6f;%.6f\n" % (i_time_frame, collocation_noise_feature_id, collocation_noise_feature_instance_id, instance_x, instance_y))
                collocation_noise_feature_instance_id += 1

        # generate additional noise features if they are requested in given time frame
        if ndf > 0:
            for i_feature in range(additional_noise_features.size):
                additional_noise_feature = additional_noise_features[i_feature]
                for additional_noise_feature_instance_id in range(additional_noise_features_instances_counts[i_feature]):
                    instance_x = np.random.uniform(high=area)
                    instance_y = np.random.uniform(high=area)
                    # print("additional noise %d inst %d \t coor:\t(%f, %f)" % (collocation_start_feature_id + additional_noise_feature, additional_noise_feature_instance[additional_noise_feature], instance_x, instance_y))
                    f.write("%d;%d;%d;%.6f;%.6f\n" % (i_time_frame, additional_noise_feature, additional_noise_feature_instance_id, instance_x, instance_y))

    # end of file writing
    f.close()


def main():
    print("main()")

    generate(output_file="SpatioTemporalBasicGenerator_output_file.txt", time_frames_number=10, area=1000, cell_size=5, n_base=2, lambda_1=5, lambda_2=100, m_clumpy=2, m_overlap=3, ncfr=0.4, ncfn=1, ncf_proportional=False, ndf=5, ndfn=200, random_seed=0)


if __name__ == "__main__":
    main()
