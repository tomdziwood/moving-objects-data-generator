import numpy as np


def generate(output_file="output_file.txt", area=1000, cell_size=5, n_colloc=3, lambda_1=5, lambda_2=100, m_clumpy=1, m_overlap=1, ncfr=1.0, ncfn=1.0, ndf=2, ndfn=5000, random_seed=None):
    print("generate()")

    if random_seed is not None:
        np.random.seed(random_seed)

    f = open(file=output_file, mode="w")
    base_collocation_lengths = np.random.poisson(lam=lambda_1, size=n_colloc)
    base_collocation_lengths[base_collocation_lengths < 2] = 2
    print("base_collocation_lengths=%s" % str(base_collocation_lengths))
    collocation_instances_counts = np.random.poisson(lam=lambda_2, size=n_colloc)
    print("collocation_instances_counts=%s" % str(collocation_instances_counts))

    collocation_features_sum = np.sum(base_collocation_lengths)
    print("collocation_features_sum=%d" % collocation_features_sum)

    last_colloc_id = 0
    area_in_cell_dim = area // cell_size
    print("area_in_cell_dim: ", area_in_cell_dim)
    for i_colloc in range(n_colloc):
        collocation_features = np.arange(last_colloc_id, last_colloc_id + base_collocation_lengths[i_colloc])
        print("collocation_features=%s" % str(collocation_features))

        collocation_feature_instance_id = 0
        while collocation_feature_instance_id < collocation_instances_counts[i_colloc]:
            cell_x_id = np.random.randint(low=area_in_cell_dim)
            cell_y_id = np.random.randint(low=area_in_cell_dim)
            # print("ids:\t(%d, %d)" % (cell_x_id, cell_y_id))

            cell_x = cell_x_id * cell_size
            cell_y = cell_y_id * cell_size
            # print("cell coor:\t(%d, %d)" % (cell_x, cell_y))

            m_clumpy_repeats = min(m_clumpy, collocation_instances_counts[i_colloc] - collocation_feature_instance_id)
            for _ in range(m_clumpy_repeats):
                for i_feature in range(base_collocation_lengths[i_colloc]):
                    instance_x = cell_x + np.random.uniform() * cell_size
                    instance_y = cell_y + np.random.uniform() * cell_size
                    # print("i_feature: %d\tinst coor:\t(%f, %f)" % (collocation_features[i_feature], instance_x, instance_y))
                    f.write("%d %d %f %f\n" % (collocation_features[i_feature], collocation_feature_instance_id, instance_x, instance_y))

                collocation_feature_instance_id += 1

        last_colloc_id += base_collocation_lengths[i_colloc]

    # collocation noise features
    collocation_noise_features_sum = round(ncfr * collocation_features_sum)
    print("collocation_noise_features_sum=%d" % collocation_noise_features_sum)

    collocation_noise_features = np.random.choice(a=collocation_features_sum, size=collocation_noise_features_sum, replace=False)
    collocation_noise_features.sort()
    print("collocation_noise_features=%s" % str(collocation_noise_features))

    collocation_features_instances_counts = np.repeat(a=collocation_instances_counts, repeats=base_collocation_lengths)
    print("collocation_features_instances_counts=%s" % str(collocation_features_instances_counts))

    collocation_noise_features_instances_counts = ncfn * collocation_features_instances_counts[collocation_noise_features]
    collocation_noise_features_instances_counts = collocation_noise_features_instances_counts.astype(np.int32)
    print("collocation_noise_features_instances_counts=%s" % str(collocation_noise_features_instances_counts))

    for i_feature in range(collocation_noise_features_sum):
        collocation_noise_feature_id = collocation_noise_features[i_feature]
        collocation_noise_feature_instances_count = collocation_noise_features_instances_counts[i_feature]
        # print("collocation_noise_feature_instances_count=%d" % collocation_noise_feature_instances_count)
        collocation_noise_feature_instance_id = collocation_features_instances_counts[collocation_noise_feature_id]
        for _ in range(collocation_noise_feature_instances_count):
            instance_x = np.random.uniform(high=area)
            instance_y = np.random.uniform(high=area)
            # print("noise feature %d \t feature instance %d \t coor:\t(%f, %f)" % (collocation_noise_feature_id, collocation_noise_feature_instance_id, instance_x, instance_y))
            f.write("%d %d %f %f\n" % (collocation_noise_feature_id, collocation_noise_feature_instance_id, instance_x, instance_y))
            collocation_noise_feature_instance_id += 1

    # additional noise feature
    if ndf > 0:
        additional_noise_feature_instance = np.zeros(shape=ndf, dtype=np.int32)
        for _ in range(ndfn):
            additional_noise_feature = np.random.randint(low=ndf)
            instance_x = np.random.uniform(high=area)
            instance_y = np.random.uniform(high=area)
            # print("additional noise %d inst %d \t coor:\t(%f, %f)" % (last_colloc_id + additional_noise_feature, additional_noise_feature_instance[additional_noise_feature], instance_x, instance_y))
            f.write("%d %d %f %f\n" % (last_colloc_id + additional_noise_feature, additional_noise_feature_instance[additional_noise_feature], instance_x, instance_y))
            additional_noise_feature_instance[additional_noise_feature] += 1

    f.close()


def main():
    print("main()")

    # generate(output_file="output_file.txt", area=1000, cell_size=5, n_colloc=3, lambda_1=5, lambda_2=100, m_clumpy=1, m_overlap=1, ncfr=0.0, ncfn=1.0, ndf=0, ndfn=10, random_seed=0)
    generate(output_file="output_file.txt", area=1000, cell_size=5, n_colloc=2, lambda_1=5, lambda_2=100, m_clumpy=1, m_overlap=1, ncfr=0.4, ncfn=0.8, ndf=5, ndfn=200, random_seed=0)


if __name__ == "__main__":
    main()
