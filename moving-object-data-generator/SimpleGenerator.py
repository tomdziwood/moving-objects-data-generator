import random
import numpy as np


def generate(output_file="output_file.txt", area=1000, cell_size=5, n_colloc=3, lambda_1=5, lambda_2=100, m_clumpy=1, m_overlap=1, ncfr=1.0, ncfn=1.0, ndf=2, ndfn=5000):
    print("generate()")
    f = open(file=output_file, mode="w")
    base_collocation_length_array = np.random.poisson(lam=lambda_1, size=n_colloc)
    base_collocation_length_array[base_collocation_length_array < 2] = 2
    print("base_collocation_length_array=%s" % str(base_collocation_length_array))
    collocation_instances_number_array = np.random.poisson(lam=lambda_2, size=n_colloc)
    print("collocation_instances_number_array=%s" % str(collocation_instances_number_array))

    collocation_features_number = np.sum(base_collocation_length_array)
    print("collocation_features_number=%d" % collocation_features_number)

    last_colloc_id = 0
    area_in_cell_dim = area / cell_size
    for i_colloc in range(n_colloc):
        collocation_feature_ids = list(range(last_colloc_id, last_colloc_id + base_collocation_length_array[i_colloc]))
        print(collocation_feature_ids)

        for i_colloc_inst in range(collocation_instances_number_array[i_colloc]):
            cell_x_id = random.randint(0, area_in_cell_dim)
            cell_y_id = random.randint(0, area_in_cell_dim)
            # print("ids:\t(%d, %d)" % (cell_x_id, cell_y_id))

            cell_x = cell_x_id * cell_size
            cell_y = cell_y_id * cell_size
            # print("cell coor:\t(%d, %d)" % (cell_x, cell_y))

            for i_feature in range(base_collocation_length_array[i_colloc]):
                instance_x = cell_x + random.random() * cell_size
                instance_y = cell_y + random.random() * cell_size
                # print("i_feature: %d\tinst coor:\t(%f, %f)" % (collocation_feature_ids[i_feature], instance_x, instance_y))
                f.write("%d %d %f %f\n" % (collocation_feature_ids[i_feature], i_colloc_inst, instance_x, instance_y))

        last_colloc_id += base_collocation_length_array[i_colloc]

    collocation_noise_feature_number = round(ncfr * collocation_features_number)
    print("collocation_noise_feature_number=%d" % collocation_noise_feature_number)

    # collocation_noise_feature = random.sample(population=range(collocation_features_number), k=collocation_noise_feature_number)
    collocation_noise_feature = np.random.choice(a=range(collocation_features_number), size=collocation_noise_feature_number, replace=False)
    collocation_noise_feature.sort()
    print("collocation_noise_feature=%s" % str(collocation_noise_feature))

    collocation_features_instances_number_array = []
    for index in range(len(base_collocation_length_array)):
        for _ in range(base_collocation_length_array[index]):
            collocation_features_instances_number_array.append(collocation_instances_number_array[index])

    collocation_noise_feature_instances_number_array = []
    for collocation_id in collocation_noise_feature:
        collocation_noise_feature_instances_number_array.append(
            int(ncfn * collocation_features_instances_number_array[collocation_id])
        )
    print("collocation_noise_feature_instances_number_array=%s" % str(collocation_noise_feature_instances_number_array))

    for i_feature in range(len(collocation_noise_feature)):
        feature = collocation_noise_feature[i_feature]
        collocation_noise_feature_instances_number = collocation_noise_feature_instances_number_array[i_feature]
        # print("collocation_noise_feature_instances_number=%d" % collocation_noise_feature_instances_number)
        feature_instance = collocation_features_instances_number_array[feature]
        for i_instance in range(collocation_noise_feature_instances_number):
            instance_x = random.random() * area
            instance_y = random.random() * area
            # print("noise feature %d \t feature instance %d \t coor:\t(%f, %f)" % (feature, feature_instance, instance_x, instance_y))
            f.write("%d %d %f %f\n" % (feature, feature_instance, instance_x, instance_y))
            feature_instance += 1

    for additional_noise_feature in range(last_colloc_id, last_colloc_id + ndf):
        print("additional_noise_feature=%d" % additional_noise_feature)
        for additional_noise_feature_instance in range(ndfn):
            instance_x = random.random() * area
            instance_y = random.random() * area
            # print("additional noise %d inst %d \t coor:\t(%f, %f)" % (additional_noise_feature, additional_noise_feature_instance, instance_x, instance_y))
            f.write("%d %d %f %f\n" % (additional_noise_feature, additional_noise_feature_instance, instance_x, instance_y))

    f.close()



def main():
    print("main()")
    # generate(output_file="output_file.txt", area=1000, cell_size=5, n_colloc=3, lambda_1=5, lambda_2=100, m_clumpy=1, m_overlap=1, ncfr=0.0, ncfn=1.0, ndf=0, ndfn=10)
    generate(output_file="output_file.txt", area=1000, cell_size=5, n_colloc=2, lambda_1=5, lambda_2=100, m_clumpy=1, m_overlap=1, ncfr=0.0, ncfn=1.0, ndf=0, ndfn=10)



if __name__ == "__main__":
    main()
