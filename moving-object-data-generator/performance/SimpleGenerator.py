import random
import numpy as np
from timeit import default_timer as timer


def generate_collocation_feature_1(area=1000, cell_size=5, n_colloc=3, lambda_1=5, lambda_2=100):
    np.random.seed(0)
    base_collocation_length_array = np.random.poisson(lam=lambda_1, size=n_colloc)
    base_collocation_length_array[base_collocation_length_array < 2] = 2
    # print("base_collocation_length_array=%s" % str(base_collocation_length_array))
    collocation_instances_number_array = np.random.poisson(lam=lambda_2, size=n_colloc)
    # print("collocation_instances_number_array=%s" % str(collocation_instances_number_array))

    collocation_features_number = np.sum(base_collocation_length_array)
    # print("collocation_features_number=%d" % collocation_features_number)

    last_colloc_id = 0
    area_in_cell_dim = area // cell_size
    for i_colloc in range(n_colloc):
        collocation_feature_ids = list(range(last_colloc_id, last_colloc_id + base_collocation_length_array[i_colloc]))
        # print(collocation_feature_ids)

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

        last_colloc_id += base_collocation_length_array[i_colloc]


def generate_collocation_feature_2(area=1000, cell_size=5, n_colloc=3, lambda_1=5, lambda_2=100):
    np.random.seed(0)
    base_collocation_length_array = np.random.poisson(lam=lambda_1, size=n_colloc)
    base_collocation_length_array[base_collocation_length_array < 2] = 2
    # print("base_collocation_length_array=%s" % str(base_collocation_length_array))
    collocation_instances_number_array = np.random.poisson(lam=lambda_2, size=n_colloc)
    # print("collocation_instances_number_array=%s" % str(collocation_instances_number_array))

    collocation_features_number = np.sum(base_collocation_length_array)
    # print("collocation_features_number=%d" % collocation_features_number)

    last_colloc_id = 0
    area_in_cell_dim = area // cell_size
    for i_colloc in range(n_colloc):
        collocation_feature_ids = np.arange(last_colloc_id, last_colloc_id + base_collocation_length_array[i_colloc])
        # print(collocation_feature_ids)

        number_of_all_features_instances_of_collocation = collocation_instances_number_array[i_colloc] * base_collocation_length_array[i_colloc]

        instance_x = np.random.randint(low=area_in_cell_dim, size=collocation_instances_number_array[i_colloc])
        instance_x = instance_x * cell_size
        instance_x = np.repeat(a=instance_x, repeats=base_collocation_length_array[i_colloc])
        instance_x = instance_x + np.random.uniform(high=cell_size, size=number_of_all_features_instances_of_collocation)

        instance_y = np.random.randint(low=area_in_cell_dim, size=collocation_instances_number_array[i_colloc])
        instance_y = instance_y * cell_size
        instance_y = np.repeat(a=instance_y, repeats=base_collocation_length_array[i_colloc])
        instance_y = instance_y + np.random.uniform(high=cell_size, size=number_of_all_features_instances_of_collocation)

        feature_id = np.tile(A=collocation_feature_ids, reps=collocation_instances_number_array[i_colloc])

        feature_instance_id = np.arange(collocation_instances_number_array[i_colloc])
        feature_instance_id = np.repeat(a=feature_instance_id, repeats=base_collocation_length_array[i_colloc])

        last_colloc_id += base_collocation_length_array[i_colloc]


def generate_collocation_feature_3(area=1000, cell_size=5, n_colloc=3, lambda_1=5, lambda_2=100):
    np.random.seed(0)
    base_collocation_length_array = np.random.poisson(lam=lambda_1, size=n_colloc)
    base_collocation_length_array[base_collocation_length_array < 2] = 2
    # print("base_collocation_length_array=%s" % str(base_collocation_length_array))
    collocation_instances_number_array = np.random.poisson(lam=lambda_2, size=n_colloc)
    # print("collocation_instances_number_array=%s" % str(collocation_instances_number_array))

    collocation_features_number = np.sum(base_collocation_length_array)
    # print("collocation_features_number=%d" % collocation_features_number)

    last_colloc_id = 0
    area_in_cell_dim = area // cell_size
    for i_colloc in range(n_colloc):
        collocation_feature_ids = np.arange(last_colloc_id, last_colloc_id + base_collocation_length_array[i_colloc])
        # print(collocation_feature_ids)

        number_of_all_features_instances_of_collocation = collocation_instances_number_array[i_colloc] * base_collocation_length_array[i_colloc]

        instance_x = np.random.randint(low=area_in_cell_dim, size=collocation_instances_number_array[i_colloc])
        instance_x *= cell_size
        instance_x = instance_x.astype(dtype='float64')
        instance_x = np.repeat(a=instance_x, repeats=base_collocation_length_array[i_colloc])
        instance_x += np.random.uniform(high=cell_size, size=number_of_all_features_instances_of_collocation)

        instance_y = np.random.randint(low=area_in_cell_dim, size=collocation_instances_number_array[i_colloc])
        instance_y *= cell_size
        instance_y = instance_y.astype(dtype='float64')
        instance_y = np.repeat(a=instance_y, repeats=base_collocation_length_array[i_colloc])
        instance_y += np.random.uniform(high=cell_size, size=number_of_all_features_instances_of_collocation)

        feature_id = np.tile(A=collocation_feature_ids, reps=collocation_instances_number_array[i_colloc])

        feature_instance_id = np.arange(collocation_instances_number_array[i_colloc])
        feature_instance_id = np.repeat(a=feature_instance_id, repeats=base_collocation_length_array[i_colloc])

        last_colloc_id += base_collocation_length_array[i_colloc]


def test_generate_collocation_feature():
    print("test_generate_collocation_feature execute")
    parameters = {
        "area": 1000,
        "cell_size": 5,
        "n_colloc": 10,
        "lambda_1": 5,
        "lambda_2": 1000
    }

    loops_number = 100
    start = timer()
    for _ in range(loops_number):
        generate_collocation_feature_1(**parameters)
    end = timer()
    print("average time execution of function generate_collocation_feature_1:\t%.12f [s]" % ((end - start) / loops_number))

    loops_number = 1000
    start = timer()
    for _ in range(loops_number):
        generate_collocation_feature_2(**parameters)
    end = timer()
    print("average time execution of function generate_collocation_feature_2:\t%.12f [s]" % ((end - start) / loops_number))

    loops_number = 1000
    start = timer()
    for _ in range(loops_number):
        generate_collocation_feature_3(**parameters)
    end = timer()
    print("average time execution of function generate_collocation_feature_3:\t%.12f [s]" % ((end - start) / loops_number))


def generate_collocation_feature_and_write_1(output_file="generate_collocation_feature_and_write_1.txt", area=1000, cell_size=5, n_colloc=3, lambda_1=5, lambda_2=100):
    np.random.seed(0)
    f = open(file=output_file, mode="w")
    base_collocation_length_array = np.random.poisson(lam=lambda_1, size=n_colloc)
    base_collocation_length_array[base_collocation_length_array < 2] = 2
    # print("base_collocation_length_array=%s" % str(base_collocation_length_array))
    collocation_instances_number_array = np.random.poisson(lam=lambda_2, size=n_colloc)
    # print("collocation_instances_number_array=%s" % str(collocation_instances_number_array))

    collocation_features_number = np.sum(base_collocation_length_array)
    # print("collocation_features_number=%d" % collocation_features_number)

    last_colloc_id = 0
    area_in_cell_dim = area // cell_size
    for i_colloc in range(n_colloc):
        collocation_feature_ids = np.arange(last_colloc_id, last_colloc_id + base_collocation_length_array[i_colloc])
        # print(collocation_feature_ids)

        number_of_all_features_instances_of_collocation = collocation_instances_number_array[i_colloc] * base_collocation_length_array[i_colloc]

        instance_x = np.random.randint(low=area_in_cell_dim, size=collocation_instances_number_array[i_colloc])
        instance_x *= cell_size
        instance_x = instance_x.astype(dtype='float64')
        instance_x = np.repeat(a=instance_x, repeats=base_collocation_length_array[i_colloc])
        instance_x += np.random.uniform(high=cell_size, size=number_of_all_features_instances_of_collocation)

        instance_y = np.random.randint(low=area_in_cell_dim, size=collocation_instances_number_array[i_colloc])
        instance_y *= cell_size
        instance_y = instance_y.astype(dtype='float64')
        instance_y = np.repeat(a=instance_y, repeats=base_collocation_length_array[i_colloc])
        instance_y += np.random.uniform(high=cell_size, size=number_of_all_features_instances_of_collocation)

        feature_id = np.tile(A=collocation_feature_ids, reps=collocation_instances_number_array[i_colloc])

        feature_instance_id = np.arange(collocation_instances_number_array[i_colloc])
        feature_instance_id = np.repeat(a=feature_instance_id, repeats=base_collocation_length_array[i_colloc])

        for i in range(number_of_all_features_instances_of_collocation):
            f.write("%d %d %.6f %.6f\n" % (feature_id[i], feature_instance_id[i], instance_x[i], instance_y[i]))

        last_colloc_id += base_collocation_length_array[i_colloc]

    f.close()


def generate_collocation_feature_and_write_2(output_file="generate_collocation_feature_and_write_2.txt", area=1000, cell_size=5, n_colloc=3, lambda_1=5, lambda_2=100):
    np.random.seed(0)
    f = open(file=output_file, mode="w")
    base_collocation_length_array = np.random.poisson(lam=lambda_1, size=n_colloc)
    base_collocation_length_array[base_collocation_length_array < 2] = 2
    # print("base_collocation_length_array=%s" % str(base_collocation_length_array))
    collocation_instances_number_array = np.random.poisson(lam=lambda_2, size=n_colloc)
    # print("collocation_instances_number_array=%s" % str(collocation_instances_number_array))

    collocation_features_number = np.sum(base_collocation_length_array)
    # print("collocation_features_number=%d" % collocation_features_number)

    last_colloc_id = 0
    area_in_cell_dim = area // cell_size
    for i_colloc in range(n_colloc):
        collocation_feature_ids = np.arange(last_colloc_id, last_colloc_id + base_collocation_length_array[i_colloc])
        # print(collocation_feature_ids)

        number_of_all_features_instances_of_collocation = collocation_instances_number_array[i_colloc] * base_collocation_length_array[i_colloc]

        instance_x = np.random.randint(low=area_in_cell_dim, size=collocation_instances_number_array[i_colloc])
        instance_x *= cell_size
        instance_x = instance_x.astype(dtype='float64')
        instance_x = np.repeat(a=instance_x, repeats=base_collocation_length_array[i_colloc])
        instance_x += np.random.uniform(high=cell_size, size=number_of_all_features_instances_of_collocation)

        instance_y = np.random.randint(low=area_in_cell_dim, size=collocation_instances_number_array[i_colloc])
        instance_y *= cell_size
        instance_y = instance_y.astype(dtype='float64')
        instance_y = np.repeat(a=instance_y, repeats=base_collocation_length_array[i_colloc])
        instance_y += np.random.uniform(high=cell_size, size=number_of_all_features_instances_of_collocation)

        for i in range(number_of_all_features_instances_of_collocation):
            f.write("%d %d %.6f %.6f\n" % (last_colloc_id + i % base_collocation_length_array[i_colloc], i // base_collocation_length_array[i_colloc], instance_x[i], instance_y[i]))

        last_colloc_id += base_collocation_length_array[i_colloc]

    f.close()


def generate_collocation_feature_and_write_3(output_file="generate_collocation_feature_and_write_3.txt", area=1000, cell_size=5, n_colloc=3, lambda_1=5, lambda_2=100):
    np.random.seed(0)
    f = open(file=output_file, mode="w")
    base_collocation_length_array = np.random.poisson(lam=lambda_1, size=n_colloc)
    base_collocation_length_array[base_collocation_length_array < 2] = 2
    # print("base_collocation_length_array=%s" % str(base_collocation_length_array))
    collocation_instances_number_array = np.random.poisson(lam=lambda_2, size=n_colloc)
    # print("collocation_instances_number_array=%s" % str(collocation_instances_number_array))

    collocation_features_number = np.sum(base_collocation_length_array)
    # print("collocation_features_number=%d" % collocation_features_number)

    last_colloc_id = 0
    area_in_cell_dim = area // cell_size
    for i_colloc in range(n_colloc):
        collocation_feature_ids = np.arange(last_colloc_id, last_colloc_id + base_collocation_length_array[i_colloc])
        # print(collocation_feature_ids)

        number_of_all_features_instances_of_collocation = collocation_instances_number_array[i_colloc] * base_collocation_length_array[i_colloc]

        instance_x = np.random.randint(low=area_in_cell_dim, size=collocation_instances_number_array[i_colloc])
        instance_x *= cell_size
        instance_x = instance_x.astype(dtype='float64')
        instance_x = np.repeat(a=instance_x, repeats=base_collocation_length_array[i_colloc])
        instance_x += np.random.uniform(high=cell_size, size=number_of_all_features_instances_of_collocation)

        instance_y = np.random.randint(low=area_in_cell_dim, size=collocation_instances_number_array[i_colloc])
        instance_y *= cell_size
        instance_y = instance_y.astype(dtype='float64')
        instance_y = np.repeat(a=instance_y, repeats=base_collocation_length_array[i_colloc])
        instance_y += np.random.uniform(high=cell_size, size=number_of_all_features_instances_of_collocation)

        feature_id = np.tile(A=collocation_feature_ids, reps=collocation_instances_number_array[i_colloc])

        feature_instance_id = np.arange(collocation_instances_number_array[i_colloc])
        feature_instance_id = np.repeat(a=feature_instance_id, repeats=base_collocation_length_array[i_colloc])

        np.savetxt(fname=f, X=np.column_stack(tup=(feature_id, feature_instance_id, instance_x, instance_y)), fmt=['%d', '%d', '%.6f', '%.6f'])

        last_colloc_id += base_collocation_length_array[i_colloc]

    f.close()


def generate_collocation_feature_and_write_4(output_file="generate_collocation_feature_and_write_4.txt", area=1000, cell_size=5, n_colloc=3, lambda_1=5, lambda_2=100):
    np.random.seed(0)
    f = open(file=output_file, mode="w")
    base_collocation_length_array = np.random.poisson(lam=lambda_1, size=n_colloc)
    base_collocation_length_array[base_collocation_length_array < 2] = 2
    # print("base_collocation_length_array=%s" % str(base_collocation_length_array))
    collocation_instances_number_array = np.random.poisson(lam=lambda_2, size=n_colloc)
    # print("collocation_instances_number_array=%s" % str(collocation_instances_number_array))

    collocation_features_number = np.sum(base_collocation_length_array)
    # print("collocation_features_number=%d" % collocation_features_number)

    last_colloc_id = 0
    area_in_cell_dim = area // cell_size
    for i_colloc in range(n_colloc):
        collocation_feature_ids = np.arange(last_colloc_id, last_colloc_id + base_collocation_length_array[i_colloc])
        # print(collocation_feature_ids)

        number_of_all_features_instances_of_collocation = collocation_instances_number_array[i_colloc] * base_collocation_length_array[i_colloc]

        instance_x = np.random.randint(low=area_in_cell_dim, size=collocation_instances_number_array[i_colloc])
        instance_x *= cell_size
        instance_x = instance_x.astype(dtype='float64')
        instance_x = np.repeat(a=instance_x, repeats=base_collocation_length_array[i_colloc])
        instance_x += np.random.uniform(high=cell_size, size=number_of_all_features_instances_of_collocation)

        instance_y = np.random.randint(low=area_in_cell_dim, size=collocation_instances_number_array[i_colloc])
        instance_y *= cell_size
        instance_y = instance_y.astype(dtype='float64')
        instance_y = np.repeat(a=instance_y, repeats=base_collocation_length_array[i_colloc])
        instance_y += np.random.uniform(high=cell_size, size=number_of_all_features_instances_of_collocation)

        feature_id = np.tile(A=collocation_feature_ids, reps=collocation_instances_number_array[i_colloc])

        feature_instance_id = np.arange(collocation_instances_number_array[i_colloc])
        feature_instance_id = np.repeat(a=feature_instance_id, repeats=base_collocation_length_array[i_colloc])

        fmt = '%d %d %.6f %.6f\n' * number_of_all_features_instances_of_collocation
        data = fmt % tuple(np.column_stack(tup=(feature_id, feature_instance_id, instance_x, instance_y)).ravel())
        f.write(data)

        last_colloc_id += base_collocation_length_array[i_colloc]

    f.close()


def test_generate_collocation_feature_and_write():
    print("test_generate_collocation_feature_and_write execute")
    parameters = {
        "area": 1000,
        "cell_size": 5,
        "n_colloc": 10,
        "lambda_1": 5,
        "lambda_2": 1000
    }

    loops_number = 10
    start = timer()
    for _ in range(loops_number):
        generate_collocation_feature_and_write_1(**parameters)
    end = timer()
    print("average time execution of function generate_collocation_feature_and_write_1:\t%.12f [s]" % ((end - start) / loops_number))

    loops_number = 10
    start = timer()
    for _ in range(loops_number):
        generate_collocation_feature_and_write_2(**parameters)
    end = timer()
    print("average time execution of function generate_collocation_feature_and_write_2:\t%.12f [s]" % ((end - start) / loops_number))

    loops_number = 10
    start = timer()
    for _ in range(loops_number):
        generate_collocation_feature_and_write_3(**parameters)
    end = timer()
    print("average time execution of function generate_collocation_feature_and_write_3:\t%.12f [s]" % ((end - start) / loops_number))

    loops_number = 10
    start = timer()
    for _ in range(loops_number):
        generate_collocation_feature_and_write_4(**parameters)
    end = timer()
    print("average time execution of function generate_collocation_feature_and_write_4:\t%.12f [s]" % ((end - start) / loops_number))


def main():
    # test_generate_collocation_feature()
    test_generate_collocation_feature_and_write()


if __name__ == "__main__":
    main()
