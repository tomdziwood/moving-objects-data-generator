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
        instance_x = instance_x.astype(dtype=np.float64)
        instance_x = np.repeat(a=instance_x, repeats=base_collocation_length_array[i_colloc])
        instance_x += np.random.uniform(high=cell_size, size=number_of_all_features_instances_of_collocation)

        instance_y = np.random.randint(low=area_in_cell_dim, size=collocation_instances_number_array[i_colloc])
        instance_y *= cell_size
        instance_y = instance_y.astype(dtype=np.float64)
        instance_y = np.repeat(a=instance_y, repeats=base_collocation_length_array[i_colloc])
        instance_y += np.random.uniform(high=cell_size, size=number_of_all_features_instances_of_collocation)

        feature_id = np.tile(A=collocation_feature_ids, reps=collocation_instances_number_array[i_colloc])

        feature_instance_id = np.arange(collocation_instances_number_array[i_colloc])
        feature_instance_id = np.repeat(a=feature_instance_id, repeats=base_collocation_length_array[i_colloc])

        last_colloc_id += base_collocation_length_array[i_colloc]


def test_generate_collocation_feature():
    # test_generate_collocation_feature execute
    # average time execution of function generate_collocation_feature_1:	0.130188207000 [s]
    # average time execution of function generate_collocation_feature_2:	0.017090251000 [s]
    # average time execution of function generate_collocation_feature_3:	0.010674810600 [s]

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
    # test_generate_collocation_feature_and_write execute
    # average time execution of function generate_collocation_feature_and_write_1:	0.455495360000 [s]
    # average time execution of function generate_collocation_feature_and_write_2:	0.528807950000 [s]
    # average time execution of function generate_collocation_feature_and_write_3:	0.687456600000 [s]
    # average time execution of function generate_collocation_feature_and_write_4:	0.239493560000 [s]

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


def generate_collocation_noise_feature_1(area, collocation_noise_feature_number, collocation_noise_feature, collocation_noise_feature_instances_number_array, collocation_features_instances_number_array):

    for i_feature in range(collocation_noise_feature_number):
        feature = collocation_noise_feature[i_feature]
        collocation_noise_feature_instances_number = collocation_noise_feature_instances_number_array[i_feature]
        # print("collocation_noise_feature_instances_number=%d" % collocation_noise_feature_instances_number)
        feature_instance = collocation_features_instances_number_array[feature]
        for _ in range(collocation_noise_feature_instances_number):
            instance_x = random.random() * area
            instance_y = random.random() * area
            feature_instance += 1


def generate_collocation_noise_feature_2(area, collocation_noise_feature_number, collocation_noise_feature, collocation_noise_feature_instances_number_array, collocation_features_instances_number_array):

    number_of_all_noise_features_instances = np.sum(collocation_noise_feature_instances_number_array)
    instance_x = np.random.uniform(high=area, size=number_of_all_noise_features_instances)
    instance_y = np.random.uniform(high=area, size=number_of_all_noise_features_instances)
    feature_id = np.repeat(a=collocation_noise_feature, repeats=collocation_noise_feature_instances_number_array)

    feature_instance_id = []
    for i_feature in range(collocation_noise_feature_number):
        feature = collocation_noise_feature[i_feature]
        feature_instance_id.append(np.arange(
            start=collocation_features_instances_number_array[feature],
            stop=collocation_features_instances_number_array[feature] + collocation_noise_feature_instances_number_array[i_feature]
        ))
    feature_instance_id = np.concatenate(feature_instance_id)


def generate_collocation_noise_feature_3(area, collocation_noise_feature_number, collocation_noise_feature, collocation_noise_feature_instances_number_array, collocation_features_instances_number_array):

    number_of_all_noise_features_instances = np.sum(collocation_noise_feature_instances_number_array)
    instance_x = np.random.uniform(high=area, size=number_of_all_noise_features_instances)
    instance_y = np.random.uniform(high=area, size=number_of_all_noise_features_instances)
    feature_id = np.repeat(a=collocation_noise_feature, repeats=collocation_noise_feature_instances_number_array)

    start = collocation_features_instances_number_array[collocation_noise_feature]
    length = collocation_noise_feature_instances_number_array
    feature_instance_id = np.repeat(start + length - length.cumsum(), length) + np.arange(number_of_all_noise_features_instances)


def test_generate_collocation_noise_feature():
    # average time execution of function generate_collocation_noise_feature_1:	0.131666840000 [s]
    # average time execution of function generate_collocation_noise_feature_2:	0.009112349000 [s]
    # average time execution of function generate_collocation_noise_feature_3:	0.010143799000 [s]

    print("test_generate_collocation_noise_feature execute")

    # starting parameters
    area = 1000
    n_colloc = 10
    lambda_1 = 5
    lambda_2 = 1000
    ncfr = 0.98
    ncfn = 1.5

    # # starting parameters
    # area = 1000
    # n_colloc = 100
    # lambda_1 = 50
    # lambda_2 = 100
    # ncfr = 0.98
    # ncfn = 1.5

    base_collocation_length_array = np.random.poisson(lam=lambda_1, size=n_colloc)
    base_collocation_length_array[base_collocation_length_array < 2] = 2
    # print("base_collocation_length_array=%s" % str(base_collocation_length_array))
    collocation_instances_number_array = np.random.poisson(lam=lambda_2, size=n_colloc)
    # print("collocation_instances_number_array=%s" % str(collocation_instances_number_array))

    collocation_features_number = np.sum(base_collocation_length_array)
    # print("collocation_features_number=%d" % collocation_features_number)

    collocation_noise_feature_number = round(ncfr * collocation_features_number)
    # print("collocation_noise_feature_number=%d" % collocation_noise_feature_number)

    # collocation_noise_feature = random.sample(population=range(collocation_features_number), k=collocation_noise_feature_number)
    collocation_noise_feature = np.random.choice(a=collocation_features_number, size=collocation_noise_feature_number, replace=False)
    collocation_noise_feature.sort()
    # print("collocation_noise_feature=%s" % str(collocation_noise_feature))

    collocation_features_instances_number_array = np.repeat(a=collocation_instances_number_array, repeats=base_collocation_length_array)
    # print("collocation_features_instances_number_array=%s" % str(collocation_features_instances_number_array))

    collocation_noise_feature_instances_number_array = ncfn * collocation_features_instances_number_array[collocation_noise_feature]
    collocation_noise_feature_instances_number_array = collocation_noise_feature_instances_number_array.astype(np.int32)
    # print("collocation_noise_feature_instances_number_array=%s" % str(collocation_noise_feature_instances_number_array))

    print("data is ready")

    # writing parameters
    parameters = {
        "area": area,
        "collocation_noise_feature_number": collocation_noise_feature_number,
        "collocation_noise_feature": collocation_noise_feature,
        "collocation_noise_feature_instances_number_array": collocation_noise_feature_instances_number_array,
        "collocation_features_instances_number_array": collocation_features_instances_number_array
    }

    loops_number = 10
    start = timer()
    for _ in range(loops_number):
        generate_collocation_noise_feature_1(**parameters)
    end = timer()
    print("average time execution of function generate_collocation_noise_feature_1:\t%.12f [s]" % ((end - start) / loops_number))

    loops_number = 100
    start = timer()
    for _ in range(loops_number):
        generate_collocation_noise_feature_2(**parameters)
    end = timer()
    print("average time execution of function generate_collocation_noise_feature_2:\t%.12f [s]" % ((end - start) / loops_number))

    loops_number = 100
    start = timer()
    for _ in range(loops_number):
        generate_collocation_noise_feature_3(**parameters)
    end = timer()
    print("average time execution of function generate_collocation_noise_feature_3:\t%.12f [s]" % ((end - start) / loops_number))


def write_collocation_noise_feature_1(output_file="write_collocation_noise_feature_1.txt", feature_id=None, feature_instance_id=None, instance_x=None, instance_y=None, number_of_all_noise_features_instances=None):
    f = open(file=output_file, mode="w")

    for i in range(number_of_all_noise_features_instances):
        f.write("%d %d %.6f %.6f\n" % (feature_id[i], feature_instance_id[i], instance_x[i], instance_y[i]))

    f.close()


def write_collocation_noise_feature_2(output_file="write_collocation_noise_feature_2.txt", feature_id=None, feature_instance_id=None, instance_x=None, instance_y=None, number_of_all_noise_features_instances=None):
    f = open(file=output_file, mode="w")

    np.savetxt(fname=f, X=np.column_stack(tup=(feature_id, feature_instance_id, instance_x, instance_y)), fmt=['%d', '%d', '%.6f', '%.6f'])

    f.close()


def write_collocation_noise_feature_3(output_file="write_collocation_noise_feature_3.txt", feature_id=None, feature_instance_id=None, instance_x=None, instance_y=None, number_of_all_noise_features_instances=None):
    f = open(file=output_file, mode="w")

    fmt = '%d %d %.6f %.6f\n' * number_of_all_noise_features_instances
    data = fmt % tuple(np.column_stack(tup=(feature_id, feature_instance_id, instance_x, instance_y)).ravel())
    f.write(data)

    f.close()


def test_write_collocation_noise_feature():
    # average time execution of function write_collocation_noise_feature_1:	0.850331040000 [s]
    # average time execution of function write_collocation_noise_feature_2:	1.243028610000 [s]
    # average time execution of function write_collocation_noise_feature_3:	0.494746820000 [s]

    print("test_generate_collocation_noise_feature_and_write execute")

    # starting parameters
    area = 1000
    n_colloc = 10
    lambda_1 = 5
    lambda_2 = 1000
    ncfr = 0.98
    ncfn = 1.5

    # # starting parameters
    # area = 1000
    # n_colloc = 100
    # lambda_1 = 50
    # lambda_2 = 100
    # ncfr = 0.98
    # ncfn = 1.5

    np.random.seed(0)
    base_collocation_length_array = np.random.poisson(lam=lambda_1, size=n_colloc)
    base_collocation_length_array[base_collocation_length_array < 2] = 2
    # print("base_collocation_length_array=%s" % str(base_collocation_length_array))
    collocation_instances_number_array = np.random.poisson(lam=lambda_2, size=n_colloc)
    # print("collocation_instances_number_array=%s" % str(collocation_instances_number_array))

    collocation_features_number = np.sum(base_collocation_length_array)
    # print("collocation_features_number=%d" % collocation_features_number)

    collocation_noise_feature_number = round(ncfr * collocation_features_number)
    # print("collocation_noise_feature_number=%d" % collocation_noise_feature_number)

    collocation_noise_feature = np.random.choice(a=collocation_features_number, size=collocation_noise_feature_number, replace=False)
    collocation_noise_feature.sort()
    # print("collocation_noise_feature=%s" % str(collocation_noise_feature))

    collocation_features_instances_number_array = np.repeat(a=collocation_instances_number_array, repeats=base_collocation_length_array)
    # print("collocation_features_instances_number_array=%s" % str(collocation_features_instances_number_array))

    collocation_noise_feature_instances_number_array = ncfn * collocation_features_instances_number_array[collocation_noise_feature]
    collocation_noise_feature_instances_number_array = collocation_noise_feature_instances_number_array.astype(np.int32)
    # print("collocation_noise_feature_instances_number_array=%s" % str(collocation_noise_feature_instances_number_array))

    number_of_all_noise_features_instances = np.sum(collocation_noise_feature_instances_number_array)
    instance_x = np.random.uniform(high=area, size=number_of_all_noise_features_instances)
    instance_y = np.random.uniform(high=area, size=number_of_all_noise_features_instances)
    feature_id = np.repeat(a=collocation_noise_feature, repeats=collocation_noise_feature_instances_number_array)

    start = collocation_features_instances_number_array[collocation_noise_feature]
    length = collocation_noise_feature_instances_number_array
    feature_instance_id = np.repeat(start + length - length.cumsum(), length) + np.arange(number_of_all_noise_features_instances)

    print("data is ready")

    # writing parameters
    parameters = {
        "feature_id": feature_id,
        "feature_instance_id": feature_instance_id,
        "instance_x": instance_x,
        "instance_y": instance_y,
        "number_of_all_noise_features_instances": number_of_all_noise_features_instances
    }

    loops_number = 10
    start = timer()
    for _ in range(loops_number):
        write_collocation_noise_feature_1(**parameters)
    end = timer()
    print("average time execution of function write_collocation_noise_feature_1:\t%.12f [s]" % ((end - start) / loops_number))

    loops_number = 10
    start = timer()
    for _ in range(loops_number):
        write_collocation_noise_feature_2(**parameters)
    end = timer()
    print("average time execution of function write_collocation_noise_feature_2:\t%.12f [s]" % ((end - start) / loops_number))

    loops_number = 10
    start = timer()
    for _ in range(loops_number):
        write_collocation_noise_feature_3(**parameters)
    end = timer()
    print("average time execution of function write_collocation_noise_feature_3:\t%.12f [s]" % ((end - start) / loops_number))


def generate_additional_noise_features_1(area, last_colloc_id, ndf, ndfn):
    additional_noise_feature_instance = np.zeros(shape=ndf, dtype=np.int32)
    for _ in range(ndfn):
        additional_noise_feature = np.random.randint(low=ndf)
        instance_x = np.random.uniform(high=area)
        instance_y = np.random.uniform(high=area)
        # print("additional noise %d inst %d \t coor:\t(%f, %f)" % (last_colloc_id + additional_noise_feature, additional_noise_feature_instance[additional_noise_feature], instance_x, instance_y))
        # f.write("%d %d %f %f\n" % (last_colloc_id + additional_noise_feature, additional_noise_feature_instance[additional_noise_feature], instance_x, instance_y))
        additional_noise_feature_instance[additional_noise_feature] += 1


def generate_additional_noise_features_2(area, last_colloc_id, ndf, ndfn):
    additional_noise_feature = np.random.randint(low=ndf, size=ndfn) + last_colloc_id
    (additional_noise_feature_unique, additional_noise_feature_counts) = np.unique(ar=additional_noise_feature, return_counts=True)
    additional_noise_feature_instance_id = np.repeat(a=(additional_noise_feature_counts - additional_noise_feature_counts.cumsum()), repeats=additional_noise_feature_counts) + np.arange(ndfn)
    additional_noise_feature_id = np.repeat(a=additional_noise_feature_unique, repeats=additional_noise_feature_counts)
    instance_x = np.random.uniform(high=area, size=ndfn)
    instance_y = np.random.uniform(high=area, size=ndfn)



def test_generate_additional_noise_feature():
    # average time execution of function generate_additional_noise_features_1:	0.224038370000 [s]
    # average time execution of function generate_additional_noise_features_2:	0.001528814300 [s]

    print("test_generate_additional_noise_feature execute")

    # generating parameters
    parameters = {
        "area": 1000,
        "last_colloc_id": 13,
        "ndf": 5,
        "ndfn": 5000,
    }

    loops_number = 10
    start = timer()
    for _ in range(loops_number):
        generate_additional_noise_features_1(**parameters)
    end = timer()
    print("average time execution of function generate_additional_noise_features_1:\t%.12f [s]" % ((end - start) / loops_number))

    loops_number = 1000
    start = timer()
    for _ in range(loops_number):
        generate_additional_noise_features_2(**parameters)
    end = timer()
    print("average time execution of function generate_additional_noise_features_2:\t%.12f [s]" % ((end - start) / loops_number))


def main():
    # test_generate_collocation_feature()
    # test_generate_collocation_feature_and_write()
    # test_generate_collocation_noise_feature()
    # test_write_collocation_noise_feature()
    test_generate_additional_noise_feature()


if __name__ == "__main__":
    main()
