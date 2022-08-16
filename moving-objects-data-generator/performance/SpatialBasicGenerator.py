import random
import sys
import numpy as np

from timeit import default_timer as timer

from algorithms.initiation.BasicInitiation import BasicInitiation
from algorithms.parameters.BasicParameters import BasicParameters
from scripts.iterative import SpatialBasicGenerator as isbg
from scripts.vectorized import SpatialBasicGenerator as vsbg


def generate_collocation_feature_1(area, cell_size, n_base, lambda_1, lambda_2, m_clumpy, m_overlap):
    np.random.seed(0)
    base_collocation_length_array = np.random.poisson(lam=lambda_1, size=n_base)
    base_collocation_length_array[base_collocation_length_array < 2] = 2
    # print("base_collocation_length_array=%s" % str(base_collocation_length_array))
    collocation_instances_number_array = np.random.poisson(lam=lambda_2, size=n_base)
    # print("collocation_instances_number_array=%s" % str(collocation_instances_number_array))

    collocation_features_number = np.sum(base_collocation_length_array)
    # print("collocation_features_number=%d" % collocation_features_number)

    last_colloc_id = 0
    area_in_cell_dim = area // cell_size
    for i_colloc in range(n_base):
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


def generate_collocation_feature_2(area, cell_size, n_base, lambda_1, lambda_2, m_clumpy, m_overlap):
    np.random.seed(0)
    base_collocation_length_array = np.random.poisson(lam=lambda_1, size=n_base)
    base_collocation_length_array[base_collocation_length_array < 2] = 2
    # print("base_collocation_length_array=%s" % str(base_collocation_length_array))
    collocation_instances_number_array = np.random.poisson(lam=lambda_2, size=n_base)
    # print("collocation_instances_number_array=%s" % str(collocation_instances_number_array))

    collocation_features_number = np.sum(base_collocation_length_array)
    # print("collocation_features_number=%d" % collocation_features_number)

    last_colloc_id = 0
    area_in_cell_dim = area // cell_size
    for i_colloc in range(n_base):
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


def generate_collocation_feature_3(area, cell_size, n_base, lambda_1, lambda_2, m_clumpy, m_overlap):
    np.random.seed(0)
    base_collocation_length_array = np.random.poisson(lam=lambda_1, size=n_base)
    base_collocation_length_array[base_collocation_length_array < 2] = 2
    # print("base_collocation_length_array=%s" % str(base_collocation_length_array))
    collocation_instances_number_array = np.random.poisson(lam=lambda_2, size=n_base)
    # print("collocation_instances_number_array=%s" % str(collocation_instances_number_array))

    collocation_features_number = np.sum(base_collocation_length_array)
    # print("collocation_features_number=%d" % collocation_features_number)

    last_colloc_id = 0
    area_in_cell_dim = area // cell_size
    for i_colloc in range(n_base):
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


def generate_collocation_feature_4(area, cell_size, n_base, lambda_1, lambda_2, m_clumpy, m_overlap):
    np.random.seed(0)
    base_collocation_lengths = np.random.poisson(lam=lambda_1, size=n_base)
    base_collocation_lengths[base_collocation_lengths < 2] = 2
    # print("base_collocation_lengths=%s" % str(base_collocation_lengths))
    collocation_instances_counts = np.random.poisson(lam=lambda_2, size=n_base)
    # print("collocation_instances_counts=%s" % str(collocation_instances_counts))

    collocation_features_sum = np.sum(base_collocation_lengths)
    # print("collocation_features_sum=%d" % collocation_features_sum)

    last_colloc_id = 0
    area_in_cell_dim = area // cell_size
    for i_colloc in range(n_base):
        collocation_features = np.arange(last_colloc_id, last_colloc_id + base_collocation_lengths[i_colloc])
        # print(collocation_features)

        collocation_feature_instance_id = 0
        while collocation_feature_instance_id < collocation_instances_counts[i_colloc]:
            cell_x_id = random.randint(0, area_in_cell_dim)
            cell_y_id = random.randint(0, area_in_cell_dim)
            # print("ids:\t(%d, %d)" % (cell_x_id, cell_y_id))

            cell_x = cell_x_id * cell_size
            cell_y = cell_y_id * cell_size
            # print("cell coor:\t(%d, %d)" % (cell_x, cell_y))

            m_clumpy_repeats = min(m_clumpy, collocation_instances_counts[i_colloc] - collocation_feature_instance_id)
            for _ in range(m_clumpy_repeats):
                for i_feature in range(base_collocation_lengths[i_colloc]):
                    instance_x = cell_x + random.random() * cell_size
                    instance_y = cell_y + random.random() * cell_size
                    # print("i_feature: %d\tinst coor:\t(%f, %f)" % (collocation_features[i_feature], instance_x, instance_y))

                collocation_feature_instance_id += 1

        last_colloc_id += base_collocation_lengths[i_colloc]


def generate_collocation_feature_5(area, cell_size, n_base, lambda_1, lambda_2, m_clumpy, m_overlap):
    np.random.seed(0)
    base_collocation_lengths = np.random.poisson(lam=lambda_1, size=n_base)
    base_collocation_lengths[base_collocation_lengths < 2] = 2
    # print("base_collocation_lengths=%s" % str(base_collocation_lengths))
    collocations_instances_counts = np.random.poisson(lam=lambda_2, size=n_base)
    # print("collocations_instances_counts=%s" % str(collocations_instances_counts))

    collocation_features_sum = np.sum(base_collocation_lengths)
    # print("collocation_features_sum=%d" % collocation_features_sum)

    last_colloc_id = 0
    area_in_cell_dim = area // cell_size
    # print("area_in_cell_dim: ", area_in_cell_dim)
    for i_colloc in range(n_base):
        collocation_features = np.arange(last_colloc_id, last_colloc_id + base_collocation_lengths[i_colloc])
        # print(collocation_features)

        collocation_features_instances_sum = collocations_instances_counts[i_colloc] * base_collocation_lengths[i_colloc]

        collocation_features_instances_x = np.random.randint(low=area_in_cell_dim, size=(collocations_instances_counts[i_colloc] - 1) // m_clumpy + 1)
        collocation_features_instances_x *= cell_size
        collocation_features_instances_x = collocation_features_instances_x.astype(dtype=np.float64)
        collocation_features_instances_x = np.repeat(a=collocation_features_instances_x, repeats=m_clumpy)[:collocations_instances_counts[i_colloc]]
        collocation_features_instances_x = np.repeat(a=collocation_features_instances_x, repeats=base_collocation_lengths[i_colloc])
        collocation_features_instances_x += np.random.uniform(high=cell_size, size=collocation_features_instances_sum)

        collocation_features_instances_y = np.random.randint(low=area_in_cell_dim, size=(collocations_instances_counts[i_colloc] - 1) // m_clumpy + 1)
        collocation_features_instances_y *= cell_size
        collocation_features_instances_y = collocation_features_instances_y.astype(dtype=np.float64)
        collocation_features_instances_y = np.repeat(a=collocation_features_instances_y, repeats=m_clumpy)[:collocations_instances_counts[i_colloc]]
        collocation_features_instances_y = np.repeat(a=collocation_features_instances_y, repeats=base_collocation_lengths[i_colloc])
        collocation_features_instances_y += np.random.uniform(high=cell_size, size=collocation_features_instances_sum)

        collocation_features_ids = np.tile(A=collocation_features, reps=collocations_instances_counts[i_colloc])

        collocation_features_instances_ids = np.arange(collocations_instances_counts[i_colloc])
        collocation_features_instances_ids = np.repeat(a=collocation_features_instances_ids, repeats=base_collocation_lengths[i_colloc])

        last_colloc_id += base_collocation_lengths[i_colloc]


def generate_collocation_feature_6(area, cell_size, n_base, lambda_1, lambda_2, m_clumpy, m_overlap):
    np.random.seed(0)

    base_collocation_lengths = np.random.poisson(lam=lambda_1, size=n_base)
    base_collocation_lengths[base_collocation_lengths < 2] = 2
    # print("base_collocation_lengths=%s" % str(base_collocation_lengths))

    if m_overlap > 1:
        collocation_lengths = np.repeat(a=base_collocation_lengths + 1, repeats=m_overlap)
    else:
        collocation_lengths = base_collocation_lengths
    # print("collocation_lengths=%s" % str(collocation_lengths))

    collocation_instances_counts = np.random.poisson(lam=lambda_2, size=n_base * m_overlap)
    # print("collocation_instances_counts=%s" % str(collocation_instances_counts))

    collocation_features_sum = np.sum(base_collocation_lengths)
    if m_overlap > 1:
        collocation_features_sum += n_base * m_overlap
    # print("collocation_features_sum=%d" % collocation_features_sum)

    # collocation_features_instances_counts = np.repeat(a=collocation_instances_counts, repeats=base_collocation_lengths)
    collocation_features_instances_counts = np.zeros(shape=collocation_features_sum, dtype=np.int32)
    # print("collocation_features_instances_counts=%s" % str(collocation_features_instances_counts))

    collocation_start_feature_id = 0
    area_in_cell_dim = area // cell_size
    # print("area_in_cell_dim: ", area_in_cell_dim)
    for i_colloc in range(n_base * m_overlap):
        collocation_features = np.arange(collocation_start_feature_id, collocation_start_feature_id + collocation_lengths[i_colloc])
        collocation_features[-1] += i_colloc % m_overlap
        # print("collocation_features=%s" % str(collocation_features))

        collocation_instance_id = 0
        while collocation_instance_id < collocation_instances_counts[i_colloc]:
            cell_x_id = np.random.randint(low=area_in_cell_dim)
            cell_y_id = np.random.randint(low=area_in_cell_dim)
            # print("ids:\t(%d, %d)" % (cell_x_id, cell_y_id))

            cell_x = cell_x_id * cell_size
            cell_y = cell_y_id * cell_size
            # print("cell coor:\t(%d, %d)" % (cell_x, cell_y))

            m_clumpy_repeats = min(m_clumpy, collocation_instances_counts[i_colloc] - collocation_instance_id)
            for _ in range(m_clumpy_repeats):
                for collocation_feature in collocation_features:
                    instance_x = cell_x + np.random.uniform() * cell_size
                    instance_y = cell_y + np.random.uniform() * cell_size
                    # print("collocation_feature: %d\tinst coor:\t(%f, %f)" % (collocation_feature, instance_x, instance_y))
                    # f.write("%d %d %f %f\n" % (collocation_feature, collocation_features_instances_counts[collocation_feature], instance_x, instance_y))
                    collocation_features_instances_counts[collocation_feature] += 1

                collocation_instance_id += 1

        if (i_colloc + 1) % m_overlap == 0:
            collocation_start_feature_id += collocation_lengths[i_colloc] + m_overlap - 1


def generate_collocation_feature_7(area, cell_size, n_base, lambda_1, lambda_2, m_clumpy, m_overlap):
    np.random.seed(0)

    base_collocation_lengths = np.random.poisson(lam=lambda_1, size=n_base)
    base_collocation_lengths[base_collocation_lengths < 2] = 2
    # print("base_collocation_lengths=%s" % str(base_collocation_lengths))

    if m_overlap > 1:
        collocation_lengths = np.repeat(a=base_collocation_lengths + 1, repeats=m_overlap)
    else:
        collocation_lengths = base_collocation_lengths
    # print("collocation_lengths=%s" % str(collocation_lengths))

    collocations_instances_counts = np.random.poisson(lam=lambda_2, size=n_base * m_overlap)
    # print("collocations_instances_counts=%s" % str(collocations_instances_counts))

    collocation_features_sum = np.sum(base_collocation_lengths)
    if m_overlap > 1:
        collocation_features_sum += n_base * m_overlap
    # print("collocation_features_sum=%d" % collocation_features_sum)

    collocation_features_instances_counts = np.zeros(shape=collocation_features_sum, dtype=np.int32)
    # print("collocation_features_instances_counts=%s" % str(collocation_features_instances_counts))

    collocation_start_feature_id = 0
    area_in_cell_dim = area // cell_size
    # print("area_in_cell_dim: ", area_in_cell_dim)
    for i_colloc in range(n_base * m_overlap):
        collocation_features = np.arange(collocation_start_feature_id, collocation_start_feature_id + collocation_lengths[i_colloc])
        collocation_features[-1] += i_colloc % m_overlap
        # print("collocation_features=%s" % str(collocation_features))

        collocation_features_instances_sum = collocations_instances_counts[i_colloc] * collocation_lengths[i_colloc]

        collocation_features_instances_x = np.random.randint(low=area_in_cell_dim, size=(collocations_instances_counts[i_colloc] - 1) // m_clumpy + 1)
        collocation_features_instances_x *= cell_size
        collocation_features_instances_x = collocation_features_instances_x.astype(dtype=np.float64)
        collocation_features_instances_x = np.repeat(a=collocation_features_instances_x, repeats=m_clumpy)[:collocations_instances_counts[i_colloc]]
        collocation_features_instances_x = np.repeat(a=collocation_features_instances_x, repeats=collocation_lengths[i_colloc])
        collocation_features_instances_x += np.random.uniform(high=cell_size, size=collocation_features_instances_sum)

        collocation_features_instances_y = np.random.randint(low=area_in_cell_dim, size=(collocations_instances_counts[i_colloc] - 1) // m_clumpy + 1)
        collocation_features_instances_y *= cell_size
        collocation_features_instances_y = collocation_features_instances_y.astype(dtype=np.float64)
        collocation_features_instances_y = np.repeat(a=collocation_features_instances_y, repeats=m_clumpy)[:collocations_instances_counts[i_colloc]]
        collocation_features_instances_y = np.repeat(a=collocation_features_instances_y, repeats=collocation_lengths[i_colloc])
        collocation_features_instances_y += np.random.uniform(high=cell_size, size=collocation_features_instances_sum)

        collocation_features_ids = np.tile(A=collocation_features, reps=collocations_instances_counts[i_colloc])

        collocation_features_instances_ids = np.arange(
            start=collocation_features_instances_counts[collocation_start_feature_id],
            stop=collocation_features_instances_counts[collocation_start_feature_id] + collocations_instances_counts[i_colloc]
        )
        collocation_features_instances_ids = np.tile(A=collocation_features_instances_ids, reps=(collocation_lengths[i_colloc] - 1, 1))
        collocation_features_instances_ids = np.concatenate((
            collocation_features_instances_ids,
            np.arange(collocations_instances_counts[i_colloc]).reshape((1, collocations_instances_counts[i_colloc]))
        ))
        collocation_features_instances_ids = collocation_features_instances_ids.T.flatten()

        collocation_features_instances_counts[collocation_features] += collocations_instances_counts[i_colloc]

        if (i_colloc + 1) % m_overlap == 0:
            collocation_start_feature_id += collocation_lengths[i_colloc] + m_overlap - 1


def test_generate_collocation_feature():
    # test_generate_collocation_feature execute
    # average time execution of function generate_collocation_feature_1:	0.056731886000 [s]
    # average time execution of function generate_collocation_feature_2:	0.004833551100 [s]
    # average time execution of function generate_collocation_feature_3:	0.004710863100 [s]
    # average time execution of function generate_collocation_feature_4:	0.077386832000 [s]
    # average time execution of function generate_collocation_feature_5:	0.004883682000 [s]
    # average time execution of function generate_collocation_feature_6:	0.748450950000 [s]
    # average time execution of function generate_collocation_feature_7:	0.005639632000 [s]

    print("test_generate_collocation_feature execute")
    parameters = {
        "area": 1000,
        "cell_size": 5,
        "n_base": 10,
        "lambda_1": 5,
        "lambda_2": 1000,
        "m_clumpy": 1,
        "m_overlap": 1
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

    loops_number = 100
    start = timer()
    for _ in range(loops_number):
        generate_collocation_feature_4(**parameters)
    end = timer()
    print("average time execution of function generate_collocation_feature_4:\t%.12f [s]" % ((end - start) / loops_number))

    loops_number = 1000
    start = timer()
    for _ in range(loops_number):
        generate_collocation_feature_5(**parameters)
    end = timer()
    print("average time execution of function generate_collocation_feature_5:\t%.12f [s]" % ((end - start) / loops_number))

    loops_number = 10
    start = timer()
    for _ in range(loops_number):
        generate_collocation_feature_6(**parameters)
    end = timer()
    print("average time execution of function generate_collocation_feature_6:\t%.12f [s]" % ((end - start) / loops_number))

    loops_number = 1000
    start = timer()
    for _ in range(loops_number):
        generate_collocation_feature_7(**parameters)
    end = timer()
    print("average time execution of function generate_collocation_feature_7:\t%.12f [s]" % ((end - start) / loops_number))


def generate_collocation_feature_and_write_1(output_file="generate_collocation_feature_and_write_1.txt", area=1000, cell_size=5, n_base=3, lambda_1=5, lambda_2=100):
    np.random.seed(0)
    f = open(file=output_file, mode="w")
    base_collocation_length_array = np.random.poisson(lam=lambda_1, size=n_base)
    base_collocation_length_array[base_collocation_length_array < 2] = 2
    # print("base_collocation_length_array=%s" % str(base_collocation_length_array))
    collocation_instances_number_array = np.random.poisson(lam=lambda_2, size=n_base)
    # print("collocation_instances_number_array=%s" % str(collocation_instances_number_array))

    collocation_features_number = np.sum(base_collocation_length_array)
    # print("collocation_features_number=%d" % collocation_features_number)

    last_colloc_id = 0
    area_in_cell_dim = area // cell_size
    for i_colloc in range(n_base):
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


def generate_collocation_feature_and_write_2(output_file="generate_collocation_feature_and_write_2.txt", area=1000, cell_size=5, n_base=3, lambda_1=5, lambda_2=100):
    np.random.seed(0)
    f = open(file=output_file, mode="w")
    base_collocation_length_array = np.random.poisson(lam=lambda_1, size=n_base)
    base_collocation_length_array[base_collocation_length_array < 2] = 2
    # print("base_collocation_length_array=%s" % str(base_collocation_length_array))
    collocation_instances_number_array = np.random.poisson(lam=lambda_2, size=n_base)
    # print("collocation_instances_number_array=%s" % str(collocation_instances_number_array))

    collocation_features_number = np.sum(base_collocation_length_array)
    # print("collocation_features_number=%d" % collocation_features_number)

    last_colloc_id = 0
    area_in_cell_dim = area // cell_size
    for i_colloc in range(n_base):
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


def generate_collocation_feature_and_write_3(output_file="generate_collocation_feature_and_write_3.txt", area=1000, cell_size=5, n_base=3, lambda_1=5, lambda_2=100):
    np.random.seed(0)
    f = open(file=output_file, mode="w")
    base_collocation_length_array = np.random.poisson(lam=lambda_1, size=n_base)
    base_collocation_length_array[base_collocation_length_array < 2] = 2
    # print("base_collocation_length_array=%s" % str(base_collocation_length_array))
    collocation_instances_number_array = np.random.poisson(lam=lambda_2, size=n_base)
    # print("collocation_instances_number_array=%s" % str(collocation_instances_number_array))

    collocation_features_number = np.sum(base_collocation_length_array)
    # print("collocation_features_number=%d" % collocation_features_number)

    last_colloc_id = 0
    area_in_cell_dim = area // cell_size
    for i_colloc in range(n_base):
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


def generate_collocation_feature_and_write_4(output_file="generate_collocation_feature_and_write_4.txt", area=1000, cell_size=5, n_base=3, lambda_1=5, lambda_2=100):
    np.random.seed(0)
    f = open(file=output_file, mode="w")
    base_collocation_length_array = np.random.poisson(lam=lambda_1, size=n_base)
    base_collocation_length_array[base_collocation_length_array < 2] = 2
    # print("base_collocation_length_array=%s" % str(base_collocation_length_array))
    collocation_instances_number_array = np.random.poisson(lam=lambda_2, size=n_base)
    # print("collocation_instances_number_array=%s" % str(collocation_instances_number_array))

    collocation_features_number = np.sum(base_collocation_length_array)
    # print("collocation_features_number=%d" % collocation_features_number)

    last_colloc_id = 0
    area_in_cell_dim = area // cell_size
    for i_colloc in range(n_base):
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
        "n_base": 10,
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
    n_base = 10
    lambda_1 = 5
    lambda_2 = 1000
    ncfr = 0.98
    ncfn = 1.5

    # # starting parameters
    # area = 1000
    # n_base = 100
    # lambda_1 = 50
    # lambda_2 = 100
    # ncfr = 0.98
    # ncfn = 1.5

    base_collocation_length_array = np.random.poisson(lam=lambda_1, size=n_base)
    base_collocation_length_array[base_collocation_length_array < 2] = 2
    # print("base_collocation_length_array=%s" % str(base_collocation_length_array))
    collocation_instances_number_array = np.random.poisson(lam=lambda_2, size=n_base)
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
    n_base = 10
    lambda_1 = 5
    lambda_2 = 1000
    ncfr = 0.98
    ncfn = 1.5

    # # starting parameters
    # area = 1000
    # n_base = 100
    # lambda_1 = 50
    # lambda_2 = 100
    # ncfr = 0.98
    # ncfn = 1.5

    np.random.seed(0)
    base_collocation_length_array = np.random.poisson(lam=lambda_1, size=n_base)
    base_collocation_length_array[base_collocation_length_array < 2] = 2
    # print("base_collocation_length_array=%s" % str(base_collocation_length_array))
    collocation_instances_number_array = np.random.poisson(lam=lambda_2, size=n_base)
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


def test_spatial_basic_generator():
    # average time execution of function isbg.generate:	5.189230070000 [s]
    # average time execution of function vsbg.generate:	0.534279507000 [s]

    print("test_spatial_basic_generator execute")

    # generating parameters
    parameters = {
        "area": 1000,
        "cell_size": 5,
        "n_base": 10,
        "lambda_1": 5,
        "lambda_2": 1000,
        "m_clumpy": 2,
        "m_overlap": 3,
        "ncfr": 0.4,
        "ncfn": 0.8,
        "ndf": 5,
        "ndfn": 5000,
        "random_seed": 0
    }

    loops_number = 10
    save_stdout = sys.stdout
    sys.stdout = open('trash.txt', 'w')
    start = timer()
    for _ in range(loops_number):
        isbg.generate(**parameters)
    end = timer()
    sys.stdout = save_stdout
    print("average time execution of function isbg.generate:\t%.12f [s]" % ((end - start) / loops_number))

    loops_number = 100
    save_stdout = sys.stdout
    sys.stdout = open('trash.txt', 'w')
    start = timer()
    for _ in range(loops_number):
        vsbg.generate(**parameters)
    end = timer()
    sys.stdout = save_stdout
    print("average time execution of function vsbg.generate:\t%.12f [s]" % ((end - start) / loops_number))


def generate_all_features_1(bi: BasicInitiation):
    # delete previous placement of the objects
    x = np.array([], dtype=np.float64)
    y = np.array([], dtype=np.float64)

    # take parameters which were used in basic initiation
    bp = bi.basic_parameters

    # generate data of every co-location in given time frame
    for i_colloc in range(bp.n_base * bp.m_overlap):
        # calculate total number of all co-location feature instances
        collocation_features_instances_sum = bi.collocation_instances_counts[i_colloc] * bi.collocation_lengths[i_colloc]

        # generate vector of x coordinates of all the consecutive instances
        collocation_features_instances_x = np.random.randint(low=bi.area_in_cell_dim, size=(bi.collocation_instances_counts[i_colloc] - 1) // bp.m_clumpy + 1)
        collocation_features_instances_x *= bp.cell_size
        collocation_features_instances_x = collocation_features_instances_x.astype(dtype=np.float64)
        collocation_features_instances_x = np.repeat(a=collocation_features_instances_x, repeats=bp.m_clumpy)[:bi.collocation_instances_counts[i_colloc]]
        collocation_features_instances_x = np.repeat(a=collocation_features_instances_x, repeats=bi.collocation_lengths[i_colloc])
        collocation_features_instances_x += np.random.uniform(high=bp.cell_size, size=collocation_features_instances_sum)

        # generate vector of y coordinates of all the consecutive instances
        collocation_features_instances_y = np.random.randint(low=bi.area_in_cell_dim, size=(bi.collocation_instances_counts[i_colloc] - 1) // bp.m_clumpy + 1)
        collocation_features_instances_y *= bp.cell_size
        collocation_features_instances_y = collocation_features_instances_y.astype(dtype=np.float64)
        collocation_features_instances_y = np.repeat(a=collocation_features_instances_y, repeats=bp.m_clumpy)[:bi.collocation_instances_counts[i_colloc]]
        collocation_features_instances_y = np.repeat(a=collocation_features_instances_y, repeats=bi.collocation_lengths[i_colloc])
        collocation_features_instances_y += np.random.uniform(high=bp.cell_size, size=collocation_features_instances_sum)

        # remember data of current co-location features
        x = np.concatenate((x, collocation_features_instances_x))
        y = np.concatenate((y, collocation_features_instances_y))

    # generate data of every co-location noise feature in given time frame
    # generate vectors of x and y coordinates of all the consecutive instances of co-location noise features
    collocation_noise_features_instances_x = np.random.uniform(high=bp.area, size=bi.collocation_noise_features_instances_sum)
    collocation_noise_features_instances_y = np.random.uniform(high=bp.area, size=bi.collocation_noise_features_instances_sum)

    # remember data of co-location noise features
    x = np.concatenate((x, collocation_noise_features_instances_x))
    y = np.concatenate((y, collocation_noise_features_instances_y))

    # generate additional noise features if they are requested in given time frame
    if bp.ndf > 0:
        # generate vectors of x and y coordinates of all the consecutive instances of additional noise features
        additional_noise_features_instances_x = np.random.uniform(high=bp.area, size=bp.ndfn)
        additional_noise_features_instances_y = np.random.uniform(high=bp.area, size=bp.ndfn)

        # remember data of additional noise features
        x = np.concatenate((x, additional_noise_features_instances_x))
        y = np.concatenate((y, additional_noise_features_instances_y))


def generate_all_features_2(bi: BasicInitiation):

    # take parameters which were used in basic initiation
    bp = bi.basic_parameters

    # vectorized method of generating all features instances coordinates - with the awareness of the m_clumpy parameter
    collocations_clumpy_instances_coor = np.random.randint(low=bi.area_in_cell_dim, size=(bi.collocations_clumpy_instances_global_sum, 2))
    collocations_clumpy_instances_coor *= bp.cell_size
    collocations_clumpy_instances_coor = collocations_clumpy_instances_coor.astype(dtype=np.float64)
    features_instances_coor = collocations_clumpy_instances_coor[bi.collocations_clumpy_instances_global_ids]
    features_instances_coor += np.random.uniform(high=bp.cell_size, size=features_instances_coor.shape)


def test_generate_all_features():
    # average time execution of function generate_all_features_1:	0.002148352800 [s]
    # average time execution of function generate_all_features_2:	0.000386859190 [s]

    print("test_generate_all_features execute")

    bp = BasicParameters(
        area=1000,
        cell_size=5,
        n_base=10,
        lambda_1=4,
        lambda_2=50,
        m_clumpy=3,
        m_overlap=2,
        ncfr=0.5,
        ncfn=0.5,
        ncf_proportional=False,
        ndf=5,
        ndfn=250,
        random_seed=0
    )

    bi = BasicInitiation()
    bi.initiate(bp=bp)

    loops_number = 1000
    start = timer()
    for _ in range(loops_number):
        generate_all_features_1(bi=bi)
    end = timer()
    print("average time execution of function generate_all_features_1:\t%.12f [s]" % ((end - start) / loops_number))

    loops_number = 10000
    start = timer()
    for _ in range(loops_number):
        generate_all_features_2(bi=bi)
    end = timer()
    print("average time execution of function generate_all_features_2:\t%.12f [s]" % ((end - start) / loops_number))


def main():
    # test_generate_collocation_feature()
    # test_generate_collocation_feature_and_write()
    # test_generate_collocation_noise_feature()
    # test_write_collocation_noise_feature()
    # test_generate_additional_noise_feature()
    # test_spatial_basic_generator()
    test_generate_all_features()


if __name__ == "__main__":
    main()
