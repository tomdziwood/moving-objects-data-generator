import numpy as np

from performance.Utils import measure_time_execution_of_function


def detect_travel_end_1(collocations_instances_global_ids, flag):
    not_reached = collocations_instances_global_ids[flag == 0]
    not_reached = np.unique(ar=not_reached)
    reached_flag = np.ones_like(a=collocations_instances_global_ids)
    reached_flag[not_reached] = 0


def test_detect_travel_end():
    # average time execution of function detect_travel_end_1:	0.000079343590 [s]

    print("test_detect_travel_end execute")

    np.random.seed(0)

    n_colloc = 10
    collocation_lengths = np.random.poisson(lam=5, size=n_colloc)
    print("collocation_lengths=%s" % str(collocation_lengths))
    collocation_instances_counts = np.random.poisson(lam=30, size=n_colloc)
    print("collocation_instances_counts=%s" % str(collocation_instances_counts))
    nfn = 1000

    collocations_instances_global_ids = np.array([], dtype=np.int32)

    last_id = 0
    for i_colloc in range(n_colloc):
        i_colloc_collocations_instances_global_ids = np.repeat(a=np.arange(last_id, last_id + collocation_instances_counts[i_colloc]), repeats=collocation_lengths[i_colloc])
        collocations_instances_global_ids = np.concatenate((collocations_instances_global_ids, i_colloc_collocations_instances_global_ids))
        last_id += collocation_instances_counts[i_colloc]

    collocations_instances_global_ids = np.concatenate((collocations_instances_global_ids, np.arange(last_id, last_id + nfn)))
    flag = np.random.randint(low=2, size=collocations_instances_global_ids.size)

    parameters = {
        "collocations_instances_global_ids": collocations_instances_global_ids,
        "flag": flag
    }

    measure_time_execution_of_function(func=detect_travel_end_1, loops_number=10000, parameters=parameters)


def set_destination_point_1(area_in_cell_dim, collocation_instances_global_sum, cell_size, collocations_instances_global_ids):
    collocation_instances_destination_coor = np.random.randint(low=area_in_cell_dim, size=(collocation_instances_global_sum, 2))
    collocation_instances_destination_coor *= cell_size
    collocation_instances_destination_coor = collocation_instances_destination_coor.astype(dtype=np.float64)
    features_instances_destination_coor = collocation_instances_destination_coor[collocations_instances_global_ids]
    features_instances_destination_coor += np.random.uniform(high=cell_size, size=features_instances_destination_coor.shape)


def test_set_destination_point():
    # average time execution of function set_destination_point_1:	0.000205877070 [s]

    print("test_set_destination_point execute")

    np.random.seed(0)

    n_colloc = 10
    collocation_lengths = np.random.poisson(lam=5, size=n_colloc)
    print("collocation_lengths=%s" % str(collocation_lengths))
    collocation_instances_counts = np.random.poisson(lam=30, size=n_colloc)
    print("collocation_instances_counts=%s" % str(collocation_instances_counts))
    nfn = 1000

    collocations_instances_global_ids = np.array([], dtype=np.int32)

    last_id = 0
    for i_colloc in range(n_colloc):
        i_colloc_collocations_instances_global_ids = np.repeat(a=np.arange(last_id, last_id + collocation_instances_counts[i_colloc]), repeats=collocation_lengths[i_colloc])
        collocations_instances_global_ids = np.concatenate((collocations_instances_global_ids, i_colloc_collocations_instances_global_ids))
        last_id += collocation_instances_counts[i_colloc]

    collocations_instances_global_ids = np.concatenate((collocations_instances_global_ids, np.arange(last_id, last_id + nfn)))
    print("collocations_instances_global_ids=%s" % str(collocations_instances_global_ids))
    print("collocations_instances_global_ids.shape=%s" % str(collocations_instances_global_ids.shape))

    collocation_instances_global_sum = last_id + nfn
    print("collocation_instances_global_sum=%s" % str(collocation_instances_global_sum))

    area_in_cell_dim = 200
    cell_size = 5

    parameters = {
        "area_in_cell_dim": area_in_cell_dim,
        "collocation_instances_global_sum": collocation_instances_global_sum,
        "cell_size": cell_size,
        "collocations_instances_global_ids": collocations_instances_global_ids
    }

    measure_time_execution_of_function(func=set_destination_point_1, loops_number=10000, parameters=parameters)


def main():
    # test_detect_travel_end()
    test_set_destination_point()


if __name__ == "__main__":
    main()
