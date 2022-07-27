import numpy as np

from performance.Utils import measure_time_execution_of_function


def detect_travel_end_1(collocations_instances_global_ids, flag, collocations_instances_global_ids_repeats):
    not_reached = collocations_instances_global_ids[flag == 0]
    not_reached = np.unique(ar=not_reached)
    collocation_reached_flag = np.ones_like(a=collocations_instances_global_ids_repeats, dtype=bool)
    collocation_reached_flag[not_reached] = False
    features_new_destination_needed = np.repeat(a=collocation_reached_flag, repeats=collocations_instances_global_ids_repeats)


def test_detect_travel_end():
    # average time execution of function detect_travel_end_1:	0.000104014380 [s]

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

    collocations_instances_global_ids_repeats = np.concatenate((
        np.repeat(a=collocation_lengths, repeats=collocation_instances_counts),
        np.ones(shape=nfn, dtype=np.int32)
    ))

    parameters = {
        "collocations_instances_global_ids": collocations_instances_global_ids,
        "flag": flag,
        "collocations_instances_global_ids_repeats": collocations_instances_global_ids_repeats
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


def out_of_range_correction_1(features_step_angle_std, features_instances_sum, features_step_angle_range, features_ids):
    np.random.seed(0)
    features_instances_step_angle = np.random.normal(loc=0.0, scale=features_step_angle_std[features_ids], size=features_instances_sum)

    angle_out_of_range_indices = np.nonzero(np.logical_or(
        features_instances_step_angle < -features_step_angle_range[features_ids],
        features_instances_step_angle > features_step_angle_range[features_ids]
    ))

    features_instances_step_angle[angle_out_of_range_indices] = np.random.uniform(
        low=-features_step_angle_range[features_ids[angle_out_of_range_indices]],
        high=features_step_angle_range[features_ids[angle_out_of_range_indices]],
        size=angle_out_of_range_indices[0].size
    )


def out_of_range_correction_2(features_step_angle_std, features_instances_sum, features_step_angle_range, features_ids):
    np.random.seed(0)
    features_instances_step_angle = np.random.normal(loc=0.0, scale=features_step_angle_std[features_ids], size=features_instances_sum)

    angle_out_of_range_indices = np.flatnonzero(np.logical_or(
        features_instances_step_angle < -features_step_angle_range[features_ids],
        features_instances_step_angle > features_step_angle_range[features_ids]
    ))
    features_instances_step_angle[angle_out_of_range_indices] = np.random.uniform(
        low=-features_step_angle_range[features_ids[angle_out_of_range_indices]],
        high=features_step_angle_range[features_ids[angle_out_of_range_indices]],
        size=angle_out_of_range_indices.size
    )


def test_out_of_range_correction():
    # average time execution of function out_of_range_correction_1:	0.000170753140 [s]
    # average time execution of function out_of_range_correction_1:	0.000166520010 [s]

    print("test_out_of_range_correction execute")

    np.random.seed(0)

    features_sum = 10
    features_step_angle_range = np.random.gamma(shape=20, scale=1.0, size=features_sum)
    features_step_angle_std = np.random.gamma(shape=10, scale=1.0, size=features_sum)
    features_instances_sum = 1000
    features_ids = np.random.randint(low=features_sum, size=features_instances_sum)

    parameters = {
        "features_step_angle_std": features_step_angle_std,
        "features_instances_sum": features_instances_sum,
        "features_step_angle_range": features_step_angle_range,
        "features_ids": features_ids
    }

    measure_time_execution_of_function(func=out_of_range_correction_1, loops_number=10000, parameters=parameters)
    measure_time_execution_of_function(func=out_of_range_correction_1, loops_number=10000, parameters=parameters)


def calculate_rotation_1(instances_coor_delta_direct, features_instances_step_angle):
    instances_coor_delta_rotated = np.empty_like(instances_coor_delta_direct)
    instances_coor_delta_rotated[:, 0] = np.cos(features_instances_step_angle) * instances_coor_delta_direct[:, 0] - np.sin(features_instances_step_angle) * instances_coor_delta_direct[:, 1]
    instances_coor_delta_rotated[:, 1] = np.sin(features_instances_step_angle) * instances_coor_delta_direct[:, 0] + np.cos(features_instances_step_angle) * instances_coor_delta_direct[:, 1]


def calculate_rotation_2(instances_coor_delta_direct, features_instances_step_angle):
    instances_coor_delta_rotated = np.empty_like(instances_coor_delta_direct)
    cos_angle = np.cos(features_instances_step_angle)
    sin_angle = np.sin(features_instances_step_angle)
    instances_coor_delta_rotated[:, 0] = cos_angle * instances_coor_delta_direct[:, 0] - sin_angle * instances_coor_delta_direct[:, 1]
    instances_coor_delta_rotated[:, 1] = sin_angle * instances_coor_delta_direct[:, 0] + cos_angle * instances_coor_delta_direct[:, 1]


def test_calculate_rotation():
    # average time execution of function calculate_rotation_1:	0.000085799460 [s]
    # average time execution of function calculate_rotation_2:	0.000054011550 [s]

    print("test_calculate_rotation execute")

    np.random.seed(0)

    features_instances_sum = 1000
    instances_coor_delta_direct = np.random.uniform(low=-100, high=100, size=(features_instances_sum, 2))
    features_instances_step_angle = np.random.uniform(low=-np.pi, high=np.pi, size=features_instances_sum)

    parameters = {
        "instances_coor_delta_direct": instances_coor_delta_direct,
        "features_instances_step_angle": features_instances_step_angle
    }

    measure_time_execution_of_function(func=calculate_rotation_1, loops_number=10000, parameters=parameters)
    measure_time_execution_of_function(func=calculate_rotation_2, loops_number=10000, parameters=parameters)


def main():
    test_detect_travel_end()
    # test_set_destination_point()
    # test_out_of_range_correction()
    # test_calculate_rotation()


if __name__ == "__main__":
    main()
