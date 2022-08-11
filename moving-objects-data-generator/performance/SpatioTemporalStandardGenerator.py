import numpy as np

from algorithms.initiation.StandardInitiation import StandardInitiation
from algorithms.parameters.StandardParameters import StandardParameters
from performance.Utils import measure_time_execution_of_function


def choose_spatial_prevalent_features_1(si, collocations_spatial_prevalent_instances_number):
    # boolean vector which tells if the given co-locations instance occurs in the current time frame
    collocations_instances_spatial_prevalent_flags = np.array([], dtype=bool)

    # determine values of the 'collocations_instances_spatial_prevalent_flags' vector
    for i_colloc in range(si.collocations_sum):
        # choose indices of the instances of the 'i_colloc' co-location which actually create co-location in the current time frame
        i_colloc_spatial_prevalent_instances_ids = np.random.choice(a=si.collocation_instances_counts[i_colloc], size=collocations_spatial_prevalent_instances_number[i_colloc], replace=False)

        # create boolean vector which tells if the given instance of the 'i_colloc' co-location occurs in the current time frame
        i_colloc_spatial_prevalent_instances_flags = np.zeros(shape=si.collocation_instances_counts[i_colloc], dtype=bool)
        i_colloc_spatial_prevalent_instances_flags[i_colloc_spatial_prevalent_instances_ids] = True

        # append flags of the instances of the 'i_colloc' co-location
        collocations_instances_spatial_prevalent_flags = np.concatenate((collocations_instances_spatial_prevalent_flags, i_colloc_spatial_prevalent_instances_flags))


def choose_spatial_prevalent_features_2(si, collocations_spatial_prevalent_instances_number):
    collocation_instances_counts_cumsum = np.cumsum(si.collocation_instances_counts)

    shuffled_values = np.repeat(a=si.collocation_instances_counts - collocation_instances_counts_cumsum, repeats=si.collocation_instances_counts) + np.arange(1, si.collocations_instances_sum + 1)

    ind_begin = np.concatenate(([0], collocation_instances_counts_cumsum[: -1]))

    [np.random.shuffle(shuffled_values[ind_begin[i]: collocation_instances_counts_cumsum[i]]) for i in range(si.collocations_sum)]

    collocations_instances_spatial_prevalent_flags = shuffled_values <= np.repeat(a=collocations_spatial_prevalent_instances_number, repeats=si.collocation_instances_counts)


def choose_spatial_prevalent_features_3_f(i, permuted_values, ind_begin, collocation_instances_counts_cumsum, si):
    permuted_values[ind_begin[i]: collocation_instances_counts_cumsum[i]] = np.random.permutation(si.collocation_instances_counts[i]) + 1


def choose_spatial_prevalent_features_3(si, collocations_spatial_prevalent_instances_number):
    permuted_values = np.zeros(shape=si.collocations_instances_sum, dtype=np.int32)

    collocation_instances_counts_cumsum = np.cumsum(si.collocation_instances_counts)

    ind_begin = np.concatenate(([0], collocation_instances_counts_cumsum[: -1]))

    [choose_spatial_prevalent_features_3_f(i, permuted_values, ind_begin, collocation_instances_counts_cumsum, si) for i in range(si.collocations_sum)]

    collocations_instances_spatial_prevalent_flags = permuted_values <= np.repeat(a=collocations_spatial_prevalent_instances_number, repeats=si.collocation_instances_counts)


def choose_spatial_prevalent_features_4(si, collocations_spatial_prevalent_instances_number):
    collocation_instances_counts_cumsum = np.cumsum(si.collocation_instances_counts)

    shuffled_values = np.repeat(a=si.collocation_instances_counts - collocation_instances_counts_cumsum, repeats=si.collocation_instances_counts) + np.arange(1, si.collocations_instances_sum + 1)

    ind_begin = np.concatenate(([0], collocation_instances_counts_cumsum[: -1]))

    for i in range(si.collocations_sum):
        np.random.shuffle(shuffled_values[ind_begin[i]: collocation_instances_counts_cumsum[i]])

    collocations_instances_spatial_prevalent_flags = shuffled_values <= np.repeat(a=collocations_spatial_prevalent_instances_number, repeats=si.collocation_instances_counts)


def choose_spatial_prevalent_features_5(si, collocations_spatial_prevalent_instances_number):
    permuted_values = np.zeros(shape=si.collocations_instances_sum, dtype=np.int32)

    collocation_instances_counts_cumsum = np.cumsum(si.collocation_instances_counts)

    ind_begin = np.concatenate(([0], collocation_instances_counts_cumsum[: -1]))

    for i in range(si.collocations_sum):
        permuted_values[ind_begin[i]: collocation_instances_counts_cumsum[i]] = np.random.permutation(si.collocation_instances_counts[i]) + 1

    collocations_instances_spatial_prevalent_flags = permuted_values <= np.repeat(a=collocations_spatial_prevalent_instances_number, repeats=si.collocation_instances_counts)


def test_choose_spatial_prevalent_features():
    # average time execution of function choose_spatial_prevalent_features_1:	0.000969810300 [s]
    # average time execution of function choose_spatial_prevalent_features_2:	0.000130195050 [s]
    # average time execution of function choose_spatial_prevalent_features_3:	0.000307862230 [s]
    # average time execution of function choose_spatial_prevalent_features_4:	0.000132216470 [s]
    # average time execution of function choose_spatial_prevalent_features_5:	0.000301331660 [s]

    print("test_choose_spatial_prevalent_features execute")
    sp = StandardParameters(
        area=1000,
        cell_size=5,
        n_colloc=10,
        lambda_1=4,
        lambda_2=30,
        m_clumpy=3,
        m_overlap=2,
        ncfr=0,
        ncfn=0,
        ncf_proportional=False,
        ndf=1,
        ndfn=40,
        random_seed=0,
        persistent_ratio=0.5,
        spatial_prevalence_threshold=0.5,
        time_prevalence_threshold=0.5
    )

    si = StandardInitiation()
    si.initiate(sp=sp)

    time_frames_number = 100

    # determine the minimal number of the given co-location instances occurrence, which makes the co-location becomes spatial prevalent
    collocations_instances_number_spatial_prevalence_threshold = np.ceil(sp.spatial_prevalence_threshold * si.collocation_instances_counts).astype(np.int32)

    # determine the number of time frames when the given co-location pattern is spatial prevalent
    collocations_time_frames_numbers_of_spatial_prevalence = np.random.randint(low=1, high=time_frames_number + 1, size=si.collocations_sum)

    # decide which of the co-locations patterns are spatial prevalent in the current time frame
    random_value = np.random.randint(low=1, high=time_frames_number + 1, size=si.collocations_sum)
    collocations_spatial_prevalence_flags = random_value <= collocations_time_frames_numbers_of_spatial_prevalence

    # determine the number of the co-locations instances which actually creates co-location in the current time frame
    collocations_spatial_prevalent_instances_number = np.zeros(shape=si.collocations_sum, dtype=np.int32)
    collocations_spatial_prevalent_instances_number[np.logical_not(collocations_spatial_prevalence_flags)] = np.random.randint(
        low=0,
        high=collocations_instances_number_spatial_prevalence_threshold[np.logical_not(collocations_spatial_prevalence_flags)]
    )
    collocations_spatial_prevalent_instances_number[collocations_spatial_prevalence_flags] = np.random.randint(
        low=collocations_instances_number_spatial_prevalence_threshold[collocations_spatial_prevalence_flags],
        high=si.collocation_instances_counts[collocations_spatial_prevalence_flags] + 1
    )

    parameters = {
        "si": si,
        "collocations_spatial_prevalent_instances_number": collocations_spatial_prevalent_instances_number
    }

    measure_time_execution_of_function(func=choose_spatial_prevalent_features_1, loops_number=1000, parameters=parameters)
    measure_time_execution_of_function(func=choose_spatial_prevalent_features_2, loops_number=10000, parameters=parameters)
    measure_time_execution_of_function(func=choose_spatial_prevalent_features_3, loops_number=10000, parameters=parameters)
    measure_time_execution_of_function(func=choose_spatial_prevalent_features_4, loops_number=10000, parameters=parameters)
    measure_time_execution_of_function(func=choose_spatial_prevalent_features_5, loops_number=10000, parameters=parameters)


def permutation_method_1(n):
    x = np.random.permutation(np.arange(1, n + 1))


def permutation_method_2(n):
    x = np.random.permutation(n) + 1


def test_permutation_method():
    # average time execution of function permutation_method_1:	0.003004908500 [s]
    # average time execution of function permutation_method_2:	0.002839237400 [s]

    print("test_permutation_method execute")

    parameters = {
        "n": 100000
    }

    measure_time_execution_of_function(func=permutation_method_1, loops_number=1000, parameters=parameters)
    measure_time_execution_of_function(func=permutation_method_2, loops_number=1000, parameters=parameters)


def arange_method_1(n):
    x = np.arange(1, n + 1)


def arange_method_2(n):
    x = np.arange(n) + 1


def test_arange_method():
    # average time execution of function arange_method_1:	0.000076439460 [s]
    # average time execution of function arange_method_2:	0.000251140710 [s]

    print("test_arange_method execute")

    parameters = {
        "n": 100000
    }

    measure_time_execution_of_function(func=arange_method_1, loops_number=10000, parameters=parameters)
    measure_time_execution_of_function(func=arange_method_2, loops_number=10000, parameters=parameters)


def main():
    # test_choose_spatial_prevalent_features()
    # test_permutation_method()
    test_arange_method()


if __name__ == "__main__":
    main()
