import sys
import numpy as np

from timeit import default_timer as timer

from algorithms.enums.InteractionApproachEnums import IdenticalFeaturesInteractionMode, DifferentFeaturesInteractionMode, MassMode, VelocityMode
from algorithms.generator.SpatioTemporalInteractionApproachGenerator import SpatioTemporalInteractionApproachGenerator
from algorithms.parameters.InteractionApproachParameters import InteractionApproachParameters
from performance.Utils import measure_time_execution_of_function


def compute_distance_1(x, y):
    d = np.empty(shape=(x.size, x.size), dtype=np.float64)
    for i in range(x.size):
        for j in range(x.size):
            d[i, j] = ((x[j] - x[i]) ** 2 + (y[j] - y[i]) ** 2) ** 0.5

    # print(d[0, :5])
    # [  0.         674.6806652  612.47448926 503.71420468 487.95184189]


def compute_distance_2(x, y):
    d = np.empty(shape=(x.size, x.size), dtype=np.float64)
    for i in range(x.size):
        d[i] = ((x - x[i]) ** 2 + (y - y[i]) ** 2) ** 0.5

    # print(d[0, :5])
    # [  0.         674.6806652  612.47448926 503.71420468 487.95184189]


def compute_distance_3(x, y):
    ind_col = np.tile(A=np.arange(x.size, dtype=np.int32), reps=(x.size, 1))
    ind_row = ind_col.T
    d = ((x[ind_col] - x[ind_row]) ** 2 + (y[ind_col] - y[ind_row]) ** 2) ** 0.5

    # print(d[0, :5])
    # [  0.         674.6806652  612.47448926 503.71420468 487.95184189]


def compute_distance_4(x, y):
    d = np.empty(shape=(x.size, x.size), dtype=np.float64)
    p = np.column_stack(tup=(x, y))
    for i in range(p.shape[0]):
        d[i] = np.sqrt(np.sum(a=(p - p[i]) ** 2, axis=1))

    # print(d[0, :5])
    # [  0.         674.6806652  612.47448926 503.71420468 487.95184189]


def compute_distance_5(x, y):
    d = np.empty(shape=(x.size, x.size), dtype=np.float64)
    p = np.column_stack(tup=(x, y))
    for i in range(p.shape[0]):
        d[i] = np.sqrt(np.sum(a=p[i] * p[i] - 2 * p[i] * p + p * p, axis=1))

    # print(d[0, :5])
    # [  0.         674.6806652  612.47448926 503.71420468 487.95184189]


def compute_distance_6(x, y):
    d = np.sqrt((x[None, :] - x[:, None]) ** 2 + (y[None, :] - y[:, None]) ** 2)

    # print(d[0, :5])
    # [  0.         674.6806652  612.47448926 503.71420468 487.95184189]


def compute_distance_7(x, y):
    p = np.column_stack(tup=(x, y))
    d = np.sqrt(np.sum(a=(p[None, :, :] - p[:, None, :]) ** 2, axis=-1))

    # print(d[0, :5])
    # [  0.         674.6806652  612.47448926 503.71420468 487.95184189]


def compute_distance_8(x, y):
    p = np.column_stack(tup=(x, y))
    p_summed_squares = np.sum(p ** 2, axis=-1)
    d = np.sqrt(p_summed_squares[:, None] - 2 * p.dot(p.T) + p_summed_squares[None, :])

    # print(d[0, :5])
    # [  0.         674.6806652  612.47448926 503.71420468 487.95184189]


def compute_distance_9(x, y):
    p = np.column_stack(tup=(x, y))
    d = np.linalg.norm(x=p[None, :, :] - p[:, None, :], axis=-1)

    # print(d[0, :5])
    # [  0.         674.6806652  612.47448926 503.71420468 487.95184189]


def compute_distance_10(x, y):
    p = np.column_stack(tup=(x, y))
    ti = np.triu_indices(n=p.shape[0], k=1)
    d = p[ti[0]] - p[ti[1]]
    d = np.sqrt(np.sum(d * d, axis=-1))

    # print(d[:5])
    # [674.6806652  612.47448926 503.71420468 487.95184189 353.92937149]


def test_compute_distance():
    # average time execution of function compute_distance_1:	5.703573300000 [s]
    # average time execution of function compute_distance_2:	0.075562165000 [s]
    # average time execution of function compute_distance_3:	0.111450958000 [s]
    # average time execution of function compute_distance_4:	0.058626845000 [s]
    # average time execution of function compute_distance_5:	0.075097362000 [s]
    # average time execution of function compute_distance_6:	0.019551780000 [s]
    # average time execution of function compute_distance_7:	0.044773641000 [s]
    # average time execution of function compute_distance_8:	0.023777113000 [s]
    # average time execution of function compute_distance_9:	0.063297421000 [s]
    # average time execution of function compute_distance_10:	0.046993753000 [s]

    print("test_compute_distance execute")

    np.random.seed(0)
    x = np.random.randint(low=1000, size=1000)
    y = np.random.randint(low=1000, size=1000)

    parameters = {
        "x": x,
        "y": y
    }

    measure_time_execution_of_function(func=compute_distance_1, loops_number=1, parameters=parameters)
    measure_time_execution_of_function(func=compute_distance_2, loops_number=100, parameters=parameters)
    measure_time_execution_of_function(func=compute_distance_3, loops_number=100, parameters=parameters)
    measure_time_execution_of_function(func=compute_distance_4, loops_number=100, parameters=parameters)
    measure_time_execution_of_function(func=compute_distance_5, loops_number=100, parameters=parameters)
    measure_time_execution_of_function(func=compute_distance_6, loops_number=100, parameters=parameters)
    measure_time_execution_of_function(func=compute_distance_7, loops_number=100, parameters=parameters)
    measure_time_execution_of_function(func=compute_distance_8, loops_number=100, parameters=parameters)
    measure_time_execution_of_function(func=compute_distance_9, loops_number=100, parameters=parameters)
    measure_time_execution_of_function(func=compute_distance_10, loops_number=100, parameters=parameters)


def compute_abs_value_3d_1(x):
    np.sqrt(np.sum(a=x ** 2, axis=2))
    # np.sum(x, axis=2)


def compute_abs_value_3d_2(x):
    np.sqrt(np.sum(a=x ** 2, axis=1))
    # np.sum(x, axis=1)


def compute_abs_value_3d_3(x):
    np.sqrt(np.sum(a=x ** 2, axis=0))
    # np.sum(x, axis=0)


def test_compute_abs_value_3d():
    # average time execution of function compute_abs_value_3d_1:	0.036540626000 [s]
    # average time execution of function compute_abs_value_3d_2:	0.022803977000 [s]
    # average time execution of function compute_abs_value_3d_3:	0.024207840000 [s]

    print("test_compute_abs_value_3d execute")

    np.random.seed(0)
    x = np.random.uniform(low=-1000, high=1000, size=(1000, 1000, 2))
    y = np.random.uniform(low=-1000, high=1000, size=(1000, 2, 1000))
    z = np.random.uniform(low=-1000, high=1000, size=(2, 1000, 1000))

    measure_time_execution_of_function(func=compute_abs_value_3d_1, loops_number=100, parameters={"x": x})
    measure_time_execution_of_function(func=compute_abs_value_3d_2, loops_number=100, parameters={"x": y})
    measure_time_execution_of_function(func=compute_abs_value_3d_3, loops_number=100, parameters={"x": z})


def compute_abs_value_2d_1(x):
    np.sqrt(np.sum(a=x ** 2, axis=1))
    # np.sum(x, axis=1)


def compute_abs_value_2d_2(x):
    np.sqrt(np.sum(a=x ** 2, axis=0))
    # np.sum(x, axis=0)


def test_compute_abs_value_2d():
    # average time execution of function compute_abs_value_2d_1:	0.000244932370 [s]
    # average time execution of function compute_abs_value_2d_2:	0.000083061050 [s]

    print("test_compute_abs_value_2d execute")

    np.random.seed(0)
    x = np.random.uniform(low=-1000, high=1000, size=(10000, 2))
    y = np.random.uniform(low=-1000, high=1000, size=(2, 10000))

    measure_time_execution_of_function(func=compute_abs_value_2d_1, loops_number=10000, parameters={"x": x})
    measure_time_execution_of_function(func=compute_abs_value_2d_2, loops_number=10000, parameters={"x": y})


def compute_sum_long_axis_3d_1(x):
    np.sum(x, axis=1)


def compute_sum_long_axis_3d_2(x):
    np.sum(x, axis=0)


def compute_sum_long_axis_3d_3(x):
    np.sum(x, axis=2)


def compute_sum_long_axis_3d_4(x):
    np.sum(x, axis=0)


def compute_sum_long_axis_3d_5(x):
    np.sum(x, axis=2)


def compute_sum_long_axis_3d_6(x):
    np.sum(x, axis=1)


def test_compute_sum_long_axis_3d():
    # average time execution of function compute_sum_long_axis_3d_1:	0.025574868000 [s]
    # average time execution of function compute_sum_long_axis_3d_2:	0.001968991800 [s]
    # average time execution of function compute_sum_long_axis_3d_3:	0.002720464700 [s]
    # average time execution of function compute_sum_long_axis_3d_4:	0.001993296800 [s]
    # average time execution of function compute_sum_long_axis_3d_5:	0.002722109700 [s]
    # average time execution of function compute_sum_long_axis_3d_6:	0.001967167400 [s]

    print("test_compute_sum_long_axis_3d execute")

    np.random.seed(0)
    x = np.random.uniform(low=-1000, high=1000, size=(1000, 1000, 2))
    y = np.random.uniform(low=-1000, high=1000, size=(1000, 2, 1000))
    z = np.random.uniform(low=-1000, high=1000, size=(2, 1000, 1000))

    measure_time_execution_of_function(func=compute_sum_long_axis_3d_1, loops_number=100, parameters={"x": x})
    measure_time_execution_of_function(func=compute_sum_long_axis_3d_2, loops_number=1000, parameters={"x": x})
    measure_time_execution_of_function(func=compute_sum_long_axis_3d_3, loops_number=1000, parameters={"x": y})
    measure_time_execution_of_function(func=compute_sum_long_axis_3d_4, loops_number=1000, parameters={"x": y})
    measure_time_execution_of_function(func=compute_sum_long_axis_3d_5, loops_number=1000, parameters={"x": z})
    measure_time_execution_of_function(func=compute_sum_long_axis_3d_6, loops_number=1000, parameters={"x": z})


def compute_abs_value_2d_after_swap_1(x):
    np.sqrt(np.sum(a=x ** 2, axis=1))
    # np.sum(x, axis=1)


def compute_abs_value_2d_after_swap_2(x):
    np.sqrt(np.sum(a=x ** 2, axis=0))
    # np.sum(x, axis=0)


def test_compute_abs_value_2d_after_swap():
    # average time execution of function compute_abs_value_2d_1:	0.000080709550 [s]
    # average time execution of function compute_abs_value_2d_1:	0.000080798720 [s]
    # average time execution of function compute_abs_value_2d_1:	0.000234643320 [s]
    # average time execution of function compute_abs_value_2d_1:	0.000220696860 [s]
    # average time execution of function compute_abs_value_2d_2:	0.000235092020 [s]
    # average time execution of function compute_abs_value_2d_2:	0.000235496460 [s]
    # average time execution of function compute_abs_value_2d_2:	0.000082360560 [s]
    # average time execution of function compute_abs_value_2d_2:	0.000101585080 [s]

    print("test_compute_abs_value_2d execute")

    np.random.seed(0)
    x = np.random.uniform(low=-1000, high=1000, size=(10000, 2))
    x_swaped = x.T
    x_copied_1 = np.copy(x.T)
    x_copied_2 = np.copy(x.T, order='C')
    x_copied_3 = np.empty(shape=(2, 10000), dtype=np.int64)
    x_copied_3[:] = x.T
    y = np.random.uniform(low=-1000, high=1000, size=(2, 10000))
    y_swaped = y.T
    y_copied_1 = np.copy(y.T)
    y_copied_2 = np.copy(y.T, order='C')
    y_copied_3 = np.empty(shape=(10000, 2), dtype=np.int64)
    y_copied_3[:] = y.T

    measure_time_execution_of_function(func=compute_abs_value_2d_1, loops_number=10000, parameters={"x": y_swaped})
    measure_time_execution_of_function(func=compute_abs_value_2d_1, loops_number=10000, parameters={"x": y_copied_1})
    measure_time_execution_of_function(func=compute_abs_value_2d_1, loops_number=10000, parameters={"x": y_copied_2})
    measure_time_execution_of_function(func=compute_abs_value_2d_1, loops_number=10000, parameters={"x": y_copied_3})
    measure_time_execution_of_function(func=compute_abs_value_2d_2, loops_number=10000, parameters={"x": x_swaped})
    measure_time_execution_of_function(func=compute_abs_value_2d_2, loops_number=10000, parameters={"x": x_copied_1})
    measure_time_execution_of_function(func=compute_abs_value_2d_2, loops_number=10000, parameters={"x": x_copied_2})
    measure_time_execution_of_function(func=compute_abs_value_2d_2, loops_number=10000, parameters={"x": x_copied_3})


def test_spatio_temporal_interaction_approach_generator():
    # --- speedtest before controlling memory order of arrays, which is optimal for reducing functions
    # average time execution of SpatioTemporalInteractionApproachGenerator:	4.576686700000 [s]
    # average time execution of SpatioTemporalInteractionApproachGenerator:	5.581902400000 [s]
    # --- speedtest after controlling memory order of arrays, while calculating resultant forces
    # average time execution of SpatioTemporalInteractionApproachGenerator:	3.487461700000 [s]
    # average time execution of SpatioTemporalInteractionApproachGenerator:	4.265457510000 [s]

    print("test_spatio_temporal_interaction_approach_generator execute")

    loops_number = 1
    save_stdout = sys.stdout
    sys.stdout = open('trash.txt', 'w')
    start = timer()
    for _ in range(loops_number):

        # generating parameters
        iap = InteractionApproachParameters(
            area=1000,
            cell_size=5,
            n_colloc=5,
            lambda_1=4,
            lambda_2=50,
            m_clumpy=1,
            m_overlap=1,
            ncfr=0.5,
            ncfn=0.3,
            ncf_proportional=False,
            ndf=3,
            ndfn=30,
            random_seed=0,
            time_unit=1,
            distance_unit=1.0,
            approx_steps_number=1,
            k_force=10,
            force_limit=20.0,
            velocity_limit=20.0,
            faraway_limit_ratio=np.sqrt(2) / 2,
            mass_mode=MassMode.CONSTANT,
            mass_mean=1.0,
            mass_normal_std_ratio=1 / 5,
            velocity_mode=VelocityMode.CONSTANT,
            velocity_mean=0.0,
            identical_features_interaction_mode=IdenticalFeaturesInteractionMode.ATTRACT,
            different_features_interaction_mode=DifferentFeaturesInteractionMode.ATTRACT
        )

        stiag = SpatioTemporalInteractionApproachGenerator(iap=iap)
        stiag.generate(
            time_frames_number=10,
            output_filename="output_file.txt",
            output_filename_timestamp=False
        )

    end = timer()
    sys.stdout = save_stdout
    print("average time execution of SpatioTemporalInteractionApproachGenerator:\t%.12f [s]" % ((end - start) / loops_number))

    loops_number = 10
    save_stdout = sys.stdout
    sys.stdout = open('trash.txt', 'w')
    start = timer()
    for _ in range(loops_number):

        # generating parameters
        iap = InteractionApproachParameters(
            area=1000,
            cell_size=3,
            n_colloc=5,
            lambda_1=4,
            lambda_2=20,
            m_clumpy=1,
            m_overlap=1,
            ncfr=0,
            ncfn=0,
            ncf_proportional=False,
            ndf=3,
            ndfn=30,
            random_seed=0,
            time_unit=1,
            distance_unit=1.0,
            approx_steps_number=1,
            k_force=10,
            force_limit=20.0,
            velocity_limit=20.0,
            faraway_limit_ratio=np.sqrt(2) / 2,
            mass_mode=MassMode.CONSTANT,
            mass_mean=1.0,
            mass_normal_std_ratio=1 / 5,
            velocity_mode=VelocityMode.CONSTANT,
            velocity_mean=0.0,
            identical_features_interaction_mode=IdenticalFeaturesInteractionMode.ATTRACT,
            different_features_interaction_mode=DifferentFeaturesInteractionMode.ATTRACT
        )

        stiag = SpatioTemporalInteractionApproachGenerator(iap=iap)
        stiag.generate(
            time_frames_number=100,
            output_filename="output_file.txt",
            output_filename_timestamp=False
        )

    end = timer()
    sys.stdout = save_stdout
    print("average time execution of SpatioTemporalInteractionApproachGenerator:\t%.12f [s]" % ((end - start) / loops_number))


def main():
    # test_compute_distance()
    # test_compute_abs_value_3d()
    # test_compute_abs_value_2d()
    # test_compute_sum_long_axis_3d()
    # test_compute_abs_value_2d_after_swap()
    test_spatio_temporal_interaction_approach_generator()


if __name__ == "__main__":
    main()
