import numpy as np

from performance.Utils import measure_time_execution_of_function


def mean_coordinates_1(ids, repeats, coor):
    slice_ind = np.concatenate(([0], repeats.cumsum()[:-1]))
    sums = np.add.reduceat(coor, indices=slice_ind, axis=0)
    mean = sums / repeats[:, None]


def mean_coordinates_2(ids, repeats, coor):
    unq, inverse_ids, counts = np.unique(ids, return_inverse=True, return_counts=True)
    x_mean = np.bincount(inverse_ids, coor[:, 0]) / counts
    y_mean = np.bincount(inverse_ids, coor[:, 1]) / counts
    mean = np.column_stack((x_mean, y_mean))


def test_mean_coordinates():
    # average time execution of function mean_coordinates_1:	0.000148614100 [s]
    # average time execution of function mean_coordinates_2:	0.001295169500 [s]

    print("test_mean_coordinates execute")

    np.random.seed(0)

    size = 2000
    repeats = np.random.randint(low=5, high=20, size=size)
    ids = np.repeat(a=np.arange(size), repeats=repeats)
    coor = np.random.uniform(high=1000.0, size=(ids.size, 2))

    parameters = {
        "ids": ids,
        "repeats": repeats,
        "coor": coor
    }

    measure_time_execution_of_function(func=mean_coordinates_1, loops_number=1000, parameters=parameters)
    measure_time_execution_of_function(func=mean_coordinates_2, loops_number=1000, parameters=parameters)


def main():
    test_mean_coordinates()


if __name__ == "__main__":
    main()