import numpy as np

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


def main():
    test_compute_distance()


if __name__ == "__main__":
    main()
