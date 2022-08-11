from timeit import default_timer as timer


def measure_time_execution_of_function(func, loops_number: int, parameters: dict):
    start = timer()
    for _ in range(loops_number):
        func(**parameters)
    end = timer()
    print("average time execution of function %s:\t%.12f [s]" % (func.__name__, (end - start) / loops_number))