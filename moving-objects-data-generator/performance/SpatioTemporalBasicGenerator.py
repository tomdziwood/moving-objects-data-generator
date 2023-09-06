import sys

from timeit import default_timer as timer
from scripts.iterative import SpatioTemporalBasicGenerator as istbg
from scripts.vectorized import SpatioTemporalBasicGenerator as vstbg


def test_spatio_temporal_standard_generator():
    # average time execution of function istbg.generate:	47.366100700000 [s]
    # average time execution of function vstbg.generate:	5.756012590000 [s]

    print("test_spatio_temporal_standard_generator execute")

    # generating parameters
    parameters = {
        "time_frames_number": 10,
        "area": 1000.0,
        "cell_size": 5.0,
        "n_base": 10,
        "lambda_1": 5.0,
        "lambda_2": 1000.0,
        "m_clumpy": 2,
        "m_overlap": 3,
        "ncfr": 0.4,
        "ncfn": 0.8,
        "ndf": 5,
        "ndfn": 5000,
        "random_seed": 0
    }

    loops_number = 1
    save_stdout = sys.stdout
    sys.stdout = open('trash.txt', 'w')
    start = timer()
    for _ in range(loops_number):
        istbg.generate(**parameters)
    end = timer()
    sys.stdout = save_stdout
    print("average time execution of function istbg.generate:\t%.12f [s]" % ((end - start) / loops_number))

    loops_number = 10
    save_stdout = sys.stdout
    sys.stdout = open('trash.txt', 'w')
    start = timer()
    for _ in range(loops_number):
        vstbg.generate(**parameters)
    end = timer()
    sys.stdout = save_stdout
    print("average time execution of function vstbg.generate:\t%.12f [s]" % ((end - start) / loops_number))


def main():
    test_spatio_temporal_standard_generator()


if __name__ == "__main__":
    main()
