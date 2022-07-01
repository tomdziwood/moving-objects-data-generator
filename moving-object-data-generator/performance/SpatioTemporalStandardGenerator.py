import sys

from timeit import default_timer as timer
from scripts.iterative import SpatioTemporalStandardGenerator as istsg
from scripts.vectorized import SpatioTemporalStandardGenerator as vstsg

def test_spatio_temporal_standard_generator():
    # average time execution of function istsg.generate:	47.366100700000 [s]
    # average time execution of function vstsg.generate:	5.756012590000 [s]

    print("test_spatio_temporal_standard_generator execute")

    # generating parameters
    parameters = {
        "time_frames_number": 10,
        "area": 1000,
        "cell_size": 5,
        "n_colloc": 10,
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

    loops_number = 1
    save_stdout = sys.stdout
    sys.stdout = open('trash.txt', 'w')
    start = timer()
    for _ in range(loops_number):
        istsg.generate(**parameters)
    end = timer()
    sys.stdout = save_stdout
    print("average time execution of function istsg.generate:\t%.12f [s]" % ((end - start) / loops_number))

    loops_number = 10
    save_stdout = sys.stdout
    sys.stdout = open('trash.txt', 'w')
    start = timer()
    for _ in range(loops_number):
        vstsg.generate(**parameters)
    end = timer()
    sys.stdout = save_stdout
    print("average time execution of function vstsg.generate:\t%.12f [s]" % ((end - start) / loops_number))


def main():
    test_spatio_temporal_standard_generator()


if __name__ == "__main__":
    main()