import numpy as np


class SpatioTemporalWriter:
    def __init__(self, output_file):
        self.f = open(file=output_file, mode="w")

    def write(self, features_instances_sum, time_frame_ids, features_ids, features_instances_ids, x, y):
        fmt = '%d;%d;%d;%.6f;%.6f\n' * features_instances_sum
        data = fmt % tuple(np.column_stack(tup=(time_frame_ids, features_ids, features_instances_ids, x, y)).ravel())
        self.f.write(data)

    def close(self):
        self.f.close()