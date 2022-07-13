import numpy as np
from datetime import datetime


class SpatioTemporalWriter:
    def __init__(self, output_filename, output_filename_timestamp):
        if output_filename_timestamp:
            idx = output_filename.rfind('.')
            if idx == -1:
                output_filename += datetime.now().strftime("_%Y-%m-%d_%H%M%S.%f")
            else:
                output_filename = output_filename[:idx] + datetime.now().strftime("_%Y-%m-%d_%H%M%S.%f") + output_filename[idx:]
        self.f = open(file=output_filename, mode="w")

    def write(self, time_frame_ids, features_ids, features_instances_ids, x, y):
        fmt = '%d;%d;%d;%.6f;%.6f\n' * time_frame_ids.size
        data = fmt % tuple(np.column_stack(tup=(time_frame_ids, features_ids, features_instances_ids, x, y)).ravel())
        self.f.write(data)

    def close(self):
        self.f.close()
