from typing import TextIO

import numpy as np
from datetime import datetime

from oop.StandardInitiation import StandardInitiation
from oop.StaticInteractionApproachInitiation import StaticInteractionApproachInitiation
from oop.TravelApproachInitiation import TravelApproachInitiation


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


def write_all_attributes_values(f: TextIO, parameters):
    parameters_names = [a for a in dir(parameters) if not a.startswith('__')]
    for parameters_name in parameters_names:
        f.write("# %s:\t%s\n" % (parameters_name, str(getattr(parameters, parameters_name))))


def write_standard_initiation_values(f: TextIO, si: StandardInitiation):
    f.write("# area_in_cell_dim:\t%s\n" % str(si.area_in_cell_dim))
    f.write("# base_collocation_lengths:\t%s\n" % str(si.base_collocation_lengths))
    f.write("# collocation_lengths:\t%s\n" % str(si.collocation_lengths))
    f.write("# collocation_features_sum:\t%s\n" % str(si.collocation_features_sum))
    f.write("# collocation_instances_counts:\t%s\n" % str(si.collocation_instances_counts))
    f.write("# collocation_features_instances_sum:\t%s\n" % str(si.collocation_features_instances_sum))
    f.write("# collocation_features_instances_counts:\t%s\n" % str(si.collocation_features_instances_counts))

    f.write("# collocation_features:\n")
    collocation_start_feature_id = 0
    for i_colloc in range(si.collocation_lengths.size):
        # get the features ids of current co-location
        collocation_features = np.arange(collocation_start_feature_id, collocation_start_feature_id + si.collocation_lengths[i_colloc])
        collocation_features[-1] += i_colloc % si.standard_parameters.m_overlap
        f.write("#\t%s\n" % str(collocation_features))

        # change starting feature of next co-location according to the m_overlap parameter value
        if (i_colloc + 1) % si.standard_parameters.m_overlap == 0:
            collocation_start_feature_id += si.collocation_lengths[i_colloc] + si.standard_parameters.m_overlap - 1

    f.write("# collocation_noise_features_sum:\t%s\n" % str(si.collocation_noise_features_sum))
    f.write("# collocation_noise_features:\t%s\n" % str(si.collocation_noise_features))
    f.write("# collocation_noise_features_instances_sum:\t%s\n" % str(si.collocation_noise_features_instances_sum))
    f.write("# collocation_noise_features_instances_counts:\t%s\n" % str(si.collocation_noise_features_instances_counts))
    f.write("# additional_noise_features:\t%s\n" % str(si.additional_noise_features))
    f.write("# additional_noise_features_instances_counts:\t%s\n" % str(si.additional_noise_features_instances_counts))
    f.write("# features_instances_sum:\t%s\n" % str(si.features_instances_sum))


class SpatioTemporalStandardWriter(SpatioTemporalWriter):

    def write_comment(self, si: StandardInitiation):
        # numpy print options - prevent line breaking
        np.set_printoptions(linewidth=np.inf)

        # write id of generator
        self.f.write("# SpatioTemporalStandardGenerator\n")
        self.f.write("#\n")

        # write values of used parameters
        self.f.write("# ---------- parameters ----------\n")
        write_all_attributes_values(self.f, si.standard_parameters)
        self.f.write("#\n")

        # write basic statistics of created features
        self.f.write("# ---------- initiated values ----------\n")
        write_standard_initiation_values(self.f, si)
        self.f.write("#\n")


class SpatioTemporalStaticInteractionApproachWriter(SpatioTemporalWriter):
    def write_comment(self, siai: StaticInteractionApproachInitiation):
        # numpy print options - prevent line breaking
        np.set_printoptions(linewidth=np.inf)

        # write id of generator
        self.f.write("# SpatioTemporalStaticInteractionApproachGenerator\n")
        self.f.write("#\n")

        # write values of used parameters
        self.f.write("# ---------- parameters ----------\n")
        write_all_attributes_values(self.f, siai.static_interaction_approach_parameters)
        self.f.write("#\n")

        # write basic statistics of created features
        self.f.write("# ---------- initiated values ----------\n")
        write_standard_initiation_values(self.f, siai)
        self.f.write("# center:\t%s\n" % str(siai.center))
        self.f.write("# time_interval:\t%s\n" % str(siai.time_interval))
        self.f.write("# approx_step_time_interval:\t%s\n" % str(siai.approx_step_time_interval))
        self.f.write("#\n")


class SpatioTemporalTravelApproachWriter(SpatioTemporalWriter):
    def write_comment(self, tai: TravelApproachInitiation):
        # numpy print options - prevent line breaking
        np.set_printoptions(linewidth=np.inf)

        # write id of generator
        self.f.write("# SpatioTemporalTravelApproachGenerator\n")
        self.f.write("#\n")

        # write values of used parameters
        self.f.write("# ---------- parameters ----------\n")
        write_all_attributes_values(self.f, tai.travel_approach_parameters)
        self.f.write("#\n")

        # write basic statistics of created features
        self.f.write("# ---------- initiated values ----------\n")
        write_standard_initiation_values(self.f, tai)
        self.f.write("# collocations_instances_global_sum:\t%d\n" % tai.collocations_instances_global_sum)
        self.f.write("# features_step_length_mean:\t%s\n" % str(tai.features_step_length_mean))
        self.f.write("# features_step_length_max:\t%s\n" % str(tai.features_step_length_max))
        self.f.write("# features_step_length_std:\t%s\n" % str(tai.features_step_length_std))
        self.f.write("# features_step_angle_range:\t%s\n" % str(tai.features_step_angle_range))
        self.f.write("# features_step_angle_std:\t%s\n" % str(tai.features_step_angle_std))
        self.f.write("#\n")
