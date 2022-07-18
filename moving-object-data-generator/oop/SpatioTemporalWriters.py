import numpy as np
from datetime import datetime

from oop.GravitationApproachInitiation import GravitationApproachInitiation
from oop.StandardInitiation import StandardInitiation
from oop.StaticInteractionApproachInitiation import StaticInteractionApproachInitiation


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


class SpatioTemporalStandardWriter(SpatioTemporalWriter):
    def write_comment(self, si: StandardInitiation):
        # numpy print options - prevent line breaking
        np.set_printoptions(linewidth=np.inf)

        # write id of generator
        self.f.write("# SpatioTemporalStandardGenerator\n")
        self.f.write("#\n")

        # write values of used parameters
        self.f.write("# ---------- parameters ----------\n")
        parameters_names = [a for a in dir(si.standard_parameters) if not a.startswith('__')]
        for parameters_name in parameters_names:
            self.f.write("# %s:\t%s\n" % (parameters_name, str(getattr(si.standard_parameters, parameters_name))))
        self.f.write("#\n")

        # write basic statistics of created features
        self.f.write("# ---------- initiated values ----------\n")
        self.f.write("# area_in_cell_dim:\t%s\n" % str(si.area_in_cell_dim))
        self.f.write("# base_collocation_lengths:\t%s\n" % str(si.base_collocation_lengths))
        self.f.write("# collocation_lengths:\t%s\n" % str(si.collocation_lengths))
        self.f.write("# collocation_features_sum:\t%s\n" % str(si.collocation_features_sum))
        self.f.write("# collocation_instances_counts:\t%s\n" % str(si.collocation_instances_counts))
        self.f.write("# collocation_features_instances_sum:\t%s\n" % str(si.collocation_features_instances_sum))
        self.f.write("# collocation_features_instances_counts:\t%s\n" % str(si.collocation_features_instances_counts))

        self.f.write("# collocation_features:\n")
        collocation_start_feature_id = 0
        for i_colloc in range(si.collocation_lengths.size):
            # get the features ids of current co-location
            collocation_features = np.arange(collocation_start_feature_id, collocation_start_feature_id + si.collocation_lengths[i_colloc])
            collocation_features[-1] += i_colloc % si.standard_parameters.m_overlap
            self.f.write("#\t%s\n" % str(collocation_features))

            # change starting feature of next co-location according to the m_overlap parameter value
            if (i_colloc + 1) % si.standard_parameters.m_overlap == 0:
                collocation_start_feature_id += si.collocation_lengths[i_colloc] + si.standard_parameters.m_overlap - 1

        self.f.write("# collocation_noise_features_sum:\t%s\n" % str(si.collocation_noise_features_sum))
        self.f.write("# collocation_noise_features:\t%s\n" % str(si.collocation_noise_features))
        self.f.write("# collocation_noise_features_instances_sum:\t%s\n" % str(si.collocation_noise_features_instances_sum))
        self.f.write("# collocation_noise_features_instances_counts:\t%s\n" % str(si.collocation_noise_features_instances_counts))
        self.f.write("# additional_noise_features:\t%s\n" % str(si.additional_noise_features))
        self.f.write("# additional_noise_features_instances_counts:\t%s\n" % str(si.additional_noise_features_instances_counts))
        self.f.write("# features_instances_sum:\t%s\n" % str(si.features_instances_sum))
        self.f.write("#\n")


class SpatioTemporalGravitationApproachWriter(SpatioTemporalWriter):
    def write_comment(self, gai: GravitationApproachInitiation):
        # numpy print options - prevent line breaking
        np.set_printoptions(linewidth=np.inf)

        # write id of generator
        self.f.write("# SpatioTemporalGravitationApproachGenerator\n")
        self.f.write("#\n")

        # write values of used parameters
        self.f.write("# ---------- parameters ----------\n")
        parameters_names = [a for a in dir(gai.gravitation_approach_parameters) if not a.startswith('__')]
        for parameters_name in parameters_names:
            self.f.write("# %s:\t%s\n" % (parameters_name, str(getattr(gai.gravitation_approach_parameters, parameters_name))))
        self.f.write("#\n")

        # write basic statistics of created features
        self.f.write("# ---------- initiated values ----------\n")
        self.f.write("# area_in_cell_dim:\t%s\n" % str(gai.area_in_cell_dim))
        self.f.write("# base_collocation_lengths:\t%s\n" % str(gai.base_collocation_lengths))
        self.f.write("# collocation_lengths:\t%s\n" % str(gai.collocation_lengths))
        self.f.write("# collocation_features_sum:\t%s\n" % str(gai.collocation_features_sum))
        self.f.write("# collocation_instances_counts:\t%s\n" % str(gai.collocation_instances_counts))
        self.f.write("# collocation_features_instances_sum:\t%s\n" % str(gai.collocation_features_instances_sum))
        self.f.write("# collocation_features_instances_counts:\t%s\n" % str(gai.collocation_features_instances_counts))

        self.f.write("# collocation_features:\n")
        collocation_start_feature_id = 0
        for i_colloc in range(gai.collocation_lengths.size):
            # get the features ids of current co-location
            collocation_features = np.arange(collocation_start_feature_id, collocation_start_feature_id + gai.collocation_lengths[i_colloc])
            collocation_features[-1] += i_colloc % gai.gravitation_approach_parameters.m_overlap
            self.f.write("#\t%s\n" % str(collocation_features))

            # change starting feature of next co-location according to the m_overlap parameter value
            if (i_colloc + 1) % gai.gravitation_approach_parameters.m_overlap == 0:
                collocation_start_feature_id += gai.collocation_lengths[i_colloc] + gai.gravitation_approach_parameters.m_overlap - 1

        self.f.write("# collocation_noise_features_sum:\t%s\n" % str(gai.collocation_noise_features_sum))
        self.f.write("# collocation_noise_features:\t%s\n" % str(gai.collocation_noise_features))
        self.f.write("# collocation_noise_features_instances_sum:\t%s\n" % str(gai.collocation_noise_features_instances_sum))
        self.f.write("# collocation_noise_features_instances_counts:\t%s\n" % str(gai.collocation_noise_features_instances_counts))
        self.f.write("# additional_noise_features:\t%s\n" % str(gai.additional_noise_features))
        self.f.write("# additional_noise_features_instances_counts:\t%s\n" % str(gai.additional_noise_features_instances_counts))
        self.f.write("# features_instances_sum:\t%s\n" % str(gai.features_instances_sum))
        self.f.write("# center:\t%s\n" % str(gai.center))
        self.f.write("# time_interval:\t%s\n" % str(gai.time_interval))
        self.f.write("# approx_step_time_interval:\t%s\n" % str(gai.approx_step_time_interval))
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
        parameters_names = [a for a in dir(siai.static_interaction_approach_parameters) if not a.startswith('__')]
        for parameters_name in parameters_names:
            self.f.write("# %s:\t%s\n" % (parameters_name, str(getattr(siai.static_interaction_approach_parameters, parameters_name))))
        self.f.write("#\n")

        # write basic statistics of created features
        self.f.write("# ---------- initiated values ----------\n")
        self.f.write("# area_in_cell_dim:\t%s\n" % str(siai.area_in_cell_dim))
        self.f.write("# base_collocation_lengths:\t%s\n" % str(siai.base_collocation_lengths))
        self.f.write("# collocation_lengths:\t%s\n" % str(siai.collocation_lengths))
        self.f.write("# collocation_features_sum:\t%s\n" % str(siai.collocation_features_sum))
        self.f.write("# collocation_instances_counts:\t%s\n" % str(siai.collocation_instances_counts))
        self.f.write("# collocation_features_instances_sum:\t%s\n" % str(siai.collocation_features_instances_sum))
        self.f.write("# collocation_features_instances_counts:\t%s\n" % str(siai.collocation_features_instances_counts))

        self.f.write("# collocation_features:\n")
        collocation_start_feature_id = 0
        for i_colloc in range(siai.collocation_lengths.size):
            # get the features ids of current co-location
            collocation_features = np.arange(collocation_start_feature_id, collocation_start_feature_id + siai.collocation_lengths[i_colloc])
            collocation_features[-1] += i_colloc % siai.static_interaction_approach_parameters.m_overlap
            self.f.write("#\t%s\n" % str(collocation_features))

            # change starting feature of next co-location according to the m_overlap parameter value
            if (i_colloc + 1) % siai.static_interaction_approach_parameters.m_overlap == 0:
                collocation_start_feature_id += siai.collocation_lengths[i_colloc] + siai.static_interaction_approach_parameters.m_overlap - 1

        self.f.write("# collocation_noise_features_sum:\t%s\n" % str(siai.collocation_noise_features_sum))
        self.f.write("# collocation_noise_features:\t%s\n" % str(siai.collocation_noise_features))
        self.f.write("# collocation_noise_features_instances_sum:\t%s\n" % str(siai.collocation_noise_features_instances_sum))
        self.f.write("# collocation_noise_features_instances_counts:\t%s\n" % str(siai.collocation_noise_features_instances_counts))
        self.f.write("# additional_noise_features:\t%s\n" % str(siai.additional_noise_features))
        self.f.write("# additional_noise_features_instances_counts:\t%s\n" % str(siai.additional_noise_features_instances_counts))
        self.f.write("# features_instances_sum:\t%s\n" % str(siai.features_instances_sum))
        self.f.write("# center:\t%s\n" % str(siai.center))
        self.f.write("# time_interval:\t%s\n" % str(siai.time_interval))
        self.f.write("# approx_step_time_interval:\t%s\n" % str(siai.approx_step_time_interval))
        self.f.write("#\n")
