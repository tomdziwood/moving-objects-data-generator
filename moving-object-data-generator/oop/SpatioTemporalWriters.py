from typing import TextIO

import numpy as np
from datetime import datetime

from oop.BasicInitiation import BasicInitiation
from oop.BasicParameters import BasicParameters
from oop.StandardInitiation import StandardInitiation
from oop.StaticInteractionApproachInitiation import StaticInteractionApproachInitiation
from oop.TravelApproachInitiation import TravelApproachInitiation


class SpatioTemporalWriter:
    """
    Basic writer type class of spatio-temporal data.

    Attributes
    ----------
    f : TextIO
        The output file of spatio-temporal data.
    """

    def __init__(self, output_filename: str, output_filename_timestamp: bool):
        """
        Initiate required data to generate spatio-temporal data in each time frame.

        Parameters
        ----------
        output_filename : str
            The filename to which output will be written.

        output_filename_timestamp : bool
            When ``True``, the filename has added unique string which is created based on the current timestamp.
            It helps to automatically recognize different output of generator.
        """

        if output_filename_timestamp:
            idx = output_filename.rfind('.')
            if idx == -1:
                output_filename += datetime.now().strftime("_%Y-%m-%d_%H%M%S.%f")
            else:
                output_filename = output_filename[:idx] + datetime.now().strftime("_%Y-%m-%d_%H%M%S.%f") + output_filename[idx:]
        self.f = open(file=output_filename, mode="w")

    def write(
            self,
            time_frame_ids: np.ndarray,
            features_ids: np.ndarray,
            features_instances_ids: np.ndarray,
            x: np.ndarray,
            y: np.ndarray):
        """
        Write spatio-temporal data to the output. The single record contains five numbers separated with semicolon. These five types of values are passed
        as the params of the method - five vectors of equal length.

        Parameters
        ----------
        time_frame_ids : np.ndarray
            The vector of time frame ids of consecutive records.

        features_ids : np.ndarray
            The vector of features' types ids of consecutive records.

        features_instances_ids : np.ndarray
            The vector of features' instances ids of consecutive records.

        x : np.ndarray
            The vector of ``x`` coordinates of consecutive records.

        y : np.ndarray
            The vector of ``y`` coordinates of consecutive records.
        """

        fmt = '%d;%d;%d;%.6f;%.6f\n' * time_frame_ids.size
        data = fmt % tuple(np.column_stack(tup=(time_frame_ids, features_ids, features_instances_ids, x, y)).ravel())
        self.f.write(data)

    def close(self):
        """
        Close output file.
        """

        self.f.close()


def write_all_attributes_values(f: TextIO, parameters: BasicParameters):
    """
    Write all parameters of spatio-temporal data generator to the given file as a comment.

    Parameters
    ----------
    f : TextIO
        The output file of spatio-temporal data.

    parameters : BasicParameters
        The object which represents set of parameters used by the generator.
    """

    parameters_names = [a for a in dir(parameters) if not a.startswith('__')]
    for parameters_name in parameters_names:
        f.write("# %s:\t%s\n" % (parameters_name, str(getattr(parameters, parameters_name))))


def write_basic_initiation_values(f: TextIO, bi: BasicInitiation):
    """
    Write all crucial data initiated by `BasicInitiation` object.

    Parameters
    ----------
    f : TextIO
        The output file of spatio-temporal data.

    bi : BasicInitiation
        The object of a `BasicInitiation` class, which stores all initial data, which is required to generate spatio-temporal data in each time frame.
    """

    f.write("# area_in_cell_dim:\t%s\n" % str(bi.area_in_cell_dim))
    f.write("# base_collocation_lengths:\t%s\n" % str(bi.base_collocation_lengths))
    f.write("# collocation_lengths:\t%s\n" % str(bi.collocation_lengths))
    f.write("# collocation_features_sum:\t%s\n" % str(bi.collocation_features_sum))
    f.write("# collocation_instances_counts:\t%s\n" % str(bi.collocation_instances_counts))
    f.write("# collocations_instances_sum:\t%s\n" % str(bi.collocations_instances_sum))
    f.write("# collocation_features_instances_sum:\t%s\n" % str(bi.collocation_features_instances_sum))
    f.write("# collocation_features_instances_counts:\t%s\n" % str(bi.collocation_features_instances_counts))

    f.write("# collocation_features:\n")
    collocation_start_feature_id = 0
    for i_colloc in range(bi.collocations_sum):
        # get the features ids of current co-location
        collocation_features = np.arange(collocation_start_feature_id, collocation_start_feature_id + bi.collocation_lengths[i_colloc])
        collocation_features[-1] += i_colloc % bi.basic_parameters.m_overlap
        f.write("#\t%s\n" % str(collocation_features))

        # change starting feature of next co-location according to the m_overlap parameter value
        if (i_colloc + 1) % bi.basic_parameters.m_overlap == 0:
            collocation_start_feature_id += bi.collocation_lengths[i_colloc] + bi.basic_parameters.m_overlap - 1

    f.write("# collocation_noise_features_sum:\t%s\n" % str(bi.collocation_noise_features_sum))
    f.write("# collocation_noise_features:\t%s\n" % str(bi.collocation_noise_features))
    f.write("# collocation_noise_features_instances_sum:\t%s\n" % str(bi.collocation_noise_features_instances_sum))
    f.write("# collocation_noise_features_instances_counts:\t%s\n" % str(bi.collocation_noise_features_instances_counts))
    f.write("# additional_noise_features:\t%s\n" % str(bi.additional_noise_features))
    f.write("# additional_noise_features_instances_counts:\t%s\n" % str(bi.additional_noise_features_instances_counts))
    f.write("# features_instances_sum:\t%s\n" % str(bi.features_instances_sum))
    f.write("# collocations_instances_global_sum:\t%d\n" % bi.collocations_instances_global_sum)
    f.write("# collocations_clumpy_instances_global_sum:\t%s\n" % str(bi.collocations_clumpy_instances_global_sum))


class SpatioTemporalBasicWriter(SpatioTemporalWriter):
    """
    Specialized type of writer. It allows writing comment of spatio-temporal data generated by the `SpatioTemporalBasicGenerator` class of a generator.
    """

    def write_comment(self, bi: BasicInitiation):
        """
        Write comment of spatio-temporal data generated by the `SpatioTemporalBasicGenerator` class of a generator.
        Comment contains all data of the used parameters and all crucial data initiated by the `BasicInitiation` object.

        Parameters
        ----------
        bi : BasicInitiation
            The object of a `BasicInitiation` class, which stores all initial data, which is required to generate spatio-temporal data in each time frame.
        """

        # numpy print options - prevent line breaking
        np.set_printoptions(linewidth=np.inf)

        # write id of a generator
        self.f.write("# SpatioTemporalBasicGenerator\n")
        self.f.write("#\n")

        # write values of used parameters
        self.f.write("# ---------- parameters ----------\n")
        write_all_attributes_values(self.f, bi.basic_parameters)
        self.f.write("#\n")

        # write basic statistics of created features
        self.f.write("# ---------- initiated values ----------\n")
        write_basic_initiation_values(self.f, bi)
        self.f.write("#\n")


class SpatioTemporalStandardWriter(SpatioTemporalWriter):
    """
    Specialized type of writer. It allows writing comment of spatio-temporal data generated by the `SpatioTemporalStandardGenerator` class of a generator.
    """

    def write_comment(self, si: StandardInitiation):
        """
        Write comment of spatio-temporal data generated by the `SpatioTemporalStandardGenerator` class of a generator.
        Comment contains all data of the used parameters and all crucial data initiated by the `StandardInitiation` object.

        Parameters
        ----------
        si : StandardInitiation
            The object of a `StandardInitiation` class, which stores all initial data, which is required to generate spatio-temporal data in each time frame.
        """

        # numpy print options - prevent line breaking
        np.set_printoptions(linewidth=np.inf)

        # write id of the generator
        self.f.write("# SpatioTemporalStandardGenerator\n")
        self.f.write("#\n")

        # write values of used parameters
        self.f.write("# ---------- parameters ----------\n")
        write_all_attributes_values(self.f, si.standard_parameters)
        self.f.write("#\n")

        # write basic statistics of created features
        self.f.write("# ---------- initiated values ----------\n")
        write_basic_initiation_values(self.f, si)
        self.f.write("# persistent_collocations_sum:\t%s\n" % str(si.persistent_collocations_sum))
        self.f.write("# persistent_collocations_ids:\t%s\n" % str(si.persistent_collocations_ids))
        self.f.write("# transient_collocations_sum:\t%s\n" % str(si.transient_collocations_sum))
        self.f.write("# transient_collocations_ids:\t%s\n" % str(si.transient_collocations_ids))
        self.f.write("#\n")


class SpatioTemporalStaticInteractionApproachWriter(SpatioTemporalWriter):
    """
    Specialized type of writer. It allows writing comment of spatio-temporal data generated by the `SpatioTemporalStaticInteractionApproachGenerator` class of a generator.
    """

    def write_comment(self, siai: StaticInteractionApproachInitiation):
        """
        Write comment of spatio-temporal data generated by the `SpatioTemporalStaticInteractionApproachGenerator` class of a generator.
        Comment contains all data of the used parameters and all crucial data initiated by the `StaticInteractionApproachInitiation` object.

        Parameters
        ----------
        siai : StaticInteractionApproachInitiation
            The object of a `StaticInteractionApproachInitiation` class, which stores all initial data, which is required to generate spatio-temporal data in each time frame.
        """

        # numpy print options - prevent line breaking
        np.set_printoptions(linewidth=np.inf)

        # write id of the generator
        self.f.write("# SpatioTemporalStaticInteractionApproachGenerator\n")
        self.f.write("#\n")

        # write values of used parameters
        self.f.write("# ---------- parameters ----------\n")
        write_all_attributes_values(self.f, siai.static_interaction_approach_parameters)
        self.f.write("#\n")

        # write basic statistics of created features
        self.f.write("# ---------- initiated values ----------\n")
        write_basic_initiation_values(self.f, siai)
        self.f.write("# center:\t%s\n" % str(siai.center))
        self.f.write("# time_interval:\t%s\n" % str(siai.time_interval))
        self.f.write("# approx_step_time_interval:\t%s\n" % str(siai.approx_step_time_interval))
        self.f.write("#\n")


class SpatioTemporalTravelApproachWriter(SpatioTemporalWriter):
    """
    Specialized type of writer. It allows writing comment of spatio-temporal data generated by the `SpatioTemporalTravelApproachGenerator` class of a generator.
    """

    def write_comment(self, tai: TravelApproachInitiation):
        """
        Write comment of spatio-temporal data generated by the `SpatioTemporalTravelApproachGenerator` class of a generator.
        Comment contains all data of the used parameters and all crucial data initiated by the `TravelApproachInitiation` object.

        Parameters
        ----------
        tai : TravelApproachInitiation
            The object of a `TravelApproachInitiation` class, which stores all initial data, which is required to generate spatio-temporal data in each time frame.
        """

        # numpy print options - prevent line breaking
        np.set_printoptions(linewidth=np.inf)

        # write id of the generator
        self.f.write("# SpatioTemporalTravelApproachGenerator\n")
        self.f.write("#\n")

        # write values of used parameters
        self.f.write("# ---------- parameters ----------\n")
        write_all_attributes_values(self.f, tai.travel_approach_parameters)
        self.f.write("#\n")

        # write basic statistics of created features
        self.f.write("# ---------- initiated values ----------\n")
        write_basic_initiation_values(self.f, tai)
        self.f.write("# features_step_length_mean:\t%s\n" % str(tai.features_step_length_mean))
        self.f.write("# features_step_length_uniform_min:\t%s\n" % str(tai.features_step_length_uniform_max))
        self.f.write("# features_step_length_uniform_max:\t%s\n" % str(tai.features_step_length_uniform_max))
        self.f.write("# features_step_length_normal_std:\t%s\n" % str(tai.features_step_length_normal_std))
        self.f.write("# features_step_angle_range:\t%s\n" % str(tai.features_step_angle_range))
        self.f.write("# features_step_angle_normal_std:\t%s\n" % str(tai.features_step_angle_normal_std))
        self.f.write("#\n")
