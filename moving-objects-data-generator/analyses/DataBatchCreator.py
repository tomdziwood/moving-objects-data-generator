from pathlib import Path

import numpy as np

from algorithms.enums.InteractionApproachEnums import MassMethod as IaeMassMethod, VelocityMethod as IaeVelocityMethod, IdenticalFeaturesInteractionMode, DifferentFeaturesInteractionMode
from algorithms.enums.OptimalDistanceApproachEnums import MassMethod as OdaeMassMethod, VelocityMethod as OdaeVelocityMethod
from algorithms.enums.TravelApproachEnums import StepLengthMethod, StepAngleMethod
from algorithms.generator.SpatioTemporalCircularMotionApproachGenerator import SpatioTemporalCircularMotionApproachGenerator
from algorithms.generator.SpatioTemporalInteractionApproachGenerator import SpatioTemporalInteractionApproachGenerator
from algorithms.generator.SpatioTemporalOptimalDistanceApproachGenerator import SpatioTemporalOptimalDistanceApproachGenerator
from algorithms.generator.SpatioTemporalStandardGenerator import SpatioTemporalStandardGenerator
from algorithms.generator.SpatioTemporalTravelApproachGenerator import SpatioTemporalTravelApproachGenerator
from algorithms.parameters.CircularMotionApproachParameters import CircularMotionApproachParameters
from algorithms.parameters.InteractionApproachParameters import InteractionApproachParameters
from algorithms.parameters.OptimalDistanceApproachParameters import OptimalDistanceApproachParameters
from algorithms.parameters.StandardParameters import StandardParameters
from algorithms.parameters.TravelApproachParameters import TravelApproachParameters


def create_batch_01():
    print("create_batch_01")

    Path("data\\batch_01").mkdir(parents=True, exist_ok=True)

    for persistent_ratio in np.arange(0, 1.01, 0.1).round(1):
        print("persistent_ratio=%d" % persistent_ratio)

        sp = StandardParameters(
            area=100,
            cell_size=5,
            n_base=10,
            lambda_1=5,
            lambda_2=10,
            m_clumpy=1,
            m_overlap=1,
            ncfr=0,
            ncfn=0,
            ncf_proportional=False,
            ndf=5,
            ndfn=100,
            random_seed=0,
            persistent_ratio=persistent_ratio,
            spatial_prevalence_threshold=0.5,
            time_prevalence_threshold=0.5
        )

        stsg = SpatioTemporalStandardGenerator(sp=sp)
        stsg.generate(
            time_frames_number=100,
            output_filename="data\\batch_01\\SpatioTemporalStandardGenerator_output_file.txt",
            output_filename_timestamp=True
        )


def create_batch_02():
    print("create_batch_02")

    Path("data\\batch_02").mkdir(parents=True, exist_ok=True)

    # SpatioTemporalStandardGenerator
    sp = StandardParameters(
        area=200,
        cell_size=5,
        n_base=10,
        lambda_1=5,
        lambda_2=10,
        m_clumpy=1,
        m_overlap=1,
        ncfr=0,
        ncfn=0,
        ncf_proportional=False,
        ndf=5,
        ndfn=50,
        random_seed=0,
        persistent_ratio=0.5,
        spatial_prevalence_threshold=0.5,
        time_prevalence_threshold=0.5
    )

    stsg = SpatioTemporalStandardGenerator(sp=sp)
    stsg.generate(
        time_frames_number=500,
        output_filename="data\\batch_02\\SpatioTemporalStandardGenerator_output_file.txt",
        output_filename_timestamp=False
    )

    # SpatioTemporalInteractionApproachGenerator
    iap = InteractionApproachParameters(
        area=200,
        cell_size=5,
        n_base=10,
        lambda_1=5,
        lambda_2=10,
        m_clumpy=1,
        m_overlap=1,
        ncfr=0,
        ncfn=0,
        ncf_proportional=False,
        ndf=5,
        ndfn=100,
        random_seed=0,
        spatial_prevalent_ratio=0.5,
        spatial_prevalence_threshold=0.5,
        time_unit=25,
        distance_unit=1.0,
        approx_steps_number=2,
        k_force=1000,
        force_limit=100.0,
        velocity_limit=10.0,
        faraway_limit_ratio=np.sqrt(2) / 2,
        mass_method=IaeMassMethod.CONSTANT,
        mass_mean=1.0,
        mass_normal_std_ratio=1 / 5,
        velocity_method=IaeVelocityMethod.CONSTANT,
        velocity_mean=0.0,
        identical_features_interaction_mode=IdenticalFeaturesInteractionMode.REPEL,
        different_features_interaction_mode=DifferentFeaturesInteractionMode.COLLOCATION_ATTRACT_OTHER_NEUTRAL
    )

    stiag = SpatioTemporalInteractionApproachGenerator(iap=iap)
    stiag.generate(
        time_frames_number=500,
        output_filename="data\\batch_02\\SpatioTemporalInteractionApproachGenerator_output_file.txt",
        output_filename_timestamp=False
    )

    # SpatioTemporalOptimalDistanceApproachGenerator
    odap = OptimalDistanceApproachParameters(
        area=200,
        cell_size=5,
        n_base=10,
        lambda_1=5,
        lambda_2=10,
        m_clumpy=1,
        m_overlap=1,
        ncfr=0,
        ncfn=0,
        ncf_proportional=False,
        ndf=5,
        ndfn=50,
        random_seed=0,
        spatial_prevalent_ratio=0.5,
        spatial_prevalence_threshold=0.5,
        time_unit=25,
        approx_steps_number=2,
        k_optimal_distance=2.0,
        k_force=100.0,
        force_limit=10.0,
        velocity_limit=3.0,
        faraway_limit_ratio=np.sqrt(2) / 2,
        mass_method=OdaeMassMethod.CONSTANT,
        mass_mean=1.0,
        mass_normal_std_ratio=1 / 5,
        velocity_method=OdaeVelocityMethod.CONSTANT,
        velocity_mean=0.0
    )

    stodag = SpatioTemporalOptimalDistanceApproachGenerator(odap=odap)
    stodag.generate(
        time_frames_number=500,
        output_filename="data\\batch_02\\SpatioTemporalOptimalDistanceApproachGenerator_output_file.txt",
        output_filename_timestamp=False
    )

    # SpatioTemporalCircularMotionApproachGenerator
    cmap = CircularMotionApproachParameters(
        area=200,
        cell_size=5,
        n_base=10,
        lambda_1=5,
        lambda_2=10,
        m_clumpy=1,
        m_overlap=1,
        ncfr=0,
        ncfn=0,
        ncf_proportional=False,
        ndf=5,
        ndfn=50,
        random_seed=0,
        spatial_prevalent_ratio=0.5,
        spatial_prevalence_threshold=0.5,
        circle_chain_size=5,
        omega_min=2 * np.pi / 200,
        omega_max=2 * np.pi / 50,
        circle_r_min=4.0,
        circle_r_max=40.0,
        center_noise_displacement=2.0
    )

    stcmag = SpatioTemporalCircularMotionApproachGenerator(cmap=cmap)
    stcmag.generate(
        time_frames_number=500,
        output_filename="data\\batch_02\\SpatioTemporalCircularMotionApproachGenerator_output_file.txt",
        output_filename_timestamp=False
    )

    # SpatioTemporalTravelApproachGenerator
    tap = TravelApproachParameters(
        area=200,
        cell_size=5,
        n_base=10,
        lambda_1=5,
        lambda_2=10,
        m_clumpy=1,
        m_overlap=1,
        ncfr=0,
        ncfn=0,
        ncf_proportional=False,
        ndf=5,
        ndfn=50,
        random_seed=0,
        spatial_prevalent_ratio=0.5,
        spatial_prevalence_threshold=0.5,
        step_length_mean=5.0,
        step_length_method=StepLengthMethod.UNIFORM,
        step_length_uniform_low_to_mean_ratio=0.75,
        step_length_normal_std_ratio=3,
        step_angle_range_mean=np.pi / 4,
        step_angle_range_limit=np.pi / 2,
        step_angle_method=StepAngleMethod.UNIFORM,
        step_angle_normal_std_ratio=1 / 3,
        waiting_time_frames=200
    )

    sttag = SpatioTemporalTravelApproachGenerator(tap=tap)
    sttag.generate(
        time_frames_number=500,
        output_filename="data\\batch_02\\SpatioTemporalTravelApproachGenerator_output_file.txt",
        output_filename_timestamp=False
    )


def create_batch_03():
    print("create_batch_03")

    Path("data\\batch_03").mkdir(parents=True, exist_ok=True)

    # # SpatioTemporalStandardGenerator
    # sp = StandardParameters(
    #     area=200,
    #     cell_size=5,
    #     n_base=10,
    #     lambda_1=5,
    #     lambda_2=5,
    #     m_clumpy=1,
    #     m_overlap=1,
    #     ncfr=0,
    #     ncfn=0,
    #     ncf_proportional=False,
    #     ndf=5,
    #     ndfn=25,
    #     random_seed=0,
    #     persistent_ratio=0.5,
    #     spatial_prevalence_threshold=0.5,
    #     time_prevalence_threshold=0.5
    # )
    #
    # stsg = SpatioTemporalStandardGenerator(sp=sp)
    # stsg.generate(
    #     time_frames_number=500,
    #     output_filename="data\\batch_03\\SpatioTemporalStandardGenerator_output_file.txt",
    #     output_filename_timestamp=False
    # )

    # SpatioTemporalInteractionApproachGenerator
    iap = InteractionApproachParameters(
        area=200,
        cell_size=5,
        n_base=10,
        lambda_1=5,
        lambda_2=5,
        m_clumpy=1,
        m_overlap=1,
        ncfr=0,
        ncfn=0,
        ncf_proportional=False,
        ndf=5,
        ndfn=25,
        random_seed=0,
        spatial_prevalent_ratio=0.5,
        spatial_prevalence_threshold=0.5,
        time_unit=25,
        distance_unit=1.0,
        approx_steps_number=2,
        k_force=1000,
        force_limit=100.0,
        velocity_limit=10.0,
        faraway_limit_ratio=np.sqrt(2) / 2,
        mass_method=IaeMassMethod.CONSTANT,
        mass_mean=1.0,
        mass_normal_std_ratio=1 / 5,
        velocity_method=IaeVelocityMethod.CONSTANT,
        velocity_mean=0.0,
        identical_features_interaction_mode=IdenticalFeaturesInteractionMode.REPEL,
        different_features_interaction_mode=DifferentFeaturesInteractionMode.COLLOCATION_ATTRACT_OTHER_NEUTRAL
    )

    stiag = SpatioTemporalInteractionApproachGenerator(iap=iap)
    stiag.generate(
        time_frames_number=500,
        output_filename="data\\batch_03\\SpatioTemporalInteractionApproachGenerator_output_file.txt",
        output_filename_timestamp=False
    )

    # # SpatioTemporalOptimalDistanceApproachGenerator
    # odap = OptimalDistanceApproachParameters(
    #     area=200,
    #     cell_size=5,
    #     n_base=10,
    #     lambda_1=5,
    #     lambda_2=5,
    #     m_clumpy=1,
    #     m_overlap=1,
    #     ncfr=0,
    #     ncfn=0,
    #     ncf_proportional=False,
    #     ndf=5,
    #     ndfn=25,
    #     random_seed=0,
    #     spatial_prevalent_ratio=0.5,
    #     spatial_prevalence_threshold=0.5,
    #     time_unit=25,
    #     approx_steps_number=2,
    #     k_optimal_distance=2.0,
    #     k_force=100.0,
    #     force_limit=10.0,
    #     velocity_limit=3.0,
    #     faraway_limit_ratio=np.sqrt(2) / 2,
    #     mass_method=OdaeMassMethod.CONSTANT,
    #     mass_mean=1.0,
    #     mass_normal_std_ratio=1 / 5,
    #     velocity_method=OdaeVelocityMethod.CONSTANT,
    #     velocity_mean=0.0
    # )
    #
    # stodag = SpatioTemporalOptimalDistanceApproachGenerator(odap=odap)
    # stodag.generate(
    #     time_frames_number=500,
    #     output_filename="data\\batch_03\\SpatioTemporalOptimalDistanceApproachGenerator_output_file.txt",
    #     output_filename_timestamp=False
    # )
    #
    # # SpatioTemporalCircularMotionApproachGenerator
    # cmap = CircularMotionApproachParameters(
    #     area=200,
    #     cell_size=5,
    #     n_base=10,
    #     lambda_1=5,
    #     lambda_2=5,
    #     m_clumpy=1,
    #     m_overlap=1,
    #     ncfr=0,
    #     ncfn=0,
    #     ncf_proportional=False,
    #     ndf=5,
    #     ndfn=25,
    #     random_seed=0,
    #     spatial_prevalent_ratio=0.5,
    #     spatial_prevalence_threshold=0.5,
    #     circle_chain_size=5,
    #     omega_min=2 * np.pi / 200,
    #     omega_max=2 * np.pi / 50,
    #     circle_r_min=4.0,
    #     circle_r_max=40.0,
    #     center_noise_displacement=2.0
    # )
    #
    # stcmag = SpatioTemporalCircularMotionApproachGenerator(cmap=cmap)
    # stcmag.generate(
    #     time_frames_number=500,
    #     output_filename="data\\batch_03\\SpatioTemporalCircularMotionApproachGenerator_output_file.txt",
    #     output_filename_timestamp=False
    # )

    # # SpatioTemporalTravelApproachGenerator
    # tap = TravelApproachParameters(
    #     area=200,
    #     cell_size=5,
    #     n_base=10,
    #     lambda_1=5,
    #     lambda_2=5,
    #     m_clumpy=1,
    #     m_overlap=1,
    #     ncfr=0,
    #     ncfn=0,
    #     ncf_proportional=False,
    #     ndf=5,
    #     ndfn=100,
    #     random_seed=0,
    #     spatial_prevalent_ratio=1.0,
    #     spatial_prevalence_threshold=1.0,
    #     step_length_mean=5.0,
    #     step_length_method=StepLengthMethod.UNIFORM,
    #     step_length_uniform_low_to_mean_ratio=0.75,
    #     step_length_normal_std_ratio=3,
    #     step_angle_range_mean=np.pi / 4,
    #     step_angle_range_limit=np.pi / 2,
    #     step_angle_method=StepAngleMethod.UNIFORM,
    #     step_angle_normal_std_ratio=1 / 3,
    #     waiting_time_frames=200
    # )
    #
    # sttag = SpatioTemporalTravelApproachGenerator(tap=tap)
    # sttag.generate(
    #     time_frames_number=500,
    #     output_filename="data\\batch_03\\SpatioTemporalTravelApproachGenerator_output_file.txt",
    #     output_filename_timestamp=False
    # )


def main():
    # create_batch_01()
    create_batch_02()
    # create_batch_03()


if __name__ == "__main__":
    main()