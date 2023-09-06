import os
import sys
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
            area=100.0,
            cell_size=5.0,
            n_base=10,
            lambda_1=5.0,
            lambda_2=10.0,
            m_clumpy=1,
            m_overlap=1,
            ncfr=0.0,
            ncfn=0.0,
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
        area=200.0,
        cell_size=5.0,
        n_base=10,
        lambda_1=5.0,
        lambda_2=10.0,
        m_clumpy=1,
        m_overlap=1,
        ncfr=0.0,
        ncfn=0.0,
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
        area=200.0,
        cell_size=5.0,
        n_base=10,
        lambda_1=5.0,
        lambda_2=10.0,
        m_clumpy=1,
        m_overlap=1,
        ncfr=0.0,
        ncfn=0.0,
        ncf_proportional=False,
        ndf=5,
        ndfn=100,
        random_seed=0,
        spatial_prevalent_ratio=0.5,
        spatial_prevalence_threshold=0.5,
        time_unit=25.0,
        distance_unit=1.0,
        approx_steps_number=2,
        k_force=1000.0,
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
        area=200.0,
        cell_size=5.0,
        n_base=10,
        lambda_1=5.0,
        lambda_2=10.0,
        m_clumpy=1,
        m_overlap=1,
        ncfr=0.0,
        ncfn=0.0,
        ncf_proportional=False,
        ndf=5,
        ndfn=50,
        random_seed=0,
        spatial_prevalent_ratio=0.5,
        spatial_prevalence_threshold=0.5,
        time_unit=25.0,
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
        area=200.0,
        cell_size=5.0,
        n_base=10,
        lambda_1=5.0,
        lambda_2=10.0,
        m_clumpy=1,
        m_overlap=1,
        ncfr=0.0,
        ncfn=0.0,
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
        area=200.0,
        cell_size=5.0,
        n_base=10,
        lambda_1=5.0,
        lambda_2=10.0,
        m_clumpy=1,
        m_overlap=1,
        ncfr=0.0,
        ncfn=0.0,
        ncf_proportional=False,
        ndf=5,
        ndfn=50,
        random_seed=0,
        spatial_prevalent_ratio=0.5,
        spatial_prevalence_threshold=0.5,
        step_length_mean=5.0,
        step_length_method=StepLengthMethod.UNIFORM,
        step_length_uniform_low_to_mean_ratio=0.75,
        step_length_normal_std_ratio=3.0,
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
    #     area=200.0,
    #     cell_size=5.0,
    #     n_base=10,
    #     lambda_1=5.0,
    #     lambda_2=5.0,
    #     m_clumpy=1,
    #     m_overlap=1,
    #     ncfr=0.0,
    #     ncfn=0.0,
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
        area=200.0,
        cell_size=5.0,
        n_base=10,
        lambda_1=5.0,
        lambda_2=5.0,
        m_clumpy=1,
        m_overlap=1,
        ncfr=0.0,
        ncfn=0.0,
        ncf_proportional=False,
        ndf=5,
        ndfn=25,
        random_seed=0,
        spatial_prevalent_ratio=0.5,
        spatial_prevalence_threshold=0.5,
        time_unit=25.0,
        distance_unit=1.0,
        approx_steps_number=2,
        k_force=1000.0,
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
    #     area=200.0,
    #     cell_size=5.0,
    #     n_base=10,
    #     lambda_1=5.0,
    #     lambda_2=5.0,
    #     m_clumpy=1,
    #     m_overlap=1,
    #     ncfr=0.0,
    #     ncfn=0.0,
    #     ncf_proportional=False,
    #     ndf=5,
    #     ndfn=25,
    #     random_seed=0,
    #     spatial_prevalent_ratio=0.5,
    #     spatial_prevalence_threshold=0.5,
    #     time_unit=25.0,
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
    #     area=200.0,
    #     cell_size=5.0,
    #     n_base=10,
    #     lambda_1=5.0,
    #     lambda_2=5.0,
    #     m_clumpy=1,
    #     m_overlap=1,
    #     ncfr=0.0,
    #     ncfn=0.0,
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
    #     area=200.0,
    #     cell_size=5.0,
    #     n_base=10,
    #     lambda_1=5.0,
    #     lambda_2=5.0,
    #     m_clumpy=1,
    #     m_overlap=1,
    #     ncfr=0.0,
    #     ncfn=0.0,
    #     ncf_proportional=False,
    #     ndf=5,
    #     ndfn=100,
    #     random_seed=0,
    #     spatial_prevalent_ratio=1.0,
    #     spatial_prevalence_threshold=1.0,
    #     step_length_mean=5.0,
    #     step_length_method=StepLengthMethod.UNIFORM,
    #     step_length_uniform_low_to_mean_ratio=0.75,
    #     step_length_normal_std_ratio=3.0,
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


def create_batch_04():
    print("create_batch_04")

    Path("data\\batch_04").mkdir(parents=True, exist_ok=True)

    random_seeds = []
    random_seeds_known = False

    try:
        os.remove("trash.txt")
    except OSError:
        pass

    for time_prevalence_threshold in np.arange(0, 1.01, 0.1).round(1):
        print("\n\ntime_prevalence_threshold=%.1f\n" % time_prevalence_threshold)

        samples_number = 100
        random_seed = 0

        for sample in range(samples_number):
            print("sample %d of %d" % (sample + 1, samples_number))

            sp = StandardParameters(
                area=1000.0,
                cell_size=5.0,
                n_base=1,
                lambda_1=8.0,
                lambda_2=1.0,
                m_clumpy=1,
                m_overlap=1,
                ncfr=0.0,
                ncfn=0.0,
                ncf_proportional=False,
                ndf=0,
                ndfn=0,
                random_seed=random_seed,
                persistent_ratio=1.0,
                spatial_prevalence_threshold=1.0,
                time_prevalence_threshold=time_prevalence_threshold
            )

            save_stdout = sys.stdout
            sys.stdout = open('trash.txt', 'a')

            if not random_seeds_known:
                while True:
                    stsg = SpatioTemporalStandardGenerator(sp=sp)
                    if ((stsg.si.collocation_lengths == np.array([8], dtype=np.int32)).all() and
                            (stsg.si.collocation_instances_counts == np.array([1], dtype=np.int32)).all()):
                        break

                    sp.random_seed += 1

                random_seeds.append(sp.random_seed)
                random_seed = sp.random_seed + 1

            else:
                sp.random_seed = random_seeds[sample]

            output_filename = "data\\batch_04\\SpatioTemporalStandardGenerator_output_file__time_prevalence_threshold_%.1f__sample_%02d.txt" % (time_prevalence_threshold, sample)

            stsg = SpatioTemporalStandardGenerator(sp=sp)
            stsg.generate(
                time_frames_number=1000,
                output_filename=output_filename,
                output_filename_timestamp=False
            )

            sys.stdout.close()
            sys.stdout = save_stdout

        if not random_seeds_known:
            print("\nrandom_seeds:\n%s" % str(random_seeds))
            random_seeds_known = True


def create_batch_05():
    print("create_batch_05")

    Path("data\\batch_05").mkdir(parents=True, exist_ok=True)

    random_seeds = []
    random_seeds_known = False

    try:
        os.remove("trash.txt")
    except OSError:
        pass

    for velocity_limit in np.arange(5, 51, 5):
        print("\n\nvelocity_limit=%.1f\n" % velocity_limit)

        samples_number = 100
        random_seed = 0

        for sample in range(samples_number):
            print("sample %d of %d" % (sample + 1, samples_number))

            iap = InteractionApproachParameters(
                area=1000.0,
                cell_size=5.0,
                n_base=1,
                lambda_1=8.0,
                lambda_2=1.0,
                m_clumpy=1,
                m_overlap=1,
                ncfr=0.0,
                ncfn=0.0,
                ncf_proportional=False,
                ndf=0,
                ndfn=0,
                random_seed=random_seed,
                spatial_prevalent_ratio=1.0,
                spatial_prevalence_threshold=1.0,
                time_unit=25.0,
                distance_unit=1.0,
                approx_steps_number=2,
                k_force=10000.0,
                force_limit=100.0,
                velocity_limit=velocity_limit,
                faraway_limit_ratio=np.sqrt(2) / 2,
                mass_method=IaeMassMethod.CONSTANT,
                mass_mean=1.0,
                mass_normal_std_ratio=1 / 5,
                velocity_method=IaeVelocityMethod.CONSTANT,
                velocity_mean=0.0,
                identical_features_interaction_mode=IdenticalFeaturesInteractionMode.REPEL,
                different_features_interaction_mode=DifferentFeaturesInteractionMode.COLLOCATION_ATTRACT_OTHER_NEUTRAL
            )

            save_stdout = sys.stdout
            sys.stdout = open('trash.txt', 'a')

            if not random_seeds_known:
                while True:
                    stiag = SpatioTemporalInteractionApproachGenerator(iap=iap)
                    if ((stiag.iai.collocation_lengths == np.array([8], dtype=np.int32)).all() and
                            (stiag.iai.collocation_instances_counts == np.array([1], dtype=np.int32)).all()):
                        break

                    iap.random_seed += 1

                random_seeds.append(iap.random_seed)
                random_seed = iap.random_seed + 1

            else:
                iap.random_seed = random_seeds[sample]

            output_filename = "data\\batch_05\\SpatioTemporalInteractionApproachGenerator_output_file__velocity_limit_%04.1f__sample_%02d.txt" % (velocity_limit, sample)

            stiag = SpatioTemporalInteractionApproachGenerator(iap=iap)
            stiag.generate(
                time_frames_number=1000,
                output_filename=output_filename,
                output_filename_timestamp=False
            )

            sys.stdout.close()
            sys.stdout = save_stdout

        if not random_seeds_known:
            print("\nrandom_seeds:\n%s" % str(random_seeds))
            random_seeds_known = True


def create_batch_06():
    print("create_batch_06")

    Path("data\\batch_06").mkdir(parents=True, exist_ok=True)

    random_seeds = []
    random_seeds_known = False

    try:
        os.remove("trash.txt")
    except OSError:
        pass

    for velocity_limit in np.arange(5, 51, 5):
        print("\n\nvelocity_limit=%.1f\n" % velocity_limit)

        samples_number = 100
        random_seed = 0

        for sample in range(samples_number):
            print("sample %d of %d" % (sample + 1, samples_number))

            odap = OptimalDistanceApproachParameters(
                area=1000.0,
                cell_size=5.0,
                n_base=1,
                lambda_1=8.0,
                lambda_2=1.0,
                m_clumpy=1,
                m_overlap=1,
                ncfr=0.0,
                ncfn=0.0,
                ncf_proportional=False,
                ndf=0,
                ndfn=0,
                random_seed=random_seed,
                spatial_prevalent_ratio=1.0,
                spatial_prevalence_threshold=1.0,
                time_unit=25.0,
                approx_steps_number=2,
                k_optimal_distance=2.0,
                k_force=100.0,
                force_limit=100.0,
                velocity_limit=velocity_limit,
                faraway_limit_ratio=np.sqrt(2) / 2,
                mass_method=OdaeMassMethod.CONSTANT,
                mass_mean=1.0,
                mass_normal_std_ratio=1 / 5,
                velocity_method=OdaeVelocityMethod.CONSTANT,
                velocity_mean=0.0
            )

            save_stdout = sys.stdout
            sys.stdout = open('trash.txt', 'a')

            if not random_seeds_known:
                while True:
                    stodag = SpatioTemporalOptimalDistanceApproachGenerator(odap=odap)
                    if ((stodag.odai.collocation_lengths == np.array([8], dtype=np.int32)).all() and
                            (stodag.odai.collocation_instances_counts == np.array([1], dtype=np.int32)).all()):
                        break

                    odap.random_seed += 1

                random_seeds.append(odap.random_seed)
                random_seed = odap.random_seed + 1

            else:
                odap.random_seed = random_seeds[sample]

            output_filename = "data\\batch_06\\SpatioTemporalOptimalDistanceApproachGenerator_output_file__velocity_limit_%04.1f__sample_%02d.txt" % (velocity_limit, sample)

            stodag = SpatioTemporalOptimalDistanceApproachGenerator(odap=odap)
            stodag.generate(
                time_frames_number=1000,
                output_filename=output_filename,
                output_filename_timestamp=False
            )

            sys.stdout.close()
            sys.stdout = save_stdout

        if not random_seeds_known:
            print("\nrandom_seeds:\n%s" % str(random_seeds))
            random_seeds_known = True


def create_batch_07():
    print("create_batch_07")

    Path("data\\batch_07").mkdir(parents=True, exist_ok=True)

    random_seeds = []
    random_seeds_known = False

    try:
        os.remove("trash.txt")
    except OSError:
        pass

    for center_noise_displacement in np.arange(0.4, 4.1, 0.4).round(1):
        print("\n\ncenter_noise_displacement=%.1f\n" % center_noise_displacement)

        samples_number = 100
        random_seed = 0

        for sample in range(samples_number):
            print("sample %d of %d" % (sample + 1, samples_number))

            cmap = CircularMotionApproachParameters(
                area=1000.0,
                cell_size=5.0,
                n_base=1,
                lambda_1=8.0,
                lambda_2=1.0,
                m_clumpy=1,
                m_overlap=1,
                ncfr=0.0,
                ncfn=0.0,
                ncf_proportional=False,
                ndf=0,
                ndfn=0,
                random_seed=random_seed,
                spatial_prevalent_ratio=1.0,
                spatial_prevalence_threshold=1.0,
                circle_chain_size=5,
                omega_min=2 * np.pi / 200,
                omega_max=2 * np.pi / 50,
                circle_r_min=4.0,
                circle_r_max=40.0,
                center_noise_displacement=center_noise_displacement
            )

            save_stdout = sys.stdout
            sys.stdout = open('trash.txt', 'a')

            if not random_seeds_known:
                while True:
                    stcmag = SpatioTemporalCircularMotionApproachGenerator(cmap=cmap)
                    if ((stcmag.cmai.collocation_lengths == np.array([8], dtype=np.int32)).all() and
                            (stcmag.cmai.collocation_instances_counts == np.array([1], dtype=np.int32)).all()):
                        break

                    cmap.random_seed += 1

                random_seeds.append(cmap.random_seed)
                random_seed = cmap.random_seed + 1

            else:
                cmap.random_seed = random_seeds[sample]

            output_filename = "data\\batch_07\\SpatioTemporalCircularMotionApproachGenerator_output_file__center_noise_displacement_%02.1f__sample_%02d.txt" % (center_noise_displacement, sample)

            stcmag = SpatioTemporalCircularMotionApproachGenerator(cmap=cmap)
            stcmag.generate(
                time_frames_number=1000,
                output_filename=output_filename,
                output_filename_timestamp=False
            )

            sys.stdout.close()
            sys.stdout = save_stdout

        if not random_seeds_known:
            print("\nrandom_seeds:\n%s" % str(random_seeds))
            random_seeds_known = True


def create_batch_08():
    print("create_batch_08")

    Path("data\\batch_08").mkdir(parents=True, exist_ok=True)

    random_seeds = []
    random_seeds_known = False

    try:
        os.remove("trash.txt")
    except OSError:
        pass

    for waiting_time_frames in np.arange(40, 401, 40, dtype=np.int32):
        print("\n\nwaiting_time_frames=%d\n" % waiting_time_frames)

        samples_number = 100
        random_seed = 0

        for sample in range(samples_number):
            print("sample %d of %d" % (sample + 1, samples_number))

            tap = TravelApproachParameters(
                area=1000.0,
                cell_size=5.0,
                n_base=1,
                lambda_1=8.0,
                lambda_2=1.0,
                m_clumpy=1,
                m_overlap=1,
                ncfr=0.0,
                ncfn=0.0,
                ncf_proportional=False,
                ndf=0,
                ndfn=0,
                random_seed=random_seed,
                spatial_prevalent_ratio=1.0,
                spatial_prevalence_threshold=1.0,
                step_length_mean=5.0,
                step_length_method=StepLengthMethod.UNIFORM,
                step_length_uniform_low_to_mean_ratio=0.75,
                step_length_normal_std_ratio=3.0,
                step_angle_range_mean=np.pi / 4,
                step_angle_range_limit=np.pi / 2,
                step_angle_method=StepAngleMethod.UNIFORM,
                step_angle_normal_std_ratio=1 / 3,
                waiting_time_frames=waiting_time_frames
            )

            save_stdout = sys.stdout
            sys.stdout = open('trash.txt', 'a')

            if not random_seeds_known:
                while True:
                    sttag = SpatioTemporalTravelApproachGenerator(tap=tap)
                    if ((sttag.tai.collocation_lengths == np.array([8], dtype=np.int32)).all() and
                            (sttag.tai.collocation_instances_counts == np.array([1], dtype=np.int32)).all()):
                        break

                    tap.random_seed += 1

                random_seeds.append(tap.random_seed)
                random_seed = tap.random_seed + 1

            else:
                tap.random_seed = random_seeds[sample]

            output_filename = "data\\batch_08\\SpatioTemporalTravelApproachGenerator_output_file__waiting_time_frames_%03d__sample_%02d.txt" % (waiting_time_frames, sample)

            sttag = SpatioTemporalTravelApproachGenerator(tap=tap)
            sttag.generate(
                time_frames_number=1000,
                output_filename=output_filename,
                output_filename_timestamp=False
            )

            sys.stdout.close()
            sys.stdout = save_stdout

        if not random_seeds_known:
            print("\nrandom_seeds:\n%s" % str(random_seeds))
            random_seeds_known = True


def main():
    # create_batch_01()
    # create_batch_02()
    # create_batch_03()
    # create_batch_04()
    # create_batch_05()
    # create_batch_06()
    # create_batch_07()
    create_batch_08()


if __name__ == "__main__":
    main()
