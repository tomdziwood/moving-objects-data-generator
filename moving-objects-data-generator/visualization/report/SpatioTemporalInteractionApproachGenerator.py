import numpy as np
from matplotlib import pyplot as plt

from algorithms.enums.InteractionApproachEnums import MassMethod, VelocityMethod, IdenticalFeaturesInteractionMode, DifferentFeaturesInteractionMode
from algorithms.generator.SpatioTemporalInteractionApproachGenerator import SpatioTemporalInteractionApproachGenerator
from algorithms.parameters.InteractionApproachParameters import InteractionApproachParameters
from visualization.report.Utils import visualize_parts, visualize_x_y, visualize_3d, visualize_perspectives


def visualize_demo_1_generate_data():
    print("SpatioTemporalInteractionApproachGenerator visualize_demo_1_generate_data()")

    iap = InteractionApproachParameters(
        area=50.0,
        cell_size=5.0,
        n_base=2,
        lambda_1=4.0,
        lambda_2=1.0,
        m_clumpy=1,
        m_overlap=1,
        ncfr=0.0,
        ncfn=0.0,
        ncf_proportional=False,
        ndf=0,
        ndfn=0,
        random_seed=24473,
        spatial_prevalent_ratio=1.0,
        spatial_prevalence_threshold=1.0,
        time_unit=25.0,
        distance_unit=1.0,
        approx_steps_number=5,
        k_force=1000.0,
        force_limit=20.0,
        velocity_limit=20.0,
        faraway_limit_ratio=np.sqrt(2) / 2,
        mass_method=MassMethod.CONSTANT,
        mass_mean=1.0,
        mass_normal_std_ratio=1 / 5,
        velocity_method=VelocityMethod.CONSTANT,
        velocity_mean=0.0,
        identical_features_interaction_mode=IdenticalFeaturesInteractionMode.REPEL,
        different_features_interaction_mode=DifferentFeaturesInteractionMode.ATTRACT
    )

    while True:
        stiag = SpatioTemporalInteractionApproachGenerator(iap=iap)
        if ((stiag.iai.collocation_lengths >= np.array([4, 4], dtype=np.int32)).all() and
                (stiag.iai.collocation_lengths <= np.array([5, 5], dtype=np.int32)).all() and
                (stiag.iai.collocation_instances_counts == np.array([2, 2], dtype=np.int32)).all()):
            print(iap.random_seed)
            break

        iap.random_seed += 1

    stiag.generate(
        time_frames_number=500,
        output_filename="data\\SpatioTemporalInteractionApproachGenerator_output_file_demo_1.txt",
        output_filename_timestamp=False
    )


def visualize_demo_1():
    print("SpatioTemporalInteractionApproachGenerator visualize_demo_1()")

    visualize_demo_1_generate_data()

    visualize_x_y("data\\SpatioTemporalInteractionApproachGenerator_output_file_demo_1.txt")

    fig = plt.gcf()
    plt.show()
    fig.savefig("output\\SpatioTemporalInteractionApproachGenerator_output_file_demo_1.png", bbox_inches='tight')
    fig.savefig("output\\SpatioTemporalInteractionApproachGenerator_output_file_demo_1.svg", bbox_inches='tight')
    fig.savefig("output\\SpatioTemporalInteractionApproachGenerator_output_file_demo_1.eps", bbox_inches='tight')


def visualize_demo_2():
    print("SpatioTemporalInteractionApproachGenerator visualize_demo_2()")

    # visualize_demo_1_generate_data()

    visualize_parts("data\\SpatioTemporalInteractionApproachGenerator_output_file_demo_1.txt", equal_aspect=False)

    fig = plt.gcf()
    plt.show()
    fig.savefig("output\\SpatioTemporalInteractionApproachGenerator_output_file_demo_2.png", bbox_inches='tight')
    fig.savefig("output\\SpatioTemporalInteractionApproachGenerator_output_file_demo_2.svg", bbox_inches='tight')
    fig.savefig("output\\SpatioTemporalInteractionApproachGenerator_output_file_demo_2.eps", bbox_inches='tight')


def visualize_demo_3():
    print("SpatioTemporalInteractionApproachGenerator visualize_demo_3()")

    # visualize_demo_1_generate_data()

    visualize_3d("data\\SpatioTemporalInteractionApproachGenerator_output_file_demo_1.txt")

    fig = plt.gcf()
    plt.show()
    fig.savefig("output\\SpatioTemporalInteractionApproachGenerator_output_file_demo_3.png", bbox_inches='tight')
    fig.savefig("output\\SpatioTemporalInteractionApproachGenerator_output_file_demo_3.svg", bbox_inches='tight')
    fig.savefig("output\\SpatioTemporalInteractionApproachGenerator_output_file_demo_3.eps", bbox_inches='tight')


def visualize_demo_4():
    print("SpatioTemporalInteractionApproachGenerator visualize_demo_4()")

    # visualize_demo_1_generate_data()

    visualize_perspectives("data\\SpatioTemporalInteractionApproachGenerator_output_file_demo_1.txt")

    fig = plt.gcf()
    plt.show()
    fig.savefig("output\\SpatioTemporalInteractionApproachGenerator_output_file_demo_4.png", bbox_inches='tight')
    fig.savefig("output\\SpatioTemporalInteractionApproachGenerator_output_file_demo_4.svg", bbox_inches='tight')
    fig.savefig("output\\SpatioTemporalInteractionApproachGenerator_output_file_demo_4.eps", bbox_inches='tight')


def visualize_demo_5_generate_data():
    print("SpatioTemporalInteractionApproachGenerator visualize_demo_5_generate_data()")

    iap = InteractionApproachParameters(
        area=1000.0,
        cell_size=5.0,
        n_base=1,
        lambda_1=5.0,
        lambda_2=1.0,
        m_clumpy=1,
        m_overlap=1,
        ncfr=0.0,
        ncfn=0.0,
        ncf_proportional=False,
        ndf=0,
        ndfn=0,
        random_seed=132,
        spatial_prevalent_ratio=1.0,
        spatial_prevalence_threshold=1.0,
        time_unit=25.0,
        distance_unit=1.0,
        approx_steps_number=5,
        k_force=1000.0,
        force_limit=20.0,
        velocity_limit=20.0,
        faraway_limit_ratio=np.sqrt(2) / 2,
        mass_method=MassMethod.CONSTANT,
        mass_mean=1.0,
        mass_normal_std_ratio=1 / 5,
        velocity_method=VelocityMethod.CONSTANT,
        velocity_mean=0.0,
        identical_features_interaction_mode=IdenticalFeaturesInteractionMode.REPEL,
        different_features_interaction_mode=DifferentFeaturesInteractionMode.ATTRACT
    )

    while True:
        stiag = SpatioTemporalInteractionApproachGenerator(iap=iap)
        if ((stiag.iai.collocation_lengths == np.array([10], dtype=np.int32)).all() and
                (stiag.iai.collocation_instances_counts == np.array([1], dtype=np.int32)).all()):
            print(iap.random_seed)
            break

        iap.random_seed += 1

    stiag.generate(
        time_frames_number=500,
        output_filename="data\\SpatioTemporalInteractionApproachGenerator_output_file_demo_5.txt",
        output_filename_timestamp=False
    )


def visualize_demo_5():
    print("SpatioTemporalInteractionApproachGenerator visualize_demo_5()")

    visualize_demo_5_generate_data()

    visualize_3d("data\\SpatioTemporalInteractionApproachGenerator_output_file_demo_5.txt")

    fig = plt.gcf()
    plt.show()
    fig.savefig("output\\SpatioTemporalInteractionApproachGenerator_output_file_demo_5.png", bbox_inches='tight')
    fig.savefig("output\\SpatioTemporalInteractionApproachGenerator_output_file_demo_5.svg", bbox_inches='tight')
    fig.savefig("output\\SpatioTemporalInteractionApproachGenerator_output_file_demo_5.eps", bbox_inches='tight')


def main():
    visualize_demo_1()
    visualize_demo_2()
    visualize_demo_3()
    visualize_demo_4()
    visualize_demo_5()


if __name__ == "__main__":
    main()
