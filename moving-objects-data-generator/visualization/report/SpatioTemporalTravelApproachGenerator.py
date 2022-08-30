import numpy as np
from matplotlib import pyplot as plt

from algorithms.enums.TravelApproachEnums import StepLengthMethod, StepAngleMethod
from algorithms.generator.SpatioTemporalTravelApproachGenerator import SpatioTemporalTravelApproachGenerator
from algorithms.parameters.TravelApproachParameters import TravelApproachParameters
from visualization.report.Utils import visualize_x_y


def visualize_demo_1_generate_data():
    print("SpatioTemporalTravelApproachGenerator visualize_demo_1_generate_data()")

    tap = TravelApproachParameters(
        area=50,
        cell_size=5,
        n_base=0,
        lambda_1=6,
        lambda_2=3,
        m_clumpy=1,
        m_overlap=1,
        ncfr=0,
        ncfn=0,
        ncf_proportional=False,
        ndf=3,
        ndfn=6,
        random_seed=4,
        spatial_prevalent_ratio=1.0,
        spatial_prevalence_threshold=1.0,
        step_length_mean=2.0,
        step_length_method=StepLengthMethod.UNIFORM,
        step_length_uniform_low_to_mean_ratio=1 / 2,
        step_length_normal_std_ratio=3,
        step_angle_range_mean=np.pi / 4,
        step_angle_range_limit=np.pi / 2,
        step_angle_method=StepAngleMethod.UNIFORM,
        step_angle_normal_std_ratio=1 / 3,
        waiting_time_frames=200
    )

    sttag = SpatioTemporalTravelApproachGenerator(tap=tap)

    sttag.generate(
        time_frames_number=50,
        output_filename="data\\SpatioTemporalTravelApproachGenerator_output_file_demo_1.txt",
        output_filename_timestamp=False
    )


def visualize_demo_1():
    print("SpatioTemporalTravelApproachGenerator visualize_demo_1()")

    visualize_demo_1_generate_data()

    visualize_x_y("data\\SpatioTemporalTravelApproachGenerator_output_file_demo_1.txt", xlim=[0, 50], ylim=[0, 50], markersize=4, markevery=1)

    fig = plt.gcf()
    plt.show()
    fig.savefig("output\\SpatioTemporalTravelApproachGenerator_output_file_demo_1.png", bbox_inches='tight')
    fig.savefig("output\\SpatioTemporalTravelApproachGenerator_output_file_demo_1.svg", bbox_inches='tight')
    fig.savefig("output\\SpatioTemporalTravelApproachGenerator_output_file_demo_1.eps", bbox_inches='tight')


def visualize_demo_2_generate_data():
    print("SpatioTemporalTravelApproachGenerator visualize_demo_2_generate_data()")

    tap = TravelApproachParameters(
        area=50,
        cell_size=5,
        n_base=1,
        lambda_1=5,
        lambda_2=1,
        m_clumpy=1,
        m_overlap=1,
        ncfr=0,
        ncfn=0,
        ncf_proportional=False,
        ndf=0,
        ndfn=0,
        random_seed=129,
        spatial_prevalent_ratio=1.0,
        spatial_prevalence_threshold=1.0,
        step_length_mean=2.0,
        step_length_method=StepLengthMethod.UNIFORM,
        step_length_uniform_low_to_mean_ratio=1 / 5,
        step_length_normal_std_ratio=3,
        step_angle_range_mean=np.pi / 4,
        step_angle_range_limit=np.pi / 2,
        step_angle_method=StepAngleMethod.UNIFORM,
        step_angle_normal_std_ratio=1 / 3,
        waiting_time_frames=15
    )

    while True:
        sttag = SpatioTemporalTravelApproachGenerator(tap=tap)
        if ((sttag.tai.collocation_lengths == np.array([5], dtype=np.int32)).all() and
                (sttag.tai.collocation_instances_counts == np.array([1], dtype=np.int32)).all()):
            print(tap.random_seed)
            break

        tap.random_seed += 1

    sttag.generate(
        time_frames_number=40,
        output_filename="data\\SpatioTemporalTravelApproachGenerator_output_file_demo_2.txt",
        output_filename_timestamp=False
    )


def visualize_demo_2():
    print("SpatioTemporalTravelApproachGenerator visualize_demo_2()")

    visualize_demo_2_generate_data()

    visualize_x_y("data\\SpatioTemporalTravelApproachGenerator_output_file_demo_2.txt", xlim=[0, 40], ylim=[10, 50], markersize=4, markevery=1)

    fig = plt.gcf()
    plt.show()
    fig.savefig("output\\SpatioTemporalTravelApproachGenerator_output_file_demo_2.png", bbox_inches='tight')
    fig.savefig("output\\SpatioTemporalTravelApproachGenerator_output_file_demo_2.svg", bbox_inches='tight')
    fig.savefig("output\\SpatioTemporalTravelApproachGenerator_output_file_demo_2.eps", bbox_inches='tight')


def main():
    visualize_demo_1()
    visualize_demo_2()


if __name__ == "__main__":
    main()
