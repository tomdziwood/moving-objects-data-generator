import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from algorithms.enums.OptimalDistanceApproachEnums import MassMethod, VelocityMethod
from algorithms.generator.SpatioTemporalOptimalDistanceApproachGenerator import SpatioTemporalOptimalDistanceApproachGenerator
from algorithms.parameters.OptimalDistanceApproachParameters import OptimalDistanceApproachParameters
from visualization.report.Utils import visualize_x_y, visualize_3d, visualize_perspectives, visualize_frame


def visualize_demo_1_generate_data():
    print("SpatioTemporalOptimalDistanceApproachGenerator visualize_demo_1_generate_data()")

    odap = OptimalDistanceApproachParameters(
        area=50,
        cell_size=5,
        n_base=2,
        lambda_1=4,
        lambda_2=2,
        m_clumpy=1,
        m_overlap=1,
        ncfr=0,
        ncfn=0,
        ncf_proportional=False,
        ndf=0,
        ndfn=0,
        random_seed=428,
        spatial_prevalent_ratio=1.0,
        spatial_prevalence_threshold=1.0,
        time_unit=25,
        approx_steps_number=5,
        k_optimal_distance=5.0,
        k_force=1.0,
        force_limit=20.0,
        velocity_limit=20.0,
        faraway_limit_ratio=np.sqrt(2) / 2,
        mass_method=MassMethod.CONSTANT,
        mass_mean=1.0,
        mass_normal_std_ratio=1 / 5,
        velocity_method=VelocityMethod.CONSTANT,
        velocity_mean=0.0
    )

    while True:
        stodag = SpatioTemporalOptimalDistanceApproachGenerator(odap=odap)
        if ((stodag.odai.collocation_lengths == np.array([4, 4], dtype=np.int32)).all() and
                (stodag.odai.collocation_instances_counts == np.array([2, 2], dtype=np.int32)).all()):
            print(odap.random_seed)
            break

        odap.random_seed += 1

    stodag.generate(
        time_frames_number=500,
        output_filename="data\\SpatioTemporalOptimalDistanceApproachGenerator_output_file_demo_1.txt",
        output_filename_timestamp=False
    )


def visualize_demo_1():
    print("SpatioTemporalOptimalDistanceApproachGenerator visualize_demo_1()")

    visualize_demo_1_generate_data()

    visualize_x_y("data\\SpatioTemporalOptimalDistanceApproachGenerator_output_file_demo_1.txt")

    fig = plt.gcf()
    plt.show()
    fig.savefig("output\\SpatioTemporalOptimalDistanceApproachGenerator_output_file_demo_1.png", bbox_inches='tight')
    fig.savefig("output\\SpatioTemporalOptimalDistanceApproachGenerator_output_file_demo_1.svg", bbox_inches='tight')
    fig.savefig("output\\SpatioTemporalOptimalDistanceApproachGenerator_output_file_demo_1.eps", bbox_inches='tight')


def visualize_demo_2():
    print("SpatioTemporalOptimalDistanceApproachGenerator visualize_demo_2()")

    # visualize_demo_1_generate_data()

    visualize_3d("data\\SpatioTemporalOptimalDistanceApproachGenerator_output_file_demo_1.txt")

    fig = plt.gcf()
    plt.show()
    fig.savefig("output\\SpatioTemporalOptimalDistanceApproachGenerator_output_file_demo_2.png", bbox_inches='tight')
    fig.savefig("output\\SpatioTemporalOptimalDistanceApproachGenerator_output_file_demo_2.svg", bbox_inches='tight')
    fig.savefig("output\\SpatioTemporalOptimalDistanceApproachGenerator_output_file_demo_2.eps", bbox_inches='tight')


def visualize_demo_3():
    print("SpatioTemporalOptimalDistanceApproachGenerator visualize_demo_3()")

    # visualize_demo_1_generate_data()

    visualize_perspectives("data\\SpatioTemporalOptimalDistanceApproachGenerator_output_file_demo_1.txt")

    fig = plt.gcf()
    plt.show()
    fig.savefig("output\\SpatioTemporalOptimalDistanceApproachGenerator_output_file_demo_3.png", bbox_inches='tight')
    fig.savefig("output\\SpatioTemporalOptimalDistanceApproachGenerator_output_file_demo_3.svg", bbox_inches='tight')
    fig.savefig("output\\SpatioTemporalOptimalDistanceApproachGenerator_output_file_demo_3.eps", bbox_inches='tight')


def visualize_demo_4_generate_data():
    print("SpatioTemporalOptimalDistanceApproachGenerator visualize_demo_4_generate_data()")

    odap = OptimalDistanceApproachParameters(
        area=100,
        cell_size=5,
        n_base=2,
        lambda_1=4,
        lambda_2=5,
        m_clumpy=1,
        m_overlap=1,
        ncfr=0,
        ncfn=0,
        ncf_proportional=False,
        ndf=1,
        ndfn=15,
        random_seed=438,
        spatial_prevalent_ratio=1.0,
        spatial_prevalence_threshold=1.0,
        time_unit=1,
        approx_steps_number=50,
        k_optimal_distance=5.0,
        k_force=1.0,
        force_limit=20.0,
        velocity_limit=5.0,
        faraway_limit_ratio=np.sqrt(2) / 2,
        mass_method=MassMethod.CONSTANT,
        mass_mean=1.0,
        mass_normal_std_ratio=1 / 5,
        velocity_method=VelocityMethod.CONSTANT,
        velocity_mean=0.0
    )

    while True:
        stodag = SpatioTemporalOptimalDistanceApproachGenerator(odap=odap)
        if ((stodag.odai.collocation_lengths >= np.array([4, 4], dtype=np.int32)).all() and
                (stodag.odai.collocation_lengths <= np.array([5, 5], dtype=np.int32)).all() and
                (stodag.odai.collocation_instances_counts == np.array([4, 4], dtype=np.int32)).all()):
            print(odap.random_seed)
            break

        odap.random_seed += 1

    stodag.generate(
        time_frames_number=500,
        output_filename="data\\SpatioTemporalOptimalDistanceApproachGenerator_output_file_demo_4.txt",
        output_filename_timestamp=False
    )


def visualize_demo_4():
    print("SpatioTemporalOptimalDistanceApproachGenerator visualize_demo_4()")

    visualize_demo_4_generate_data()

    visualize_frame("data\\SpatioTemporalOptimalDistanceApproachGenerator_output_file_demo_4.txt", time_frame=400, markersize=40)

    fig = plt.gcf()
    plt.show()
    fig.savefig("output\\SpatioTemporalOptimalDistanceApproachGenerator_output_file_demo_4.png", bbox_inches='tight')
    fig.savefig("output\\SpatioTemporalOptimalDistanceApproachGenerator_output_file_demo_4.svg", bbox_inches='tight')
    fig.savefig("output\\SpatioTemporalOptimalDistanceApproachGenerator_output_file_demo_4.eps", bbox_inches='tight')


def visualize_demo_5_generate_data():
    print("SpatioTemporalOptimalDistanceApproachGenerator visualize_demo_5_generate_data()")

    odap = OptimalDistanceApproachParameters(
        area=100,
        cell_size=5,
        n_base=1,
        lambda_1=7,
        lambda_2=1,
        m_clumpy=1,
        m_overlap=1,
        ncfr=0,
        ncfn=0,
        ncf_proportional=False,
        ndf=0,
        ndfn=0,
        random_seed=4,
        spatial_prevalent_ratio=1.0,
        spatial_prevalence_threshold=1.0,
        time_unit=25,
        approx_steps_number=2,
        k_optimal_distance=5.0,
        k_force=1.0,
        force_limit=20.0,
        velocity_limit=10.0,
        faraway_limit_ratio=np.sqrt(2) / 2,
        mass_method=MassMethod.CONSTANT,
        mass_mean=1.0,
        mass_normal_std_ratio=1 / 5,
        velocity_method=VelocityMethod.CONSTANT,
        velocity_mean=0.0
    )

    while True:
        stodag = SpatioTemporalOptimalDistanceApproachGenerator(odap=odap)
        if ((stodag.odai.collocation_lengths == np.array([7], dtype=np.int32)).all() and
                (stodag.odai.collocation_instances_counts == np.array([1], dtype=np.int32)).all()):
            print(odap.random_seed)
            break

        odap.random_seed += 1

    stodag.generate(
        time_frames_number=500,
        output_filename="data\\SpatioTemporalOptimalDistanceApproachGenerator_output_file_demo_5.txt",
        output_filename_timestamp=False
    )


def visualize_demo_6_generate_data():
    print("SpatioTemporalOptimalDistanceApproachGenerator visualize_demo_5_generate_data()")

    odap = OptimalDistanceApproachParameters(
        area=100,
        cell_size=5,
        n_base=1,
        lambda_1=7,
        lambda_2=1,
        m_clumpy=1,
        m_overlap=1,
        ncfr=0,
        ncfn=0,
        ncf_proportional=False,
        ndf=0,
        ndfn=0,
        random_seed=4,
        spatial_prevalent_ratio=1.0,
        spatial_prevalence_threshold=1.0,
        time_unit=25,
        approx_steps_number=2,
        k_optimal_distance=5.0,
        k_force=1.0,
        force_limit=20.0,
        velocity_limit=2.5,
        faraway_limit_ratio=np.sqrt(2) / 2,
        mass_method=MassMethod.CONSTANT,
        mass_mean=1.0,
        mass_normal_std_ratio=1 / 5,
        velocity_method=VelocityMethod.CONSTANT,
        velocity_mean=0.0
    )

    while True:
        stodag = SpatioTemporalOptimalDistanceApproachGenerator(odap=odap)
        if ((stodag.odai.collocation_lengths == np.array([7], dtype=np.int32)).all() and
                (stodag.odai.collocation_instances_counts == np.array([1], dtype=np.int32)).all()):
            print(odap.random_seed)
            break

        odap.random_seed += 1

    stodag.generate(
        time_frames_number=500,
        output_filename="data\\SpatioTemporalOptimalDistanceApproachGenerator_output_file_demo_6.txt",
        output_filename_timestamp=False
    )


def visualize_demo_5():
    print("SpatioTemporalOptimalDistanceApproachGenerator visualize_demo_5()")

    visualize_demo_5_generate_data()
    visualize_demo_6_generate_data()

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })

    plt.rcParams['axes.axisbelow'] = True

    cm = 1 / 2.54
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20*cm, 10*cm))
    plt.tight_layout(pad=1.5, w_pad=1, h_pad=2)

    markers_list = ["o", "s", "^", "+", "*", "x", "2", (6, 1, 0), (6, 2, 0), "h", "v", "<", ">", "1", "3", "4", "D", "8", "p", "P", "H", "X", "d"]
    markers_list_length = len(markers_list)

    start_frame = 50
    end_frame = 150

    xlim = [33, 47]
    ylim = [91, 105]

    input_filenames = [
        "data\\SpatioTemporalOptimalDistanceApproachGenerator_output_file_demo_5.txt",
        "data\\SpatioTemporalOptimalDistanceApproachGenerator_output_file_demo_6.txt"
    ]

    for i in range(len(input_filenames)):
        df = pd.read_csv(input_filenames[i], sep=';', header=None, comment="#")
        df.columns = ["time_frame", "feature_id", "feature_instance_id", "x", "y"]
        time_frames = df.time_frame.unique()
        sorted(time_frames)
        print("time_frames size: %d" % time_frames.size)
        df_filtered = df[np.logical_and(df.time_frame >= start_frame, df.time_frame <= end_frame)]

        (x_min, y_min) = (np.int32(df.x.min()), np.int32(df.y.min()))
        print("min coor:\t(%d, %d)" % (x_min, y_min))
        (x_max, y_max) = (np.int32(df.x.max()), np.int32(df.y.max()))
        print("max coor:\t(%d, %d)" % (x_max, y_max))

        ax = axs[i]

        ax.set_aspect('equal')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(r"x")
        ax.set_ylabel(r"y")

        df_sorted = df_filtered.sort_values(['feature_id', 'feature_instance_id', 'time_frame'])
        df_grouped = df_sorted.groupby(['feature_id', 'feature_instance_id'])
        df_listed = df_grouped.agg({'x': lambda x: list(x), 'y': lambda x: list(x)}).reset_index()

        objects_number = len(df_listed)
        colors_list = np.linspace(0, 1, objects_number)

        for index, row in df_listed.iterrows():
            print("Drawing route of object %d out of %d" % (index + 1, objects_number))
            ax.plot(row.x, row.y,
                    color=plt.get_cmap("nipy_spectral")(colors_list[index]),
                    marker=markers_list[row.feature_id % markers_list_length],
                    markersize=3, linewidth=1, markevery=1)

    plt.show()
    fig.savefig("output\\SpatioTemporalOptimalDistanceApproachGenerator_output_file_demo_5.png", bbox_inches='tight')
    fig.savefig("output\\SpatioTemporalOptimalDistanceApproachGenerator_output_file_demo_5.svg", bbox_inches='tight')
    fig.savefig("output\\SpatioTemporalOptimalDistanceApproachGenerator_output_file_demo_5.eps", bbox_inches='tight')


def main():
    visualize_demo_1()
    visualize_demo_2()
    visualize_demo_3()
    visualize_demo_4()
    visualize_demo_5()


if __name__ == "__main__":
    main()
