import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from algorithms.generator.SpatioTemporalCircularMotionApproachGenerator import SpatioTemporalCircularMotionApproachGenerator
from algorithms.initiation.CircularMotionApproachInitiation import CircularMotionApproachInitiation
from algorithms.parameters.CircularMotionApproachParameters import CircularMotionApproachParameters
from visualization.report.Utils import visualize_3d, visualize_x_y


def visualize_demo_1_generate_data():
    print("SpatioTemporalCircularMotionApproachGenerator visualize_demo_1_generate_data()")

    cmap = CircularMotionApproachParameters(
        area=1000,
        cell_size=5,
        n_base=2,
        lambda_1=5,
        lambda_2=3,
        m_clumpy=1,
        m_overlap=1,
        ncfr=0,
        ncfn=0,
        ncf_proportional=False,
        ndf=0,
        ndfn=0,
        random_seed=753,
        spatial_prevalent_ratio=1.0,
        spatial_prevalence_threshold=1.0,
        circle_chain_size=4,
        omega_min=2 * np.pi / 200,
        omega_max=2 * np.pi / 50,
        circle_r_min=20.0,
        circle_r_max=200.0,
        center_noise_displacement=5.0
    )

    while True:
        stcmag = SpatioTemporalCircularMotionApproachGenerator(cmap=cmap)
        if ((stcmag.cmai.collocation_lengths == np.array([3, 3], dtype=np.int32)).all() and
                (stcmag.cmai.collocation_instances_counts == np.array([2, 2], dtype=np.int32)).all()):
            print(cmap.random_seed)
            break

        cmap.random_seed += 1

    stcmag.generate(
        time_frames_number=500,
        output_filename="data\\SpatioTemporalCircularMotionApproachGenerator_output_file_demo_1.txt",
        output_filename_timestamp=False
    )


def visualize_demo_1():
    print("SpatioTemporalCircularMotionApproachGenerator visualize_demo_1()")

    visualize_demo_1_generate_data()

    visualize_x_y("data\\SpatioTemporalCircularMotionApproachGenerator_output_file_demo_1.txt")

    fig = plt.gcf()
    plt.show()
    fig.savefig("output\\SpatioTemporalCircularMotionApproachGenerator_output_file_demo_1.png", bbox_inches='tight')
    fig.savefig("output\\SpatioTemporalCircularMotionApproachGenerator_output_file_demo_1.svg", bbox_inches='tight')
    fig.savefig("output\\SpatioTemporalCircularMotionApproachGenerator_output_file_demo_1.eps", bbox_inches='tight')


def visualize_demo_2():
    print("SpatioTemporalCircularMotionApproachGenerator visualize_demo_2()")

    # visualize_demo_1_generate_data()

    visualize_3d("data\\SpatioTemporalCircularMotionApproachGenerator_output_file_demo_1.txt")

    fig = plt.gcf()
    plt.show()
    fig.savefig("output\\SpatioTemporalCircularMotionApproachGenerator_output_file_demo_2.png", bbox_inches='tight')
    fig.savefig("output\\SpatioTemporalCircularMotionApproachGenerator_output_file_demo_2.svg", bbox_inches='tight')
    fig.savefig("output\\SpatioTemporalCircularMotionApproachGenerator_output_file_demo_2.eps", bbox_inches='tight')


def visualize_demo_3_generate_data(circle_chain_size):
    print("SpatioTemporalCircularMotionApproachGenerator visualize_demo_3_generate_data()")

    cmap = CircularMotionApproachParameters(
        area=1000,
        cell_size=5,
        n_base=1,
        lambda_1=4,
        lambda_2=3,
        m_clumpy=1,
        m_overlap=1,
        ncfr=0,
        ncfn=0,
        ncf_proportional=False,
        ndf=0,
        ndfn=0,
        random_seed=22,
        spatial_prevalent_ratio=1.0,
        spatial_prevalence_threshold=1.0,
        circle_chain_size=circle_chain_size,
        omega_min=2 * np.pi / 200,
        omega_max=2 * np.pi / 50,
        circle_r_min=20.0,
        circle_r_max=200.0,
        center_noise_displacement=5.0
    )

    while True:
        stcmag = SpatioTemporalCircularMotionApproachGenerator(cmap=cmap)
        if ((stcmag.cmai.collocation_lengths == np.array([4], dtype=np.int32)).all() and
                (stcmag.cmai.collocation_instances_counts == np.array([3], dtype=np.int32)).all()):
            print(cmap.random_seed)
            break

        cmap.random_seed += 1

    stcmag.generate(
        time_frames_number=500,
        output_filename="data\\SpatioTemporalCircularMotionApproachGenerator_output_file_demo_3.txt",
        output_filename_timestamp=False
    )


def visualize_demo_3():
    print("SpatioTemporalCircularMotionApproachGenerator visualize_demo_3()")

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })

    plt.rcParams['axes.axisbelow'] = True

    cm = 1 / 2.54
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20 * cm, 20 * cm))
    plt.tight_layout(pad=1.5, w_pad=0, h_pad=4)

    markers_list = ["o", "s", "^", "+", "*", "x", "2", (6, 1, 0), (6, 2, 0), "h", "v", "<", ">", "1", "3", "4", "D", "8", "p", "P", "H", "X", "d"]
    markers_list_length = len(markers_list)

    for i_data in range(4):
        visualize_demo_3_generate_data(circle_chain_size=i_data + 1)

        df = pd.read_csv("data\\SpatioTemporalCircularMotionApproachGenerator_output_file_demo_3.txt", sep=';', header=None, comment="#")
        df.columns = ["time_frame", "feature_id", "feature_instance_id", "x", "y"]
        time_frames = df.time_frame.unique()
        sorted(time_frames)
        print("time_frames size: %d" % time_frames.size)

        (x_min, y_min) = (np.int32(df.x.min()), np.int32(df.y.min()))
        print("min coor:\t(%d, %d)" % (x_min, y_min))
        (x_max, y_max) = (np.int32(df.x.max()), np.int32(df.y.max()))
        print("max coor:\t(%d, %d)" % (x_max, y_max))
        # xlim = [x_min, x_max]
        # ylim = [y_min, y_max]

        ax = axs[i_data // 2, i_data % 2]

        ax.set_aspect('equal')
        # ax.set_xlim(xlim)
        # ax.set_ylim(ylim)

        ax.set_xlabel(r"x")
        ax.set_ylabel(r"y")
        ax.set_title(r"$n_{circle} = %d$" % (i_data + 1))

        df_sorted = df.sort_values(['feature_id', 'feature_instance_id', 'time_frame'])
        df_grouped = df_sorted.groupby(['feature_id', 'feature_instance_id'])
        df_listed = df_grouped.agg({'x': lambda x: list(x), 'y': lambda x: list(x)}).reset_index()

        objects_number = len(df_listed)

        color_per_feature = False
        if color_per_feature:
            colors_list = np.linspace(0, 1, df.feature_id.unique().size)
        else:
            colors_list = np.linspace(0, 1, objects_number)

        for index, row in df_listed.iterrows():
            print("Drawing route of object %d out of %d" % (index + 1, objects_number))
            if color_per_feature:
                color = plt.get_cmap("nipy_spectral")(colors_list[row.feature_id])
            else:
                color = plt.get_cmap("nipy_spectral")(colors_list[index])
            ax.plot(row.x, row.y,
                    color=color,
                    marker=markers_list[row.feature_id % markers_list_length],
                    markersize=0, linewidth=1, markevery=10)

    plt.show()
    fig.savefig("output\\SpatioTemporalCircularMotionApproachGenerator_output_file_demo_3.png", bbox_inches='tight')
    fig.savefig("output\\SpatioTemporalCircularMotionApproachGenerator_output_file_demo_3.svg", bbox_inches='tight')
    fig.savefig("output\\SpatioTemporalCircularMotionApproachGenerator_output_file_demo_3.eps", bbox_inches='tight')


def visualize_demo_4_generate_data(center_noise_displacement):
    print("SpatioTemporalCircularMotionApproachGenerator visualize_demo_4_generate_data()")

    cmap = CircularMotionApproachParameters(
        area=100,
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
        random_seed=25,
        spatial_prevalent_ratio=1.0,
        spatial_prevalence_threshold=1.0,
        circle_chain_size=2,
        omega_min=2 * np.pi / 200,
        omega_max=2 * np.pi / 50,
        circle_r_min=2.0,
        circle_r_max=40.0,
        center_noise_displacement=center_noise_displacement
    )

    while True:
        stcmag = SpatioTemporalCircularMotionApproachGenerator(cmap=cmap)
        if ((stcmag.cmai.collocation_lengths == np.array([5], dtype=np.int32)).all() and
                (stcmag.cmai.collocation_instances_counts == np.array([1], dtype=np.int32)).all()):
            print(cmap.random_seed)
            break

        cmap.random_seed += 1

    stcmag.generate(
        time_frames_number=200,
        output_filename="data\\SpatioTemporalCircularMotionApproachGenerator_output_file_demo_4.txt",
        output_filename_timestamp=False
    )


def visualize_demo_4():
    print("SpatioTemporalCircularMotionApproachGenerator visualize_demo_4()")

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })

    plt.rcParams['axes.axisbelow'] = True

    cm = 1 / 2.54
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20 * cm, 10 * cm))
    plt.tight_layout(pad=1.5, w_pad=2, h_pad=0)

    markers_list = ["o", "s", "^", "+", "*", "x", "2", (6, 1, 0), (6, 2, 0), "h", "v", "<", ">", "1", "3", "4", "D", "8", "p", "P", "H", "X", "d"]
    markers_list_length = len(markers_list)

    for i_data in range(2):
        visualize_demo_4_generate_data(center_noise_displacement=5.0 * i_data)

        df = pd.read_csv("data\\SpatioTemporalCircularMotionApproachGenerator_output_file_demo_4.txt", sep=';', header=None, comment="#")
        df.columns = ["time_frame", "feature_id", "feature_instance_id", "x", "y"]
        time_frames = df.time_frame.unique()
        sorted(time_frames)
        print("time_frames size: %d" % time_frames.size)

        (x_min, y_min) = (np.int32(df.x.min()), np.int32(df.y.min()))
        print("min coor:\t(%d, %d)" % (x_min, y_min))
        (x_max, y_max) = (np.int32(df.x.max()), np.int32(df.y.max()))
        print("max coor:\t(%d, %d)" % (x_max, y_max))
        xlim = [40, 180]
        ylim = [-20, 120]

        ax = axs[i_data % 2]

        ax.set_aspect('equal')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.set_xlabel(r"x")
        ax.set_ylabel(r"y")
        ax.set_title(r"$r_{displacement} = %.1f$" % (5.0 * i_data))

        df_sorted = df.sort_values(['feature_id', 'feature_instance_id', 'time_frame'])
        df_grouped = df_sorted.groupby(['feature_id', 'feature_instance_id'])
        df_listed = df_grouped.agg({'x': lambda x: list(x), 'y': lambda x: list(x)}).reset_index()

        objects_number = len(df_listed)

        color_per_feature = False
        if color_per_feature:
            colors_list = np.linspace(0, 1, df.feature_id.unique().size)
        else:
            colors_list = np.linspace(0, 1, objects_number)

        for index, row in df_listed.iterrows():
            print("Drawing route of object %d out of %d" % (index + 1, objects_number))
            if color_per_feature:
                color = plt.get_cmap("nipy_spectral")(colors_list[row.feature_id])
            else:
                color = plt.get_cmap("nipy_spectral")(colors_list[index])
            ax.plot(row.x, row.y,
                    color=color,
                    marker=markers_list[row.feature_id % markers_list_length],
                    markersize=0, linewidth=1, markevery=10)

    plt.show()
    fig.savefig("output\\SpatioTemporalCircularMotionApproachGenerator_output_file_demo_4.png", bbox_inches='tight')
    fig.savefig("output\\SpatioTemporalCircularMotionApproachGenerator_output_file_demo_4.svg", bbox_inches='tight')
    fig.savefig("output\\SpatioTemporalCircularMotionApproachGenerator_output_file_demo_4.eps", bbox_inches='tight')


def visualize_demo_5_generate_data(random_seed, center_noise_displacement):
    print("SpatioTemporalCircularMotionApproachGenerator visualize_demo_5_generate_data()")

    cmap = CircularMotionApproachParameters(
        area=100,
        cell_size=5,
        n_base=1,
        lambda_1=3,
        lambda_2=1,
        m_clumpy=1,
        m_overlap=1,
        ncfr=0,
        ncfn=0,
        ncf_proportional=False,
        ndf=0,
        ndfn=0,
        random_seed=random_seed,
        spatial_prevalent_ratio=1.0,
        spatial_prevalence_threshold=1.0,
        circle_chain_size=5,
        omega_min=2 * np.pi / 200,
        omega_max=2 * np.pi / 50,
        circle_r_min=2.0,
        circle_r_max=40.0,
        center_noise_displacement=center_noise_displacement
    )

    while True:
        cmai = CircularMotionApproachInitiation()
        cmai.initiate(cmap=cmap, report_output_filename="data\\SpatioTemporalCircularMotionApproachGenerator_output_file_demo_5.txt")
        if ((cmai.collocation_lengths == np.array([3], dtype=np.int32)).all() and
                (cmai.collocation_instances_counts == np.array([1], dtype=np.int32)).all()):
            print(cmap.random_seed)
            break

        cmap.random_seed += 1


def visualize_demo_5():
    print("SpatioTemporalCircularMotionApproachGenerator visualize_demo_5()")

    random_seed = 30
    center_noise_displacement = 5.0

    visualize_demo_5_generate_data(random_seed=random_seed, center_noise_displacement=center_noise_displacement)

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })

    plt.rcParams['axes.axisbelow'] = True

    cm = 1 / 2.54
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20 * cm, 10 * cm))
    plt.tight_layout(pad=1.5, w_pad=2, h_pad=0)

    # markers_list = ["o", "s", "^", "+", "*", "x", "2", (6, 1, 0), (6, 2, 0), "h", "v", "<", ">", "1", "3", "4", "D", "8", "p", "P", "H", "X", "d"]
    markers_list = ["+", "*", "x", "2", (6, 1, 0), (6, 2, 0), "h", "o", "s", "^", "v", "<", ">", "1", "3", "4", "D", "8", "p", "P", "H", "X", "d"]
    markers_list_length = len(markers_list)

    df = pd.read_csv("data\\SpatioTemporalCircularMotionApproachGenerator_output_file_demo_5.txt", sep=';', header=None, comment="#")
    df_x = df[df.columns[::2]]
    df_y = df[df.columns[1::2]]

    (x_min, y_min) = (np.int32(df_x.stack().min()), np.int32(df_y.stack().min()))
    print("min coor:\t(%d, %d)" % (x_min, y_min))
    (x_max, y_max) = (np.int32(df_x.stack().max()), np.int32(df_y.stack().max()))
    print("max coor:\t(%d, %d)" % (x_max, y_max))
    # xlim = [40, 180]
    # ylim = [-20, 120]

    center_x = df_x.iloc[0, :-1].to_numpy()
    center_y = df_y.iloc[0, :-1].to_numpy()

    paths_number = len(df) // 2
    circles_number = len(center_x)

    titles = [r"Przed przemieszczeniem środków okregów", r"Po przemieszczeniu środków okregów"]

    for df_part in range(2):
        index_start = df_part * paths_number
        index_stop = (df_part + 1) * paths_number
        df_half = df.iloc[index_start:index_stop]

        df_x = df_half[df_half.columns[::2]]
        df_y = df_half[df_half.columns[1::2]]

        ax = axs[df_part % 2]

        ax.set_aspect('equal')
        # ax.set_xlim(xlim)
        # ax.set_ylim(ylim)

        ax.set_xlabel(r"x")
        ax.set_ylabel(r"y")
        ax.set_title(titles[df_part])

        colors_list = np.linspace(0, 1, paths_number)

        for i_row in range(paths_number):
            x = df_x.iloc[i_row].to_numpy()
            y = df_y.iloc[i_row].to_numpy()
            color = plt.get_cmap("brg")(colors_list[i_row])
            ax.plot(x, y, color=color, linewidth=1, alpha=0.8, zorder=1)

        for i_row in range(paths_number):
            x = df_x.iloc[i_row, :-1].to_numpy()
            y = df_y.iloc[i_row, :-1].to_numpy()
            color = plt.get_cmap("brg")(colors_list[i_row])
            ax.scatter(x=x, y=y, s=3, color=color, marker='o', linewidth=1, alpha=0.8, zorder=2)

        for i_circle in range(circles_number):
            x = center_x[i_circle]
            y = center_y[i_circle]
            ax.add_patch(plt.Circle(xy=(x, y), radius=center_noise_displacement, color='0.7', fill=False, zorder=0))
            ax.add_patch(plt.Circle(xy=(x, y), radius=center_noise_displacement, color='0.95', fill=True, zorder=-1))

        for i_row in range(paths_number):
            x = df_x.iloc[i_row, -1:].to_numpy()
            y = df_y.iloc[i_row, -1:].to_numpy()
            color = plt.get_cmap("brg")(colors_list[i_row])
            ax.scatter(x=x, y=y, s=30, color=color, marker=markers_list[i_row % markers_list_length], linewidth=1, zorder=2)

    plt.show()
    fig.savefig("output\\SpatioTemporalCircularMotionApproachGenerator_output_file_demo_5.png", bbox_inches='tight')
    fig.savefig("output\\SpatioTemporalCircularMotionApproachGenerator_output_file_demo_5.svg", bbox_inches='tight')
    fig.savefig("output\\SpatioTemporalCircularMotionApproachGenerator_output_file_demo_5.eps", bbox_inches='tight')


def main():
    visualize_demo_1()
    visualize_demo_2()
    visualize_demo_3()
    visualize_demo_4()
    visualize_demo_5()


if __name__ == "__main__":
    main()
