import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from algorithms.generator.SpatioTemporalCircularMotionApproachGenerator import SpatioTemporalCircularMotionApproachGenerator
from algorithms.generator.SpatioTemporalStandardGenerator import SpatioTemporalStandardGenerator
from algorithms.parameters.CircularMotionApproachParameters import CircularMotionApproachParameters
from algorithms.parameters.StandardParameters import StandardParameters
from visualization.report.Utils import visualize_neighbours_in_space


def visualize_spatial_data_generate_data(area, random_seed):
    print("ExtraVisualization visualize_spatial_data_generate_data()")

    sp = StandardParameters(
        area=area,
        cell_size=5.0,
        n_base=2,
        lambda_1=2.0,
        lambda_2=2.0,
        m_clumpy=1,
        m_overlap=1,
        ncfr=0.4,
        ncfn=0.3,
        ncf_proportional=False,
        ndf=1,
        ndfn=3,
        random_seed=random_seed,
        persistent_ratio=1.0,
        spatial_prevalence_threshold=1.0,
        time_prevalence_threshold=1.0
    )

    while True:
        stsg = SpatioTemporalStandardGenerator(sp=sp)
        if ((stsg.si.collocation_lengths == np.array([3, 2], dtype=np.int32)).all() and
                (stsg.si.collocation_instances_counts == np.array([2, 3], dtype=np.int32)).all()):
            print(sp.random_seed)
            break

        sp.random_seed += 1

    stsg.generate(
        time_frames_number=1,
        output_filename="data\\ExtraVisualization_output_file_spatial_data.txt",
        output_filename_timestamp=False
    )


def visualize_spatial_data():
    print("ExtraVisualization visualize_spatial_data()")

    area = 20.0
    random_seed = 6824  # 36, 1494, 6821
    distance = 5

    visualize_spatial_data_generate_data(area=area, random_seed=random_seed)

    df = pd.read_csv("data\\ExtraVisualization_output_file_spatial_data.txt", sep=';', header=None, comment="#")
    df.columns = ["time_frame", "feature_id", "feature_instance_id", "x", "y"]

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })

    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(20 * cm, 20 * cm))
    fig.tight_layout(pad=1.5, w_pad=0, h_pad=2)

    visualize_neighbours_in_space(ax, df, area, distance)

    fig = plt.gcf()
    plt.show()
    fig.savefig("output\\ExtraVisualization_output_file_spatial_data.png", bbox_inches='tight')
    fig.savefig("output\\ExtraVisualization_output_file_spatial_data.svg", bbox_inches='tight')
    fig.savefig("output\\ExtraVisualization_output_file_spatial_data.eps", bbox_inches='tight')


def visualize_spatiotemporal_data_generate_data(area, random_seed):
    print("ExtraVisualization visualize_spatiotemporal_data_generate_data()")

    cmap = CircularMotionApproachParameters(
        area=area,
        cell_size=5.0,
        n_base=1,
        lambda_1=3.0,
        lambda_2=3.0,
        m_clumpy=1,
        m_overlap=1,
        ncfr=0.7,
        ncfn=0.2,
        ncf_proportional=False,
        ndf=1,
        ndfn=2,
        random_seed=random_seed,
        spatial_prevalent_ratio=1.0,
        spatial_prevalence_threshold=1.0,
        circle_chain_size=2,
        omega_min=np.pi / 8,
        omega_max=np.pi / 4,
        circle_r_min=2.0,
        circle_r_max=3.0,
        center_noise_displacement=0.5
    )

    while True:
        stcmag = SpatioTemporalCircularMotionApproachGenerator(cmap=cmap)
        if ((stcmag.cmai.collocation_lengths == np.array([3], dtype=np.int32)).all() and
                (stcmag.cmai.collocation_instances_counts == np.array([3], dtype=np.int32)).all()):
            print(cmap.random_seed)
            break

        cmap.random_seed += 1

    stcmag.generate(
        time_frames_number=4,
        output_filename="data\\ExtraVisualization_output_file_spatiotemporal_data.txt",
        output_filename_timestamp=False
    )


def visualize_spatiotemporal_data():
    print("ExtraVisualization visualize_spatiotemporal_data()")

    area = 15.0
    random_seed = 316  # 56, 160, 316, 381
    distance = 4

    visualize_spatiotemporal_data_generate_data(area=area, random_seed=random_seed)

    df = pd.read_csv("data\\ExtraVisualization_output_file_spatiotemporal_data.txt", sep=';', header=None, comment="#")
    df.columns = ["time_frame", "feature_id", "feature_instance_id", "x", "y"]
    time_frames = df.time_frame.unique()
    sorted(time_frames)
    print("time_frames size: %d" % time_frames.size)

    (x_min, y_min) = (np.int32(df.x.min()), np.int32(df.y.min()))
    print("min coor:\t(%d, %d)" % (x_min, y_min))
    (x_max, y_max) = (np.int32(df.x.max()), np.int32(df.y.max()))
    print("max coor:\t(%d, %d)" % (x_max, y_max))
    xlim = [x_min - 2, x_max + 2]
    ylim = [y_min - 2, y_max + 2]

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })

    cm = 1 / 2.54
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20 * cm, 20 * cm))
    fig.tight_layout(pad=0, w_pad=0, h_pad=0)

    for time_frame in time_frames:
        print("time_frame=%d" % time_frame)
        df_tf = df[df.time_frame == time_frame]

        ax = axs[time_frame // 2, time_frame % 2]

        ax.set_title(r"moment czasowy: $%d$" % time_frame)

        visualize_neighbours_in_space(ax, df_tf, area, distance, xlim=xlim, ylim=ylim, markersize=36, markerlinewidth=1.0, linewidth=1.0, fontsize=12)

    fig = plt.gcf()
    plt.show()
    fig.savefig("output\\ExtraVisualization_output_file_spatiotemporal_data.png", bbox_inches='tight')
    fig.savefig("output\\ExtraVisualization_output_file_spatiotemporal_data.svg", bbox_inches='tight')
    fig.savefig("output\\ExtraVisualization_output_file_spatiotemporal_data.eps", bbox_inches='tight')


def main():
    visualize_spatial_data()
    visualize_spatiotemporal_data()


if __name__ == "__main__":
    main()
