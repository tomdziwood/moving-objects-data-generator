import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib import pyplot as plt

from algorithms.generator.SpatioTemporalStandardGenerator import SpatioTemporalStandardGenerator
from algorithms.parameters.StandardParameters import StandardParameters


def visualize_demo_1_generate_data(area, random_seed):
    print("SpatioTemporalStandardGenerator visualize_demo_1_generate_data()")

    sp = StandardParameters(
        area=area,
        cell_size=5,
        n_base=2,
        lambda_1=3,
        lambda_2=3,
        m_clumpy=1,
        m_overlap=1,
        ncfr=0,
        ncfn=0,
        ncf_proportional=False,
        ndf=1,
        ndfn=5,
        random_seed=random_seed,
        persistent_ratio=0.5,
        spatial_prevalence_threshold=0.7,
        time_prevalence_threshold=0.6
    )

    while True:
        stsg = SpatioTemporalStandardGenerator(sp=sp)
        if ((stsg.si.collocation_lengths == np.array([3, 4], dtype=np.int32)).all() and
                (stsg.si.collocation_instances_counts == np.array([4, 3], dtype=np.int32)).all()):
            print(sp.random_seed)
            break

        sp.random_seed += 1

    stsg.generate(
        time_frames_number=4,
        output_filename="data\\SpatioTemporalStandardGenerator_output_file_demo_1.txt",
        output_filename_timestamp=False
    )


def visualize_demo_1():
    print("SpatioTemporalStandardGenerator visualize_demo_1()")

    area = 30
    random_seed = 1911

    visualize_demo_1_generate_data(area=area, random_seed=random_seed)

    df = pd.read_csv("data\\SpatioTemporalStandardGenerator_output_file_demo_1.txt", sep=';', header=None, comment="#")
    df.columns = ["time_frame", "feature_id", "feature_instance_id", "x", "y"]
    time_frames = df.time_frame.unique()
    sorted(time_frames)
    print("time_frames size: %d" % time_frames.size)

    (x_min, y_min) = (np.int32(df.x.min()), np.int32(df.y.min()))
    print("min coor:\t(%d, %d)" % (x_min, y_min))
    (x_max, y_max) = (np.int32(df.x.max()), np.int32(df.y.max()))
    print("max coor:\t(%d, %d)" % (x_max, y_max))
    xlim = [0, area]
    ylim = [0, area]

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })

    plt.rcParams['axes.axisbelow'] = True

    cm = 1 / 2.54
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20*cm, 20*cm))
    plt.tight_layout(pad=1.5, w_pad=0, h_pad=2)

    markers_list = ["o", ",", "^", "+", "x", "2", (6, 2, 0), "*", (6, 1, 0), "h", "v", "<", ">", "1", "3", "4", "D", "8", "s", "p", "P", "H", "X", "d"]
    markers_list_length = len(markers_list)
    colors_list = ['b', 'r', 'k', 'g', 'c', 'm', 'y']

    major_ticks = np.arange(0, area + 1, 5)
    minor_ticks = np.arange(0, area + 1, 5)

    for time_frame in time_frames:
        print("time_frame=%d" % time_frame)
        df_tf = df[df.time_frame == time_frame]

        ax = axs[time_frame // 2, time_frame % 2]

        ax.set_title(r"moment czasowy: $%d$" % time_frame)

        ax.set_aspect('equal')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.set_xticks(major_ticks)
        ax.set_yticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(minor_ticks, minor=True)
        ax.grid(which='both', color='0.85', linestyle='-', linewidth=1)

        markers = [markers_list[x % markers_list_length] for x in df_tf.feature_id]
        colors = [colors_list[x // markers_list_length] for x in df_tf.feature_id]
        texts = []

        for i in range(len(df_tf)):
            row = df_tf.iloc[i]
            ax.scatter(x=row.x, y=row.y, s=20, marker=markers[i], color=colors[i], linewidths=0.5, alpha=0.8)
            texts.append(ax.text(row.x, row.y, "%s.%d" % (chr(int(ord('A') + row.feature_id)), row.feature_instance_id), ha='center', va='center', fontsize=9))

        adjust_text(texts)

    plt.show()
    fig.savefig("output\\SpatioTemporalStandardGenerator_output_file_demo_1.png", bbox_inches='tight')
    fig.savefig("output\\SpatioTemporalStandardGenerator_output_file_demo_1.svg", bbox_inches='tight')
    fig.savefig("output\\SpatioTemporalStandardGenerator_output_file_demo_1.eps", bbox_inches='tight')


def visualize_demo_2_generate_data(area, random_seed):
    print("SpatioTemporalStandardGenerator visualize_demo_2_generate_data()")

    sp = StandardParameters(
        area=area,
        cell_size=5,
        n_base=1,
        lambda_1=3,
        lambda_2=5,
        m_clumpy=3,
        m_overlap=2,
        ncfr=0,
        ncfn=0,
        ncf_proportional=False,
        ndf=0,
        ndfn=0,
        random_seed=random_seed,
        persistent_ratio=1.0,
        spatial_prevalence_threshold=0.6,
        time_prevalence_threshold=1.0
    )

    while True:
        stsg = SpatioTemporalStandardGenerator(sp=sp)
        if ((stsg.si.collocation_lengths == np.array([4, 4], dtype=np.int32)).all() and
                (stsg.si.collocation_instances_counts == np.array([4, 3], dtype=np.int32)).all()):
            print(sp.random_seed)
            break

        sp.random_seed += 1

    stsg.generate(
        time_frames_number=2,
        output_filename="data\\SpatioTemporalStandardGenerator_output_file_demo_2.txt",
        output_filename_timestamp=False
    )


def visualize_demo_2():
    print("SpatioTemporalStandardGenerator visualize_demo_2()")

    area = 30
    random_seed = 2704

    visualize_demo_2_generate_data(area=area, random_seed=random_seed)

    df = pd.read_csv("data\\SpatioTemporalStandardGenerator_output_file_demo_2.txt", sep=';', header=None, comment="#")
    df.columns = ["time_frame", "feature_id", "feature_instance_id", "x", "y"]
    time_frames = df.time_frame.unique()
    sorted(time_frames)
    print("time_frames size: %d" % time_frames.size)

    (x_min, y_min) = (np.int32(df.x.min()), np.int32(df.y.min()))
    print("min coor:\t(%d, %d)" % (x_min, y_min))
    (x_max, y_max) = (np.int32(df.x.max()), np.int32(df.y.max()))
    print("max coor:\t(%d, %d)" % (x_max, y_max))
    xlim = [0, area]
    ylim = [0, area]

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })

    plt.rcParams['axes.axisbelow'] = True

    cm = 1 / 2.54
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20*cm, 10*cm))
    plt.tight_layout(pad=1.5, w_pad=0, h_pad=2)

    markers_list = ["o", ",", "^", "+", "x", "2", (6, 2, 0), "*", (6, 1, 0), "h", "v", "<", ">", "1", "3", "4", "D", "8", "s", "p", "P", "H", "X", "d"]
    markers_list_length = len(markers_list)
    colors_list = ['b', 'r', 'k', 'g', 'c', 'm', 'y']

    major_ticks = np.arange(0, area + 1, 5)
    minor_ticks = np.arange(0, area + 1, 5)

    for time_frame in time_frames:
        print("time_frame=%d" % time_frame)
        df_tf = df[df.time_frame == time_frame]

        ax = axs[time_frame]

        ax.set_title(r"moment czasowy: $%d$" % time_frame)

        ax.set_aspect('equal')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.set_xticks(major_ticks)
        ax.set_yticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(minor_ticks, minor=True)
        ax.grid(which='both', color='0.85', linestyle='-', linewidth=1)

        markers = [markers_list[x % markers_list_length] for x in df_tf.feature_id]
        colors = [colors_list[x // markers_list_length] for x in df_tf.feature_id]
        texts = []

        for i in range(len(df_tf)):
            row = df_tf.iloc[i]
            ax.scatter(x=row.x, y=row.y, s=20, marker=markers[i], color=colors[i], linewidths=0.5, alpha=0.8)
            texts.append(ax.text(row.x, row.y, "%s.%d" % (chr(int(ord('A') + row.feature_id)), row.feature_instance_id), ha='center', va='center', fontsize=9))

        adjust_text(texts)

    plt.show()
    fig.savefig("output\\SpatioTemporalStandardGenerator_output_file_demo_2.png", bbox_inches='tight')
    fig.savefig("output\\SpatioTemporalStandardGenerator_output_file_demo_2.svg", bbox_inches='tight')
    fig.savefig("output\\SpatioTemporalStandardGenerator_output_file_demo_2.eps", bbox_inches='tight')


def main():
    visualize_demo_1()
    visualize_demo_2()


if __name__ == "__main__":
    main()
