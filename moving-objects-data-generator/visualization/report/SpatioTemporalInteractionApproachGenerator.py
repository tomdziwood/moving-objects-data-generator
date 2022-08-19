import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from algorithms.enums.InteractionApproachEnums import MassMethod, VelocityMethod, IdenticalFeaturesInteractionMode, DifferentFeaturesInteractionMode
from algorithms.generator.SpatioTemporalInteractionApproachGenerator import SpatioTemporalInteractionApproachGenerator
from algorithms.parameters.InteractionApproachParameters import InteractionApproachParameters
from visualization.report.Utils import visualize_parts


def visualize_demo_1_generate_data():
    print("SpatioTemporalInteractionApproachGenerator visualize_demo_1_generate_data()")

    iap = InteractionApproachParameters(
        area=50,
        cell_size=5,
        n_base=2,
        lambda_1=4,
        lambda_2=1,
        m_clumpy=1,
        m_overlap=1,
        ncfr=0,
        ncfn=0,
        ncf_proportional=False,
        ndf=0,
        ndfn=0,
        random_seed=61,
        spatial_prevalent_ratio=1.0,
        spatial_prevalence_threshold=1.0,
        time_unit=25,
        distance_unit=1.0,
        approx_steps_number=5,
        k_force=1000,
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

    df = pd.read_csv("data\\SpatioTemporalInteractionApproachGenerator_output_file_demo_1.txt", sep=';', header=None, comment="#")
    df.columns = ["time_frame", "feature_id", "feature_instance_id", "x", "y"]
    time_frames = df.time_frame.unique()
    sorted(time_frames)
    print("time_frames size: %d" % time_frames.size)

    start_frame = 0
    end_frame = time_frames[-1]
    # start_frame = 0
    # end_frame = 150

    df_filtered = df[np.logical_and(df.time_frame >= start_frame, df.time_frame <= end_frame)]

    (x_min, y_min) = (np.int32(df_filtered.x.min()), np.int32(df_filtered.y.min()))
    print("min coor:\t(%d, %d)" % (x_min, y_min))
    (x_max, y_max) = (np.int32(df_filtered.x.max()), np.int32(df_filtered.y.max()))
    print("max coor:\t(%d, %d)" % (x_max, y_max))
    # xlim = [10, 60]
    # ylim = [-20, 50]
    xlim = [x_min, x_max]
    ylim = [y_min, y_max]

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })

    plt.rcParams['axes.axisbelow'] = True

    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(20*cm, 20*cm))
    plt.tight_layout(pad=1.5, w_pad=0, h_pad=2)

    colors_list = np.linspace(0, 1, df.feature_id.unique().size)

    ax.set_aspect('equal')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    df_sorted = df_filtered.sort_values(['feature_id', 'feature_instance_id', 'time_frame'])
    df_grouped = df_sorted.groupby(['feature_id', 'feature_instance_id'])
    df_listed = df_grouped.agg({'x': lambda x: list(x), 'y': lambda x: list(x)}).reset_index()

    objects_number = len(df_listed)

    for index, row in df_listed.iterrows():
        print("Drawing route of object %d out of %d" % (index + 1, objects_number))
        ax.plot(row.x, row.y, color=plt.get_cmap("nipy_spectral")(colors_list[row.feature_id]), linewidth=1)

    plt.show()
    fig.savefig("output\\SpatioTemporalInteractionApproachGenerator_output_file_demo_1.png")
    fig.savefig("output\\SpatioTemporalInteractionApproachGenerator_output_file_demo_1.svg")
    fig.savefig("output\\SpatioTemporalInteractionApproachGenerator_output_file_demo_1.eps")


def visualize_demo_2():
    print("SpatioTemporalInteractionApproachGenerator visualize_demo_2()")

    # visualize_demo_1_generate_data()

    visualize_parts("data\\SpatioTemporalInteractionApproachGenerator_output_file_demo_1.txt", equal_aspect=False)

    # df = pd.read_csv("data\\SpatioTemporalInteractionApproachGenerator_output_file_demo_1.txt", sep=';', header=None, comment="#")
    # df.columns = ["time_frame", "feature_id", "feature_instance_id", "x", "y"]
    # time_frames = df.time_frame.unique()
    # sorted(time_frames)
    # print("time_frames size: %d" % time_frames.size)
    #
    # start_frame = 0
    # interval_frames = 100
    #
    # df_filtered = df[np.logical_and(df.time_frame >= start_frame, df.time_frame < start_frame + 4 * interval_frames)]
    #
    # (x_min, y_min) = (np.int32(df_filtered.x.min()), np.int32(df_filtered.y.min()))
    # print("min coor:\t(%d, %d)" % (x_min, y_min))
    # (x_max, y_max) = (np.int32(df_filtered.x.max()), np.int32(df_filtered.y.max()))
    # print("max coor:\t(%d, %d)" % (x_max, y_max))
    # # xlim = [10, 60]
    # # ylim = [-20, 50]
    # xlim = [x_min, x_max]
    # ylim = [y_min, y_max]
    #
    # plt.rcParams.update({
    #     "text.usetex": True,
    #     "font.family": "serif",
    #     "font.serif": ["Computer Modern Roman"],
    # })
    #
    # plt.rcParams['axes.axisbelow'] = True
    #
    # cm = 1 / 2.54
    # fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20*cm, 20*cm))
    # plt.tight_layout(pad=1.5, w_pad=0, h_pad=2)
    #
    # colors_list = np.linspace(0, 1, df.feature_id.unique().size)
    #
    # for time_frames_batch_number in range(4):
    #     start_batch = start_frame + time_frames_batch_number * interval_frames
    #     end_batch = start_batch + interval_frames
    #
    #     df_filtered_batch = df_filtered[np.logical_and(df.time_frame >= start_batch, df.time_frame < end_batch)]
    #
    #     ax = axs[time_frames_batch_number // 2, time_frames_batch_number % 2]
    #
    #     ax.set_title(r"zakres momentow czasowych: $%d$ -- $%d$" % (start_batch, end_batch - 1))
    #
    #     ax.set_aspect('equal')
    #     ax.set_xlim(xlim)
    #     ax.set_ylim(ylim)
    #
    #     df_sorted = df_filtered_batch.sort_values(['feature_id', 'feature_instance_id', 'time_frame'])
    #     df_grouped = df_sorted.groupby(['feature_id', 'feature_instance_id'])
    #     df_listed = df_grouped.agg({'x': lambda x: list(x), 'y': lambda x: list(x)}).reset_index()
    #
    #     objects_number = len(df_listed)
    #
    #     for index, row in df_listed.iterrows():
    #         print("Drawing route of object %d out of %d" % (index + 1, objects_number))
    #         ax.plot(row.x, row.y, color=plt.get_cmap("nipy_spectral")(colors_list[row.feature_id]), linewidth=1)

    fig = plt.gcf()
    plt.show()
    fig.savefig("output\\SpatioTemporalInteractionApproachGenerator_output_file_demo_2.png")
    fig.savefig("output\\SpatioTemporalInteractionApproachGenerator_output_file_demo_2.svg")
    fig.savefig("output\\SpatioTemporalInteractionApproachGenerator_output_file_demo_2.eps")


def visualize_demo_3_generate_data():
    print("SpatioTemporalInteractionApproachGenerator visualize_demo_3_generate_data()")

    iap = InteractionApproachParameters(
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
        random_seed=0,
        spatial_prevalent_ratio=1.0,
        spatial_prevalence_threshold=1.0,
        time_unit=25,
        distance_unit=1.0,
        approx_steps_number=5,
        k_force=1000,
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
        if ((stiag.iai.collocation_lengths == np.array([5], dtype=np.int32)).all() and
                (stiag.iai.collocation_instances_counts == np.array([1], dtype=np.int32)).all()):
            print(iap.random_seed)
            break

        iap.random_seed += 1

    stiag.generate(
        time_frames_number=500,
        output_filename="data\\SpatioTemporalInteractionApproachGenerator_output_file_demo_3.txt",
        output_filename_timestamp=False
    )


def visualize_demo_3():
    print("SpatioTemporalInteractionApproachGenerator visualize_demo_3()")

    visualize_demo_3_generate_data()

    df = pd.read_csv("data\\SpatioTemporalInteractionApproachGenerator_output_file_demo_3.txt", sep=';', header=None, comment="#")
    df.columns = ["time_frame", "feature_id", "feature_instance_id", "x", "y"]
    time_frames = df.time_frame.unique()
    sorted(time_frames)
    print("time_frames size: %d" % time_frames.size)

    start_frame = 0
    end_frame = time_frames[-1]
    # start_frame = 0
    # end_frame = 150

    df_filtered = df[np.logical_and(df.time_frame >= start_frame, df.time_frame <= end_frame)]

    (x_min, y_min) = (np.int32(df_filtered.x.min()), np.int32(df_filtered.y.min()))
    print("min coor:\t(%d, %d)" % (x_min, y_min))
    (x_max, y_max) = (np.int32(df_filtered.x.max()), np.int32(df_filtered.y.max()))
    print("max coor:\t(%d, %d)" % (x_max, y_max))
    # xlim = [10, 60]
    # ylim = [-20, 50]
    xlim = [x_min, x_max]
    ylim = [y_min, y_max]

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })

    plt.rcParams['axes.axisbelow'] = True

    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(20*cm, 20*cm))
    plt.tight_layout(pad=1.5, w_pad=0, h_pad=2)

    ax.set_aspect('equal')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    df_sorted = df_filtered.sort_values(['feature_id', 'feature_instance_id', 'time_frame'])
    df_grouped = df_sorted.groupby(['feature_id', 'feature_instance_id'])
    df_listed = df_grouped.agg({'x': lambda x: list(x), 'y': lambda x: list(x)}).reset_index()

    objects_number = len(df_listed)

    for index, row in df_listed.iterrows():
        print("Drawing route of object %d out of %d" % (index + 1, objects_number))
        ax.plot(row.x, row.y)

    plt.show()
    fig.savefig("output\\SpatioTemporalInteractionApproachGenerator_output_file_demo_3.png")
    fig.savefig("output\\SpatioTemporalInteractionApproachGenerator_output_file_demo_3.svg")
    fig.savefig("output\\SpatioTemporalInteractionApproachGenerator_output_file_demo_3.eps")


def main():
    # visualize_demo_1()
    visualize_demo_2()
    # visualize_demo_3()


if __name__ == "__main__":
    main()
