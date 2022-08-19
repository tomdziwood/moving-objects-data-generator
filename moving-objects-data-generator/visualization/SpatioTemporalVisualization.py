import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import matplotlib.animation as plt_ani


def visualize_stack(input_filename="input_file.txt"):
    print("SpatioTemporalVisualization visualize_stack()")

    df = pd.read_csv(input_filename, sep=';', header=None, comment="#")
    df.columns = ["time_frame", "feature_id", "feature_instance_id", "x", "y"]
    time_frames = df.time_frame.unique()
    sorted(time_frames)
    print("time_frames size: %d" % time_frames.size)

    (x_min, y_min) = (np.int32(df.x.min()), np.int32(df.y.min()))
    print("min coor:\t(%d, %d)" % (x_min, y_min))
    (x_max, y_max) = (np.int32(df.x.max()), np.int32(df.y.max()))
    print("max coor:\t(%d, %d)" % (x_max, y_max))
    lim_margin_percent = 0.01
    xlim = [x_min - lim_margin_percent * (x_max - x_min), x_max + lim_margin_percent * (x_max - x_min)]
    ylim = [y_min - lim_margin_percent * (y_max - y_min), y_max + lim_margin_percent * (y_max - y_min)]

    ncols = 1
    nrows = time_frames.size
    plt.rcParams["figure.figsize"] = (ncols * 10, nrows * 7)
    plt.suptitle("Lokalizacja obiektów\nw kolejnych momentach czasowych", fontsize=30, y=0.92)

    i_subplot = 1
    for time_frame in time_frames:
        print("time_frame=%d" % time_frame)
        df_tf = df[df.time_frame == time_frame]

        plt.subplot(nrows, ncols, i_subplot)
        plt.scatter(x=df_tf.x, y=df_tf.y)
        plt.title("czas: %d" % time_frame)

        ax = plt.gca()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        i_subplot += 1

    # plt.show()
    plt.savefig("output\\SpatioTemporalVisualization.png", dpi=300)
    plt.savefig("output\\SpatioTemporalVisualization.svg", format="svg")


def animate(frame, *fargs):
    (df, ax, xlim, ylim) = fargs
    ax.clear()

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    df_tf = df[df.time_frame == frame]
    ax.scatter(x=df_tf.x, y=df_tf.y)
    plt.title("czas: %d" % (frame + 1))


def visualize_gif(input_filename="input_file.txt", output_filename=None, fps=1):
    print("SpatioTemporalVisualization visualize_gif()")

    df = pd.read_csv(input_filename, sep=';', header=None, comment="#")
    df.columns = ["time_frame", "feature_id", "feature_instance_id", "x", "y"]
    time_frames = df.time_frame.unique()
    sorted(time_frames)
    print("time_frames size: %d" % time_frames.size)

    (x_min, y_min) = (np.int32(df.x.min()), np.int32(df.y.min()))
    print("min coor:\t(%d, %d)" % (x_min, y_min))
    (x_max, y_max) = (np.int32(df.x.max()), np.int32(df.y.max()))
    print("max coor:\t(%d, %d)" % (x_max, y_max))
    lim_margin_percent = 0.01
    xlim = [x_min - lim_margin_percent * (x_max - x_min), x_max + lim_margin_percent * (x_max - x_min)]
    ylim = [y_min - lim_margin_percent * (y_max - y_min), y_max + lim_margin_percent * (y_max - y_min)]

    fig, ax = plt.subplots()
    plt.rcParams["figure.figsize"] = (10, 7)

    fa = plt_ani.FuncAnimation(
        fig=fig,
        func=animate,
        frames=time_frames,
        fargs=(df, ax, xlim, ylim)
    )

    if output_filename is None:
        idx_start = input_filename.rfind('\\') + 1
        idx_stop = input_filename.rfind('.')
        if idx_stop == -1:
            idx_stop = len(input_filename)
        output_filename = "output\\" + input_filename[idx_start:idx_stop] + "_%03.dfps.gif" % fps

    fa.save(
        filename=output_filename,
        writer=plt_ani.PillowWriter(fps=fps),
        progress_callback=lambda i, n: print(f'Saving frame {i} of {n}')
    )


def visualize_route(input_filename="input_file.txt", output_filename=None, start_frame=None, end_frame=None):
    print("SpatioTemporalVisualization visualize_route()")

    df = pd.read_csv(input_filename, sep=';', header=None, comment="#")
    df.columns = ["time_frame", "feature_id", "feature_instance_id", "x", "y"]
    time_frames = df.time_frame.unique()
    sorted(time_frames)
    print("time_frames size: %d" % time_frames.size)
    print(time_frames)
    print(time_frames[0])
    print(time_frames[-1])

    if start_frame is None or start_frame < time_frames[0]:
        start_frame = 0
    if end_frame is None or end_frame > time_frames[-1]:
        end_frame = time_frames[-1]

    (x_min, y_min) = (np.int32(df.x.min()), np.int32(df.y.min()))
    print("min coor:\t(%d, %d)" % (x_min, y_min))
    (x_max, y_max) = (np.int32(df.x.max()), np.int32(df.y.max()))
    print("max coor:\t(%d, %d)" % (x_max, y_max))
    lim_margin_percent = 0.01
    xlim = [x_min - lim_margin_percent * (x_max - x_min), x_max + lim_margin_percent * (x_max - x_min)]
    ylim = [y_min - lim_margin_percent * (y_max - y_min), y_max + lim_margin_percent * (y_max - y_min)]

    ax = plt.gca()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    df_filtered = df[np.logical_and(df.time_frame >= start_frame, df.time_frame <= end_frame)]
    df_sorted = df_filtered.sort_values(['feature_id', 'feature_instance_id', 'time_frame'])
    df_grouped = df_sorted.groupby(['feature_id', 'feature_instance_id'])
    df_listed = df_grouped.agg({'x': lambda x: list(x), 'y': lambda x: list(x)}).reset_index()

    plt.rcParams["figure.figsize"] = (10, 7)
    plt.title("Ścieżki zakreślone przez poruszające się obiekty", fontsize=18)

    objects_number = len(df_listed)

    for index, row in df_listed.iterrows():
        print("Drawing route of object %d out of %d" % (index + 1, objects_number))
        plt.plot(row.x, row.y)

    output_format = "svg"

    if output_filename is None:
        idx_start = input_filename.rfind('\\') + 1
        idx_stop = input_filename.rfind('.')
        if idx_stop == -1:
            idx_stop = len(input_filename)
        output_filename = "output\\Route_" + input_filename[idx_start:idx_stop] + "_%03.d-%03.d.%s" % (start_frame, end_frame, output_format)

    plt.savefig(output_filename, format=output_format)


def visualize_time_frame(input_filename="input_file.txt", output_filename=None, time_frame=0):
    print("SpatioTemporalVisualization visualize_time_frame()")

    df = pd.read_csv(input_filename, sep=';', header=None, comment="#")
    df.columns = ["time_frame", "feature_id", "feature_instance_id", "x", "y"]
    time_frames = df.time_frame.unique()
    sorted(time_frames)
    print("time_frames size: %d" % time_frames.size)

    (x_min, y_min) = (np.int32(df.x.min()), np.int32(df.y.min()))
    print("min coor:\t(%d, %d)" % (x_min, y_min))
    (x_max, y_max) = (np.int32(df.x.max()), np.int32(df.y.max()))
    print("max coor:\t(%d, %d)" % (x_max, y_max))
    lim_margin_percent = 0.01
    xlim = [x_min - lim_margin_percent * (x_max - x_min), x_max + lim_margin_percent * (x_max - x_min)]
    ylim = [y_min - lim_margin_percent * (y_max - y_min), y_max + lim_margin_percent * (y_max - y_min)]

    fig, ax = plt.subplots()

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title("czas: %d" % time_frame)

    plt.rcParams["figure.figsize"] = (10, 7)

    df_selected_time_frame = df[df.time_frame == time_frame]

    markers_list = ["o", ",", "^", "+", "*", "x", "2", (6, 1, 0), (6, 2, 0), "h", "v", "<", ">", "1", "3", "4", "D", "8", "s", "p", "P", "H", "X", "d"]
    markers_list_length = len(markers_list)
    colors_list = ['b', 'r', 'k', 'g', 'c', 'm', 'y']

    markers = [markers_list[x % markers_list_length] for x in df_selected_time_frame.feature_id]
    colors = [colors_list[x // markers_list_length] for x in df_selected_time_frame.feature_id]

    for i in range(len(df_selected_time_frame)):
        plt.scatter(x=df_selected_time_frame.x[i], y=df_selected_time_frame.y[i], s=20, marker=markers[i], color=colors[i], linewidths=0.5, alpha=0.8)

    output_format = "svg"

    if output_filename is None:
        idx_start = input_filename.rfind('\\') + 1
        idx_stop = input_filename.rfind('.')
        if idx_stop == -1:
            idx_stop = len(input_filename)
        output_filename = "output\\Time_frame_" + input_filename[idx_start:idx_stop] + ".%s" % output_format

    plt.savefig(output_filename, format=output_format)


def main():
    print("main()")
    # visualize_stack(input_filename="..\\scripts\\vectorized\\SpatioTemporalBasicGenerator_output_file.txt")
    # visualize_stack(input_filename="..\\algorithms\\generator\\output\\SpatioTemporalBasicGenerator_output_file.txt")

    # visualize_gif(input_filename="..\\scripts\\vectorized\\SpatioTemporalBasicGenerator_output_file.txt")
    # visualize_gif(input_filename="..\\algorithms\\generator\\output\\SpatioTemporalBasicGenerator_output_file.txt", fps=2)
    # visualize_gif(input_filename="..\\algorithms\\generator\\output\\SpatioTemporalBasicGenerator_output_file_2022-07-29_142325.758471.txt", fps=1)
    # visualize_gif(input_filename="..\\algorithms\\generator\\output\\SpatioTemporalStandardGenerator_output_file.txt", fps=1 / 3)
    # visualize_gif(input_filename="..\\algorithms\\generator\\output\\SpatioTemporalStandardGenerator_output_file_2022-08-05_132504.425978.txt", fps=1 / 3)
    # visualize_gif(input_filename="..\\algorithms\\generator\\output\\SpatioTemporalInteractionApproachGenerator_output_file.txt", fps=25)
    # visualize_gif(input_filename="..\\algorithms\\generator\\output\\SpatioTemporalInteractionApproachGenerator_output_file_2022-08-04_120813.994445.txt", fps=25)
    # visualize_gif(input_filename="..\\algorithms\\generator\\output\\SpatioTemporalCircularMotionApproachGenerator_output_file.txt", fps=25)
    # visualize_gif(input_filename="..\\algorithms\\generator\\output\\SpatioTemporalCircularMotionApproachGenerator_output_file_2022-07-29_142325.758471.txt", fps=1)
    visualize_gif(input_filename="..\\algorithms\\generator\\output\\SpatioTemporalOptimalDistanceApproachGenerator_output_file.txt", fps=25)
    # visualize_gif(input_filename="..\\algorithms\\generator\\output\\SpatioTemporalOptimalDistanceApproachGenerator_output_file_2022-08-04_092614.520252.txt", fps=25)
    # visualize_gif(input_filename="..\\algorithms\\generator\\output\\SpatioTemporalTravelApproachGenerator_output_file.txt", fps=25)
    # visualize_gif(input_filename="..\\algorithms\\generator\\output\\SpatioTemporalTravelApproachGenerator_output_file_2022-08-05_131819.469017.txt", fps=10)

    # visualize_route(input_filename="..\\algorithms\\generator\\output\\SpatioTemporalBasicGenerator_output_file.txt")
    # visualize_route(input_filename="..\\algorithms\\generator\\output\\SpatioTemporalBasicGenerator_output_file_2022-07-29_142325.758471.txt")
    # visualize_route(input_filename="..\\algorithms\\generator\\output\\SpatioTemporalStandardGenerator_output_file.txt")
    # visualize_route(input_filename="..\\algorithms\\generator\\output\\SpatioTemporalStandardGenerator_output_file_2022-08-05_132504.425978.txt")
    # visualize_route(input_filename="..\\algorithms\\generator\\output\\SpatioTemporalInteractionApproachGenerator_output_file.txt")
    # visualize_route(input_filename="..\\algorithms\\generator\\output\\SpatioTemporalInteractionApproachGenerator_output_file_2022-08-04_120813.994445.txt")
    # visualize_route(input_filename="..\\algorithms\\generator\\output\\SpatioTemporalCircularMotionApproachGenerator_output_file.txt", start_frame=100, end_frame=150)
    # visualize_route(input_filename="..\\algorithms\\generator\\output\\SpatioTemporalCircularMotionApproachGenerator_output_file_2022-07-29_142325.758471.txt")
    # visualize_route(input_filename="..\\algorithms\\generator\\output\\SpatioTemporalOptimalDistanceApproachGenerator_output_file.txt")
    # visualize_route(input_filename="..\\algorithms\\generator\\output\\SpatioTemporalOptimalDistanceApproachGenerator_output_file_2022-08-04_092614.520252.txt")
    # visualize_route(input_filename="..\\algorithms\\generator\\output\\SpatioTemporalTravelApproachGenerator_output_file.txt")
    # visualize_route(input_filename="..\\algorithms\\generator\\output\\SpatioTemporalTravelApproachGenerator_output_file_2022-08-05_131819.469017.txt")

    # visualize_time_frame(input_filename="..\\algorithms\\generator\\output\\SpatioTemporalStandardGenerator_output_file.txt")


if __name__ == "__main__":
    main()
