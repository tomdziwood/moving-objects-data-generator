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


def main():
    print("main()")
    # visualize_stack(input_filename="..\\scripts\\vectorized\\SpatioTemporalBasicGenerator_output_file.txt")
    # visualize_stack(input_filename="..\\algorithms\\generator\\output\\SpatioTemporalBasicGenerator_output_file.txt")

    # visualize_gif(input_filename="..\\scripts\\vectorized\\SpatioTemporalBasicGenerator_output_file.txt")
    # visualize_gif(input_filename="..\\algorithms\\generator\\output\\SpatioTemporalBasicGenerator_output_file.txt", fps=2)
    # visualize_gif(input_filename="..\\algorithms\\generator\\output\\SpatioTemporalBasicGenerator_output_file_2022-07-29_142325.758471.txt", fps=1)
    # visualize_gif(input_filename="..\\algorithms\\generator\\output\\SpatioTemporalStandardGenerator_output_file.txt", fps=2)
    # visualize_gif(input_filename="..\\algorithms\\generator\\output\\SpatioTemporalStandardGenerator_output_file_2022-07-29_142325.758471.txt", fps=1)
    # visualize_gif(input_filename="..\\algorithms\\generator\\output\\SpatioTemporalStaticInteractionApproachGenerator_output_file.txt", fps=25)
    visualize_gif(input_filename="..\\algorithms\\generator\\output\\SpatioTemporalStaticInteractionApproachGenerator_output_file_2022-08-04_120813.994445.txt", fps=25)
    # visualize_gif(input_filename="..\\algorithms\\generator\\output\\SpatioTemporalCircularMotionApproachGenerator_output_file.txt", fps=25)
    # visualize_gif(input_filename="..\\algorithms\\generator\\output\\SpatioTemporalCircularMotionApproachGenerator_output_file_2022-07-29_142325.758471.txt", fps=1)
    # visualize_gif(input_filename="..\\algorithms\\generator\\output\\SpatioTemporalOptimalDistanceApproachGenerator_output_file.txt", fps=25)
    # visualize_gif(input_filename="..\\algorithms\\generator\\output\\SpatioTemporalOptimalDistanceApproachGenerator_output_file_2022-08-04_092614.520252.txt", fps=25)
    # visualize_gif(input_filename="..\\algorithms\\generator\\output\\SpatioTemporalTravelApproachGenerator_output_file.txt", fps=2)
    # visualize_gif(input_filename="..\\algorithms\\generator\\output\\SpatioTemporalTravelApproachGenerator_output_file_2022-07-29_103324.421115.txt", fps=25)


if __name__ == "__main__":
    main()
