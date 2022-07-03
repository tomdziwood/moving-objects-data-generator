import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import matplotlib.animation as plt_ani



def visualize_stack(input_file="output_file.txt"):
    print("visualizating...")

    df = pd.read_csv(input_file, sep=';', header=None)
    df.columns = ["time_frame", "feature_id", "feature_instance_id", "x", "y"]
    time_frames = df.time_frame.unique()
    sorted(time_frames)
    print("time_frames=%s" % str(time_frames))

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
    plt.savefig("SpatioTemporalVisualization.png", dpi=300)
    plt.savefig("SpatioTemporalVisualization.svg", format="svg")


def animate(frame, *fargs):
    (df, ax, xlim, ylim) = fargs
    ax.clear()

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    df_tf = df[df.time_frame == frame]
    ax.scatter(x=df_tf.x, y=df_tf.y)
    plt.title("czas: %d" % (frame + 1))


def visualize_gif(input_file="output_file.txt"):
    print("visualizating...")

    df = pd.read_csv(input_file, sep=';', header=None)
    df.columns = ["time_frame", "feature_id", "feature_instance_id", "x", "y"]
    time_frames = df.time_frame.unique()
    sorted(time_frames)
    print("time_frames=%s" % str(time_frames))

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

    plt.rcParams["figure.figsize"] = (10, 7)

    fa = plt_ani.FuncAnimation(
        fig=fig,
        func= animate,
        frames=time_frames,
        interval=1000,
        repeat_delay=1000,
        fargs=(df, ax, xlim, ylim)
    )

    fa.save(
        filename="SpatioTemporalVisualization.gif",
        writer=plt_ani.PillowWriter(fps=1),
        progress_callback=lambda i, n: print(f'Saving frame {i} of {n}')
    )


def main():
    print("main()")
    # visualize_stack(input_file="..\\scripts\\vectorized\\SpatioTemporalStandardGenerator_output_file.txt")
    # visualize_stack(input_file="..\\oop\\SpatioTemporalStandardGenerator_output_file.txt")
    # visualize_stack(input_file="..\\oop\\SpatioTemporalGravitationApproachGenerator_output_file.txt")

    # visualize_gif(input_file="..\\scripts\\vectorized\\SpatioTemporalStandardGenerator_output_file.txt")
    # visualize_gif(input_file="..\\oop\\SpatioTemporalStandardGenerator_output_file.txt")
    visualize_gif(input_file="..\\oop\\SpatioTemporalGravitationApproachGenerator_output_file.txt")


if __name__ == "__main__":
    main()
