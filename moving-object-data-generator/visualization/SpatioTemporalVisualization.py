import matplotlib.pyplot as plt
import pandas as pd


def visualize(input_file="output_file.txt"):
    print("visualizating...")

    df = pd.read_csv(input_file, sep=';', header=None)
    df.columns = ["time_frame", "feature_id", "feature_instance_id", "x", "y"]
    time_frames = df.time_frame.unique()
    sorted(time_frames)
    print("time_frames=%s" % str(time_frames))

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

        i_subplot += 1

    # plt.show()
    plt.savefig("SpatioTemporalVisualization.png", dpi=300)
    plt.savefig("SpatioTemporalVisualization.svg", format="svg")


def main():
    print("main()")
    visualize(input_file="..\\scripts\\vectorized\\SpatioTemporalStandardGenerator_output_file.txt")


if __name__ == "__main__":
    main()
