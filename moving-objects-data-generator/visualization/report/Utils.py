import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def visualize_x_y(input_file, start_frame=None, end_frame=None, xlim=None, ylim=None, equal_aspect=True, color_per_feature=False, markersize=3, linewidth=1, markevery=10):
    print("Utils visualize_x_y()")

    df = pd.read_csv(input_file, sep=';', header=None, comment="#")
    df.columns = ["time_frame", "feature_id", "feature_instance_id", "x", "y"]
    time_frames = df.time_frame.unique()
    sorted(time_frames)
    print("time_frames size: %d" % time_frames.size)

    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = time_frames[-1]

    df_filtered = df[np.logical_and(df.time_frame >= start_frame, df.time_frame <= end_frame)]

    (x_min, y_min) = (np.int32(df_filtered.x.min()), np.int32(df_filtered.y.min()))
    print("min coor:\t(%d, %d)" % (x_min, y_min))
    (x_max, y_max) = (np.int32(df_filtered.x.max()), np.int32(df_filtered.y.max()))
    print("max coor:\t(%d, %d)" % (x_max, y_max))
    # xlim = [x_min, x_max]
    # ylim = [y_min, y_max]

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })

    plt.rcParams['axes.axisbelow'] = True

    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(20*cm, 20*cm))
    plt.tight_layout(pad=1.5, w_pad=0, h_pad=2)

    markers_list = ["o", "s", "^", "+", "*", "x", "2", (6, 1, 0), (6, 2, 0), "h", "v", "<", ">", "1", "3", "4", "D", "8", "p", "P", "H", "X", "d"]
    markers_list_length = len(markers_list)

    if equal_aspect:
        ax.set_aspect('equal')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlabel(r"x")
    ax.set_ylabel(r"y")

    df_sorted = df_filtered.sort_values(['feature_id', 'feature_instance_id', 'time_frame'])
    df_grouped = df_sorted.groupby(['feature_id', 'feature_instance_id'])
    df_listed = df_grouped.agg({'x': lambda x: list(x), 'y': lambda x: list(x)}).reset_index()

    objects_number = len(df_listed)

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
                markersize=markersize, linewidth=linewidth, markevery=markevery)


def visualize_3d(input_file, start_frame=None, end_frame=None, markersize=3, linewidth=1, markevery=10, elev=20, azim=-60):
    print("Utils visualize_3d()")

    df = pd.read_csv(input_file, sep=';', header=None, comment="#")
    df.columns = ["time_frame", "feature_id", "feature_instance_id", "x", "y"]
    time_frames = df.time_frame.unique()
    sorted(time_frames)
    print("time_frames size: %d" % time_frames.size)

    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = time_frames[-1]

    df_filtered = df[np.logical_and(df.time_frame >= start_frame, df.time_frame <= end_frame)]
    time_frames = df_filtered.time_frame.unique()

    (x_min, y_min) = (np.int32(df_filtered.x.min()), np.int32(df_filtered.y.min()))
    print("min coor:\t(%d, %d)" % (x_min, y_min))
    (x_max, y_max) = (np.int32(df_filtered.x.max()), np.int32(df_filtered.y.max()))
    print("max coor:\t(%d, %d)" % (x_max, y_max))
    # xlim = [x_min, x_max]
    # ylim = [y_min, y_max]

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })

    plt.rcParams['axes.axisbelow'] = True

    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(20*cm, 20*cm), subplot_kw=dict(projection="3d"))
    plt.tight_layout(pad=1.5, w_pad=0, h_pad=2)

    markers_list = ["o", "s", "^", "+", "*", "x", "2", (6, 1, 0), (6, 2, 0), "h", "v", "<", ">", "1", "3", "4", "D", "8", "p", "P", "H", "X", "d"]
    markers_list_length = len(markers_list)
    # colors_list = np.linspace(0, 1, df.feature_id.unique().size)

    # ax.set_xlim(xlim)
    # ax.set_ylim(ylim)

    df_sorted = df_filtered.sort_values(['feature_id', 'feature_instance_id', 'time_frame'])
    df_grouped = df_sorted.groupby(['feature_id', 'feature_instance_id'])
    df_listed = df_grouped.agg({'x': lambda x: list(x), 'y': lambda x: list(x)}).reset_index()

    objects_number = len(df_listed)
    colors_list = np.linspace(0, 1, objects_number)

    for index, row in df_listed.iterrows():
        print("Drawing route of object %d out of %d" % (index + 1, objects_number))
        ax.plot(row.x, row.y, time_frames,
                color=plt.get_cmap("nipy_spectral")(colors_list[index]),
                marker=markers_list[row.feature_id % markers_list_length],
                markersize=markersize, linewidth=linewidth, markevery=markevery)

    ax.set_xlabel(r"x")
    ax.set_ylabel(r"y")
    ax.set_zlabel(r"moment czasowy")
    ax.view_init(elev=elev, azim=azim)


def visualize_perspectives(input_file, start_frame=None, end_frame=None, equal_aspect=True, markersize=3, linewidth=1, markevery=10, markevery3d=20):
    print("Utils visualize_perspectives()")

    df = pd.read_csv(input_file, sep=';', header=None, comment="#")
    df.columns = ["time_frame", "feature_id", "feature_instance_id", "x", "y"]
    time_frames = df.time_frame.unique()
    sorted(time_frames)
    print("time_frames size: %d" % time_frames.size)

    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = time_frames[-1]

    df_filtered = df[np.logical_and(df.time_frame >= start_frame, df.time_frame <= end_frame)]
    time_frames = df_filtered.time_frame.unique()

    (x_min, y_min) = (np.int32(df_filtered.x.min()), np.int32(df_filtered.y.min()))
    print("min coor:\t(%d, %d)" % (x_min, y_min))
    (x_max, y_max) = (np.int32(df_filtered.x.max()), np.int32(df_filtered.y.max()))
    print("max coor:\t(%d, %d)" % (x_max, y_max))
    # xlim = [x_min, x_max]
    # ylim = [y_min, y_max]

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })

    plt.rcParams['axes.axisbelow'] = True

    cm = 1 / 2.54
    fig = plt.figure(figsize=(20*cm, 20*cm))
    plt.tight_layout(pad=2, w_pad=1, h_pad=3)

    markers_list = ["o", "s", "^", "+", "*", "x", "2", (6, 1, 0), (6, 2, 0), "h", "v", "<", ">", "1", "3", "4", "D", "8", "p", "P", "H", "X", "d"]
    markers_list_length = len(markers_list)
    # colors_list = np.linspace(0, 1, df.feature_id.unique().size)

    df_sorted = df_filtered.sort_values(['feature_id', 'feature_instance_id', 'time_frame'])
    df_grouped = df_sorted.groupby(['feature_id', 'feature_instance_id'])
    df_listed = df_grouped.agg({'x': lambda x: list(x), 'y': lambda x: list(x)}).reset_index()

    objects_number = len(df_listed)
    colors_list = np.linspace(0, 1, objects_number)

    # ax = axs[0, 0]
    ax = fig.add_subplot(2, 2, 1)
    ax.set_xlabel(r"x")
    ax.set_ylabel(r"moment czasowy")
    for index, row in df_listed.iterrows():
        print("Drawing route of object %d out of %d" % (index + 1, objects_number))
        ax.plot(row.x, time_frames,
                color=plt.get_cmap("nipy_spectral")(colors_list[index]),
                marker=markers_list[row.feature_id % markers_list_length],
                markersize=markersize, linewidth=linewidth, markevery=markevery)

    # ax = axs[0, 1]
    ax = fig.add_subplot(2, 2, 2)
    ax.set_xlabel(r"y")
    ax.set_ylabel(r"moment czasowy")
    for index, row in df_listed.iterrows():
        print("Drawing route of object %d out of %d" % (index + 1, objects_number))
        ax.plot(row.y, time_frames,
                color=plt.get_cmap("nipy_spectral")(colors_list[index]),
                marker=markers_list[row.feature_id % markers_list_length],
                markersize=markersize, linewidth=linewidth, markevery=markevery)

    # ax = axs[1, 0]
    ax = fig.add_subplot(2, 2, 3)
    ax.set_xlabel(r"x")
    ax.set_ylabel(r"y")
    for index, row in df_listed.iterrows():
        print("Drawing route of object %d out of %d" % (index + 1, objects_number))
        ax.plot(row.x, row.y,
                color=plt.get_cmap("nipy_spectral")(colors_list[index]),
                marker=markers_list[row.feature_id % markers_list_length],
                markersize=markersize, linewidth=linewidth, markevery=markevery)

    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.set_xlabel(r"x")
    ax.set_ylabel(r"y")
    ax.set_zlabel(r"moment czasowy")
    ax.view_init(elev=20, azim=-60)
    for index, row in df_listed.iterrows():
        print("Drawing route of object %d out of %d" % (index + 1, objects_number))
        ax.plot(row.x, row.y, time_frames,
                color=plt.get_cmap("nipy_spectral")(colors_list[index]),
                marker=markers_list[row.feature_id % markers_list_length],
                markersize=markersize, linewidth=linewidth, markevery=markevery3d)


def visualize_parts(input_file, start_frame=None, interval_frames=100, equal_aspect=True, markersize=3, linewidth=1, markevery=10):
    print("Utils visualize_parts()")

    df = pd.read_csv(input_file, sep=';', header=None, comment="#")
    df.columns = ["time_frame", "feature_id", "feature_instance_id", "x", "y"]
    time_frames = df.time_frame.unique()
    sorted(time_frames)
    print("time_frames size: %d" % time_frames.size)

    if start_frame is None:
        start_frame = 0
    if interval_frames is None:
        interval_frames = (time_frames[-1] - 0) // 4

    df_filtered = df[np.logical_and(df.time_frame >= start_frame, df.time_frame < start_frame + 4 * interval_frames)]

    (x_min, y_min) = (np.int32(df_filtered.x.min()), np.int32(df_filtered.y.min()))
    print("min coor:\t(%d, %d)" % (x_min, y_min))
    (x_max, y_max) = (np.int32(df_filtered.x.max()), np.int32(df_filtered.y.max()))
    print("max coor:\t(%d, %d)" % (x_max, y_max))
    xlim = [x_min, x_max]
    ylim = [y_min, y_max]

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })

    plt.rcParams['axes.axisbelow'] = True

    cm = 1 / 2.54
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20*cm, 20*cm))
    plt.tight_layout(pad=2, w_pad=2, h_pad=3)

    markers_list = ["o", "s", "^", "+", "*", "x", "2", (6, 1, 0), (6, 2, 0), "h", "v", "<", ">", "1", "3", "4", "D", "8", "p", "P", "H", "X", "d"]
    markers_list_length = len(markers_list)
    # colors_list = np.linspace(0, 1, df.feature_id.unique().size)

    df_grouped = df_filtered.groupby(['feature_id', 'feature_instance_id'])
    df_listed = df_grouped.agg({'x': lambda x: list(x), 'y': lambda x: list(x)}).reset_index()
    objects_number = len(df_listed)
    colors_list = np.linspace(0, 1, objects_number)

    for time_frames_batch_number in range(4):
        start_batch = start_frame + time_frames_batch_number * interval_frames
        end_batch = start_batch + interval_frames

        df_filtered_batch = df_filtered[np.logical_and(df.time_frame >= start_batch, df.time_frame < end_batch)]

        ax = axs[time_frames_batch_number // 2, time_frames_batch_number % 2]

        ax.set_title(r"zakres momentow czasowych: $%d$ -- $%d$" % (start_batch, end_batch - 1))

        if equal_aspect:
            ax.set_aspect('equal')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.set_xlabel(r"x", labelpad=0)
        ax.set_ylabel(r"y")

        df_sorted = df_filtered_batch.sort_values(['feature_id', 'feature_instance_id', 'time_frame'])
        df_grouped = df_sorted.groupby(['feature_id', 'feature_instance_id'])
        df_listed = df_grouped.agg({'x': lambda x: list(x), 'y': lambda x: list(x)}).reset_index()

        objects_number = len(df_listed)

        for index, row in df_listed.iterrows():
            print("Drawing route of object %d out of %d" % (index + 1, objects_number))
            ax.plot(row.x, row.y,
                    color=plt.get_cmap("nipy_spectral")(colors_list[index]),
                    marker=markers_list[row.feature_id % markers_list_length],
                    markersize=markersize, linewidth=linewidth, markevery=markevery)


def visualize_frame(input_file, time_frame=0, markersize=20, xlim=None, ylim=None):
    print("Utils visualize_frame()")

    df = pd.read_csv(input_file, sep=';', header=None, comment="#")
    df.columns = ["time_frame", "feature_id", "feature_instance_id", "x", "y"]
    time_frames = df.time_frame.unique()
    sorted(time_frames)
    print("time_frames size: %d" % time_frames.size)

    df_selected_time_frame = df[df.time_frame == time_frame]

    (x_min, y_min) = (np.int32(df_selected_time_frame.x.min()), np.int32(df_selected_time_frame.y.min()))
    print("min coor:\t(%d, %d)" % (x_min, y_min))
    (x_max, y_max) = (np.int32(df_selected_time_frame.x.max()), np.int32(df_selected_time_frame.y.max()))
    print("max coor:\t(%d, %d)" % (x_max, y_max))

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })

    plt.rcParams['axes.axisbelow'] = True

    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(20*cm, 20*cm))
    plt.tight_layout(pad=1.5, w_pad=0, h_pad=2)

    markers_list = ["o", "s", "^", "+", "*", "x", "2", (6, 1, 0), (6, 2, 0), "h", "v", "<", ">", "1", "3", "4", "D", "8", "p", "P", "H", "X", "d"]
    markers_list_length = len(markers_list)
    # colors_list = np.linspace(0, 1, df.feature_id.unique().size)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlabel(r"x")
    ax.set_ylabel(r"y")

    objects_number = len(df_selected_time_frame)
    colors_list = np.linspace(0, 1, objects_number)

    markers = [markers_list[x % markers_list_length] for x in df_selected_time_frame.feature_id]

    for i in range(objects_number):
        row = df_selected_time_frame.iloc[[i]]
        ax.scatter(x=row.x, y=row.y, s=markersize,
                   marker=markers[i],
                   color=plt.get_cmap("nipy_spectral")(colors_list[i]),
                   linewidths=0.5, alpha=0.8)
