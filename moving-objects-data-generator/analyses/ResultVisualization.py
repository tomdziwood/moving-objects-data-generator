import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def visualize_minfreq(input_filename):
    print("ResultVisualization visualize_minfreq()")

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"]
    })

    # plt.rc('text.latex', preamble=r"\usepackage[T1]{fontenc}")

    plt.rcParams['axes.axisbelow'] = True

    df = pd.read_csv(input_filename, sep=',')

    columns_flags_basic = df.columns.isin(["filename", "maxdist", "minprev", "minfreq", "time"])
    columns_flags_pattern_instances_size = df.columns.str.startswith('pattern_instances_size_')
    columns_flags_pattern_colocations_size = df.columns.str.startswith('pattern_colocations_size_')
    df = df[df.columns[columns_flags_basic + columns_flags_pattern_instances_size + columns_flags_pattern_colocations_size]]

    minfreqs = df.minfreq.unique()
    print(minfreqs)

    suptitles = [
        "Generator oparty na zastosowaniu ruchu jednostajnego po okregu",
        "Generator oparty na zasadzie wzajemnego oddziaływania",
        "Generator oparty na metodzie zachowania optymalnego dystansu",
        "Standardowy generator",
        "Generator oparty na wyznaczaniu celu podróży"
    ]

    markers_list = ["o", "s", "^", "+", "x", "2", (6, 2, 0), "*", (6, 1, 0)]

    filenames = df.filename.unique()
    for filename_idx, filename in enumerate(filenames):
        idx = filename.rfind('\\') + 1
        output_filename = filename[idx:]
        idx = output_filename.find('_')
        output_filename = output_filename[:idx]
        output_filename = "minfreq_" + output_filename

        df_selected_filename = df[df.filename == filename]
        print("df_selected_filename.shape:\t%s" % str(df_selected_filename.shape))

        df_pattern_instances_size = df_selected_filename[df_selected_filename.columns[df_selected_filename.columns.str.startswith('pattern_instances_size_')]]
        df_pattern_instances_size = df_pattern_instances_size.loc[:, df_pattern_instances_size.isnull().sum() != len(df_pattern_instances_size)]
        df_pattern_instances_size = df_pattern_instances_size.fillna(0)
        df_pattern_instances_size = df_pattern_instances_size.astype(int)
        df_pattern_instances_size = df_pattern_instances_size.loc[:, ~(df_pattern_instances_size == 0).all()]
        print("\npattern_instances_size:\n%s" % df_pattern_instances_size.to_string())

        df_pattern_colocations_size = df_selected_filename[df_selected_filename.columns[df_selected_filename.columns.str.startswith('pattern_colocations_size_')]]
        df_pattern_colocations_size = df_pattern_colocations_size.loc[:, df_pattern_colocations_size.isnull().sum() != len(df_pattern_colocations_size)]
        df_pattern_colocations_size = df_pattern_colocations_size.fillna(0)
        df_pattern_colocations_size = df_pattern_colocations_size.astype(int)
        df_pattern_colocations_size = df_pattern_colocations_size.loc[:, ~(df_pattern_colocations_size == 0).all()]
        print("\npattern_colocations_size:\n%s" % df_pattern_colocations_size.to_string())

        color_map = []
        for i in range(len(df_pattern_colocations_size.columns)):
            v = 0.8 - 0.8 * i / (len(df_pattern_colocations_size.columns) - 1)
            if v < 0:
                v = 0
            color_map.append((v, v, v))

        cm = 1 / 2.54
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20 * cm, 10 * cm))
        plt.tight_layout(pad=1.5, w_pad=2, h_pad=0, rect=[0, 0.03, 0.93, 0.92])

        fig.suptitle(suptitles[filename_idx], fontsize=14)

        ax = axs[0]

        ax.set_xlabel(r"Próg minimalnej powszechności czasowej")
        ax.set_title(r"Liczba instancji wzorców o danej długości", fontsize=11)

        column_idx = 0
        for (columnName, columnData) in df_pattern_instances_size.iteritems():
            idx = len(columnData)
            idxs = np.where(columnData.values == 0)[0]
            if idxs.size > 0:
                idx = idxs[0]

            marker = markers_list[column_idx]

            if idx <= 1:
                ax.scatter(minfreqs[:idx], columnData.values[:idx], marker=marker, s=49, color=color_map[column_idx], label=r"$%d$" % (column_idx + 2))
            else:
                ax.plot(minfreqs[:idx], columnData.values[:idx], marker=marker, markersize=7, linewidth=1.5, linestyle='dashed', color=color_map[column_idx], label=r"$%d$" % (column_idx + 2))
            column_idx += 1

        xlim = [0, 1]
        ylim = [0, ax.get_ylim()[1]]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks(minfreqs)
        # ax.set_yscale('log')

        ax = axs[1]

        ax.set_xlabel(r"Próg minimalnej powszechności czasowej")
        ax.set_title(r"Liczba wzorców o danej długości", fontsize=11)

        column_idx = 0
        for (columnName, columnData) in df_pattern_colocations_size.iteritems():
            idx = len(columnData)
            idxs = np.where(columnData.values == 0)[0]
            if idxs.size > 0:
                idx = idxs[0]

            marker = markers_list[column_idx]

            if idx <= 1:
                ax.scatter(minfreqs[:idx], columnData.values[:idx], marker=marker, s=49, color=color_map[column_idx], label=r"$%d$" % (column_idx + 2))
            else:
                ax.plot(minfreqs[:idx], columnData.values[:idx], marker=marker, markersize=7, linewidth=1.5, linestyle='dashed', color=color_map[column_idx], label=r"$%d$" % (column_idx + 2))
            column_idx += 1

        ax.legend(title=r"\noindent Długość \\ \vspace{20mm} wzorca", loc='center left', bbox_to_anchor=(1, 0.5))

        xlim = [0, 1]
        ylim = [0, ax.get_ylim()[1]]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks(minfreqs)
        # ax.set_yscale('log')

        # plt.show()
        fig.savefig("output\\" + output_filename + ".png", bbox_inches='tight')
        fig.savefig("output\\" + output_filename + ".svg", bbox_inches='tight')
        # fig.savefig("output\\" + output_filename + ".eps", bbox_inches='tight')


def visualize_minprev(input_filename):
    print("ResultVisualization visualize_minprev()")

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"]
    })

    # plt.rc('text.latex', preamble=r"\usepackage[T1]{fontenc}")

    plt.rcParams['axes.axisbelow'] = True

    df = pd.read_csv(input_filename, sep=',')

    columns_flags_basic = df.columns.isin(["filename", "maxdist", "minprev", "minfreq", "time"])
    columns_flags_pattern_instances_size = df.columns.str.startswith('pattern_instances_size_')
    columns_flags_pattern_colocations_size = df.columns.str.startswith('pattern_colocations_size_')
    df = df[df.columns[columns_flags_basic + columns_flags_pattern_instances_size + columns_flags_pattern_colocations_size]]

    minprevs = df.minprev.unique()
    print(minprevs)

    suptitles = [
        "Generator oparty na zastosowaniu ruchu jednostajnego po okregu",
        "Generator oparty na zasadzie wzajemnego oddziaływania",
        "Generator oparty na metodzie zachowania optymalnego dystansu",
        "Standardowy generator",
        "Generator oparty na wyznaczaniu celu podróży"
    ]

    markers_list = ["o", "s", "^", "+", "x", "2", (6, 2, 0), "*", (6, 1, 0)]

    filenames = df.filename.unique()
    for filename_idx, filename in enumerate(filenames):
        idx = filename.rfind('\\') + 1
        output_filename = filename[idx:]
        idx = output_filename.find('_')
        output_filename = output_filename[:idx]
        output_filename = "minprev_" + output_filename

        df_selected_filename = df[df.filename == filename]
        print("df_selected_filename.shape:\t%s" % str(df_selected_filename.shape))

        df_pattern_instances_size = df_selected_filename[df_selected_filename.columns[df_selected_filename.columns.str.startswith('pattern_instances_size_')]]
        df_pattern_instances_size = df_pattern_instances_size.loc[:, df_pattern_instances_size.isnull().sum() != len(df_pattern_instances_size)]
        df_pattern_instances_size = df_pattern_instances_size.fillna(0)
        df_pattern_instances_size = df_pattern_instances_size.astype(int)
        df_pattern_instances_size = df_pattern_instances_size.loc[:, ~(df_pattern_instances_size == 0).all()]
        print("\npattern_instances_size:\n%s" % df_pattern_instances_size.to_string())

        df_pattern_colocations_size = df_selected_filename[df_selected_filename.columns[df_selected_filename.columns.str.startswith('pattern_colocations_size_')]]
        df_pattern_colocations_size = df_pattern_colocations_size.loc[:, df_pattern_colocations_size.isnull().sum() != len(df_pattern_colocations_size)]
        df_pattern_colocations_size = df_pattern_colocations_size.fillna(0)
        df_pattern_colocations_size = df_pattern_colocations_size.astype(int)
        df_pattern_colocations_size = df_pattern_colocations_size.loc[:, ~(df_pattern_colocations_size == 0).all()]
        print("\npattern_colocations_size:\n%s" % df_pattern_colocations_size.to_string())

        color_map = []
        for i in range(len(df_pattern_colocations_size.columns)):
            v = 0.8 - 0.8 * i / (len(df_pattern_colocations_size.columns) - 1)
            if v < 0:
                v = 0
            color_map.append((v, v, v))

        cm = 1 / 2.54
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20 * cm, 10 * cm))
        plt.tight_layout(pad=1.5, w_pad=2, h_pad=0, rect=[0, 0.03, 0.93, 0.92])

        fig.suptitle(suptitles[filename_idx], fontsize=14)

        ax = axs[0]

        ax.set_xlabel(r"Próg minimalnej powszechności przestrzennej")
        ax.set_title(r"Liczba instancji wzorców o danej długości", fontsize=11)

        column_idx = 0
        for (columnName, columnData) in df_pattern_instances_size.iteritems():
            idx = len(columnData)
            idxs = np.where(columnData.values == 0)[0]
            if idxs.size > 0:
                idx = idxs[0]

            marker = markers_list[column_idx]

            if idx <= 1:
                ax.scatter(minprevs[:idx], columnData.values[:idx], marker=marker, s=49, color=color_map[column_idx], label=r"$%d$" % (column_idx + 2))
            else:
                ax.plot(minprevs[:idx], columnData.values[:idx], marker=marker, markersize=7, linewidth=1.5, linestyle='dashed', color=color_map[column_idx], label=r"$%d$" % (column_idx + 2))
            column_idx += 1

        xlim = [0.25, 0.75]
        ylim = [0, ax.get_ylim()[1]]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks(minprevs)
        # ax.set_yscale('log')

        ax = axs[1]

        ax.set_xlabel(r"Próg minimalnej powszechności przestrzennej")
        ax.set_title(r"Liczba wzorców o danej długości", fontsize=11)

        column_idx = 0
        for (columnName, columnData) in df_pattern_colocations_size.iteritems():
            idx = len(columnData)
            idxs = np.where(columnData.values == 0)[0]
            if idxs.size > 0:
                idx = idxs[0]

            marker = markers_list[column_idx]

            if idx <= 1:
                ax.scatter(minprevs[:idx], columnData.values[:idx], marker=marker, s=49, color=color_map[column_idx], label=r"$%d$" % (column_idx + 2))
            else:
                ax.plot(minprevs[:idx], columnData.values[:idx], marker=marker, markersize=7, linewidth=1.5, linestyle='dashed', color=color_map[column_idx], label=r"$%d$" % (column_idx + 2))
            column_idx += 1

        ax.legend(title=r"\noindent Długość \\ \vspace{20mm} wzorca", loc='center left', bbox_to_anchor=(1, 0.5))

        xlim = [0.25, 0.75]
        ylim = [0, ax.get_ylim()[1]]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks(minprevs)
        # ax.set_yscale('log')

        # plt.show()
        fig.savefig("output\\" + output_filename + ".png", bbox_inches='tight')
        fig.savefig("output\\" + output_filename + ".svg", bbox_inches='tight')
        # fig.savefig("output\\" + output_filename + ".eps", bbox_inches='tight')


def extract_filename_parameter_values_regex(filenames, expression, type_func):
    time_prevalence_thresholds_set = set()
    regex_time_prevalence_threshold = re.compile(expression)

    for filename in filenames:
        match = regex_time_prevalence_threshold.match(filename)
        if match is not None:
            time_prevalence_thresholds_set.add(type_func(match.group(1)))

    return sorted(time_prevalence_thresholds_set)


def visualize_parameter_value_change(input_filename, parameter_name, parameter_value_regex, parameter_dtype, suptitle, xlabel, xlim):
    print("ResultVisualization visualize_parameter_change()")

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"]
    })

    df = pd.read_csv(input_filename, sep=',')

    columns_flags_basic = df.columns.isin(["filename", "maxdist", "minprev", "minfreq", "time"])
    columns_flags_pattern_instances_size = df.columns.str.startswith('pattern_instances_size_')
    columns_flags_pattern_colocations_size = df.columns.str.startswith('pattern_colocations_size_')
    df = df[df.columns[columns_flags_basic + columns_flags_pattern_instances_size + columns_flags_pattern_colocations_size]]

    filenames = df.filename.unique()
    print(filenames.size)
    for (idx, filename) in enumerate(df.iloc[::50].filename):
        print("%d\t%s" % (idx * 50, filename))

    df[parameter_name] = df.filename.str.extract(parameter_value_regex)

    df_pattern_instances_size = df[df.columns[df.columns.str.startswith('pattern_instances_size_')]]
    df_pattern_instances_size = df_pattern_instances_size.loc[:, df_pattern_instances_size.isnull().sum() != len(df_pattern_instances_size)]
    df_pattern_instances_size = df_pattern_instances_size.fillna(0)
    df_pattern_instances_size = df_pattern_instances_size.astype(int)
    df_pattern_instances_size = df_pattern_instances_size.loc[:, ~(df_pattern_instances_size == 0).all()]
    df_pattern_instances_size[parameter_name] = df[parameter_name]
    print("\npattern_instances_size:\n%s" % df_pattern_instances_size.iloc[::50, :].to_string())

    df_pattern_colocations_size = df[df.columns[df.columns.str.startswith('pattern_colocations_size_')]]
    df_pattern_colocations_size = df_pattern_colocations_size.loc[:, df_pattern_colocations_size.isnull().sum() != len(df_pattern_colocations_size)]
    df_pattern_colocations_size = df_pattern_colocations_size.fillna(0)
    df_pattern_colocations_size = df_pattern_colocations_size.astype(int)
    df_pattern_colocations_size = df_pattern_colocations_size.loc[:, ~(df_pattern_colocations_size == 0).all()]
    df_pattern_colocations_size[parameter_name] = df[parameter_name]
    print("\npattern_colocations_size:\n%s" % df_pattern_colocations_size.iloc[::50, :].to_string())

    df_pattern_instances_size = df_pattern_instances_size.groupby(parameter_name).mean()
    print(df_pattern_instances_size.to_string())

    df_pattern_colocations_size = df_pattern_colocations_size.groupby(parameter_name).mean()
    print(df_pattern_colocations_size.to_string())

    parameter_values_array = df_pattern_instances_size.index.to_numpy(dtype=parameter_dtype)

    markers_list = ["o", "s", "^", "+", "x", "2", (6, 2, 0), "*", (6, 1, 0)]

    color_map = []
    for i in range(len(df_pattern_colocations_size.columns)):
        v = 0.8 - 0.8 * i / (len(df_pattern_colocations_size.columns) - 1)
        if v < 0:
            v = 0
        color_map.append((v, v, v))

    idx = input_filename.rfind('\\') + 1
    output_filename = input_filename[idx:]
    idx = output_filename.find('.')
    output_filename = "parameter_value_change_" + output_filename[:idx]

    cm = 1 / 2.54
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20 * cm, 10 * cm))
    plt.tight_layout(pad=1.5, w_pad=2, h_pad=0, rect=[0, 0.03, 0.93, 0.92])

    fig.suptitle(suptitle, fontsize=14)

    ax = axs[0]

    ax.set_xlabel(xlabel)
    ax.set_title(r"Liczba instancji wzorców o danej długości", fontsize=11)

    column_idx = 0
    for (columnName, columnData) in df_pattern_instances_size.iteritems():
        marker = markers_list[column_idx]

        idx_start = 0
        while idx_start < len(columnData):
            while (idx_start < len(columnData)) and (columnData[idx_start] == 0):
                idx_start += 1

            idx_stop = idx_start
            while (idx_stop < len(columnData)) and (columnData[idx_stop] != 0):
                idx_stop += 1

            if idx_start < len(columnData):
                if idx_start == idx_stop:
                    ax.scatter(parameter_values_array[idx_start], columnData.values[idx_start], marker=marker, s=49, color=color_map[column_idx])
                else:
                    ax.plot(parameter_values_array[idx_start:idx_stop], columnData.values[idx_start:idx_stop], marker=marker, markersize=7, linewidth=1.5, linestyle='dashed', color=color_map[column_idx])

            idx_start = idx_stop

        column_idx += 1

    # ylim = [0, ax.get_ylim()[1]]
    ax.set_xlim(xlim)
    # ax.set_ylim(ylim)
    ax.set_xticks(parameter_values_array)
    ax.set_yscale('log')

    ax = axs[1]

    ax.set_xlabel(xlabel)
    ax.set_title(r"Liczba wzorców o danej długości", fontsize=11)

    column_idx = 0
    for (columnName, columnData) in df_pattern_colocations_size.iteritems():
        marker = markers_list[column_idx]
        legend_label_added = False

        idx_start = 0
        while idx_start < len(columnData):
            while (idx_start < len(columnData)) and (columnData[idx_start] == 0):
                idx_start += 1

            idx_stop = idx_start
            while (idx_stop < len(columnData)) and (columnData[idx_stop] != 0):
                idx_stop += 1

            if idx_start < len(columnData):
                if idx_start == idx_stop:
                    if not legend_label_added:
                        ax.scatter(parameter_values_array[idx_start], columnData.values[idx_start], marker=marker, s=49, color=color_map[column_idx], label=r"$%d$" % (column_idx + 2))
                        legend_label_added = True
                    else:
                        ax.scatter(parameter_values_array[idx_start], columnData.values[idx_start], marker=marker, s=49, color=color_map[column_idx])
                else:
                    if not legend_label_added:
                        ax.plot(parameter_values_array[idx_start:idx_stop], columnData.values[idx_start:idx_stop], marker=marker, markersize=7, linewidth=1.5, linestyle='dashed', color=color_map[column_idx], label=r"$%d$" % (column_idx + 2))
                        legend_label_added = True
                    else:
                        ax.plot(parameter_values_array[idx_start:idx_stop], columnData.values[idx_start:idx_stop], marker=marker, markersize=7, linewidth=1.5, linestyle='dashed', color=color_map[column_idx])

            idx_start = idx_stop

        column_idx += 1

    ax.legend(title=r"\noindent Długość \\ \vspace{20mm} wzorca", loc='center left', bbox_to_anchor=(1, 0.5))

    # ylim = [0, ax.get_ylim()[1]]
    ax.set_xlim(xlim)
    # ax.set_ylim(ylim)
    ax.set_xticks(parameter_values_array)
    ax.set_yscale('log')

    # plt.show()
    fig.savefig("output\\" + output_filename + ".png", bbox_inches='tight')
    fig.savefig("output\\" + output_filename + ".svg", bbox_inches='tight')
    # fig.savefig("output\\" + output_filename + ".eps", bbox_inches='tight')


def visualize_batch_04_execute_01():
    print("ResultVisualization visualize_batch_04_execute_01()")

    input_filename = "result\\batch_04_execute_01.csv"
    parameter_name = "time_prevalence_threshold"
    parameter_value_regex = ".*time_prevalence_threshold_(?P<time_prevalence_threshold>[0-9.]*).*"
    parameter_dtype = np.float64
    suptitle = "Standardowy generator"
    xlabel = r"Wartość parametru generatora $\theta_t$"
    xlim = [-0.05, 1.05]

    visualize_parameter_value_change(
        input_filename=input_filename,
        parameter_name=parameter_name,
        parameter_value_regex=parameter_value_regex,
        parameter_dtype=parameter_dtype,
        suptitle=suptitle,
        xlabel=xlabel,
        xlim=xlim
    )


def visualize_batch_05_execute_01():
    print("ResultVisualization visualize_batch_05_execute_01()")

    input_filename = "result\\batch_05_execute_01.csv"
    parameter_name = "velocity_limit"
    parameter_value_regex = ".*velocity_limit_(?P<velocity_limit>[0-9.]*).*"
    parameter_dtype = np.float64
    suptitle = "Generator oparty na zasadzie wzajemnego oddziaływania"
    xlabel = r"Wartość parametru generatora $v_{max}$"
    xlim = [2.5, 52.5]

    visualize_parameter_value_change(
        input_filename=input_filename,
        parameter_name=parameter_name,
        parameter_value_regex=parameter_value_regex,
        parameter_dtype=parameter_dtype,
        suptitle=suptitle,
        xlabel=xlabel,
        xlim=xlim
    )


def visualize_batch_06_execute_01():
    print("ResultVisualization visualize_batch_06_execute_01()")

    input_filename = "result\\batch_06_execute_01.csv"
    parameter_name = "velocity_limit"
    parameter_value_regex = ".*velocity_limit_(?P<velocity_limit>[0-9.]*).*"
    parameter_dtype = np.float64
    suptitle = "Generator oparty na metodzie zachowania optymalnego dystansu"
    xlabel = r"Wartość parametru generatora $v_{max}$"
    xlim = [2.5, 52.5]

    visualize_parameter_value_change(
        input_filename=input_filename,
        parameter_name=parameter_name,
        parameter_value_regex=parameter_value_regex,
        parameter_dtype=parameter_dtype,
        suptitle=suptitle,
        xlabel=xlabel,
        xlim=xlim
    )


def visualize_batch_07_execute_01():
    print("ResultVisualization visualize_batch_07_execute_01()")

    input_filename = "result\\batch_07_execute_01.csv"
    parameter_name = "center_noise_displacement"
    parameter_value_regex = ".*center_noise_displacement_(?P<center_noise_displacement>[0-9.]*).*"
    parameter_dtype = np.float64
    suptitle = "Generator oparty na zastosowaniu ruchu jednostajnego po okregu"
    xlabel = r"Wartość parametru generatora $r_{displacement}$"
    xlim = [0.2, 4.2]

    visualize_parameter_value_change(
        input_filename=input_filename,
        parameter_name=parameter_name,
        parameter_value_regex=parameter_value_regex,
        parameter_dtype=parameter_dtype,
        suptitle=suptitle,
        xlabel=xlabel,
        xlim=xlim
    )


def visualize_batch_08_execute_01():
    print("ResultVisualization visualize_batch_08_execute_01()")

    input_filename = "result\\batch_08_execute_01.csv"
    parameter_name = "waiting_time_frames"
    parameter_value_regex = ".*waiting_time_frames_(?P<waiting_time_frames>[0-9]*).*"
    parameter_dtype = np.int32
    suptitle = "Generator oparty na wyznaczaniu celu podróży"
    xlabel = r"Wartość parametru generatora $TF_{waiting}$"
    xlim = [20, 420]

    visualize_parameter_value_change(
        input_filename=input_filename,
        parameter_name=parameter_name,
        parameter_value_regex=parameter_value_regex,
        parameter_dtype=parameter_dtype,
        suptitle=suptitle,
        xlabel=xlabel,
        xlim=xlim
    )


def main():
    print("main()")

    # visualize_minfreq(input_filename="result\\batch_02_execute_01.csv")
    # visualize_minprev(input_filename="result\\batch_02_execute_02.csv")
    visualize_batch_04_execute_01()
    visualize_batch_05_execute_01()
    visualize_batch_06_execute_01()
    visualize_batch_07_execute_01()
    visualize_batch_08_execute_01()


if __name__ == "__main__":
    main()
