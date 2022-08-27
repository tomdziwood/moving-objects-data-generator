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
            ax.plot(minfreqs[:idx], columnData.values[:idx], marker='o', markersize=5, linewidth=1.5, linestyle='dashed', color=color_map[column_idx], label=r"$%d$" % (column_idx + 2))
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
            ax.plot(minfreqs[:idx], columnData.values[:idx], marker='o', markersize=5, linewidth=1.5, linestyle='dashed', color=color_map[column_idx], label=r"$%d$" % (column_idx + 2))
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
            ax.plot(minprevs[:idx], columnData.values[:idx], marker='o', markersize=5, linewidth=1.5, linestyle='dashed', color=color_map[column_idx], label=r"$%d$" % (column_idx + 2))
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
            ax.plot(minprevs[:idx], columnData.values[:idx], marker='o', markersize=5, linewidth=1.5, linestyle='dashed', color=color_map[column_idx], label=r"$%d$" % (column_idx + 2))
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


def main():
    print("main()")

    visualize_minfreq(input_filename="result\\batch_02_execute_01.csv")
    visualize_minprev(input_filename="result\\batch_02_execute_02.csv")


if __name__ == "__main__":
    main()
