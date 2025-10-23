"""This is a script which plots from multiple algorithm's regret files."""

from itertools import product, cycle
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from cuts import get_params_from_filename


def map_plot_labels(label):
    if ("p0." in label) and ("1 stages" in label):
        prob = label.split("-")[1][:6]
        return f"UTS_{prob}_1s"
    elif "p0." in label:
        list_size = int(label.split("-")[2].split(" ")[0])
        prob = label.split("-")[1][:6]
        return f"CUTS_{prob}_{list_size}s"
    elif ("full-wifi" in label) and ("1 stages" in label):
        return "TS_WIFI"
    elif "full-wifi" in label:
        list_size = int(label.split("-")[-1].split(" ")[0])
        return f"CTS_{list_size}s_WIFI"
    elif ("full" in label) and ("1 stages" in label):
        return "TS"
    elif "full" in label:
        list_size = int(label.split("-")[2].split(" ")[0])
        return f"CTS_{list_size}s"
    elif "line-csv-wifi" in label:
        list_size = int(label.split("-")[-1].split(" ")[0])
        return f"WIFI-CUTS_line_{list_size}s-WIFI"
    elif "line-csv" in label:
        list_size = int(label.split("-")[-1].split(" ")[0])
        return f"WIFI-CUTS_line_{list_size}s"
    elif "line" in label and ("1 stages" in label):
        return "UTS_line"
    elif "line" in label:
        list_size = int(label.split("-")[2].split(" ")[0])
        return f"CUTS-line_{list_size}s"
    else:
        raise ValueError(f"Unknown label: {label}")


# Function to load data from individual files and plot
def plot_from_multiple_simdatafiles(
    filelist, save_as=None, plot_percentile=True, plot_reward=False
):
    plt.figure(figsize=(12, 8), dpi=300)
    ax = plt.gca()
    linestyles = cycle(
        [
            "-",
            "--",
            "-.",
            ":",
            (0, (3, 1, 1, 1)),
            (5, (10, 3)),
            (0, (5, 10)),
            (0, (3, 10, 1, 10)),
        ]
    )
    for filename in filelist:
        # Check if the file exists
        if os.path.isfile(filename):
            (
                num_arms,
                list_size,
                num_rounds,
                num_simulations,
                graph_type,
            ) = get_params_from_filename(filename)
            # Load data
            data = np.load(filename, allow_pickle=True)
            avg_cumulative_regret = data["avg_cumulative_regret"]
            percentile_2_5 = data["percentile_2_5"]
            percentile_97_5 = data["percentile_97_5"]
            avg_reward = data["avg_reward"]
            percentile_2_5_reward = data["percentile_2_5_reward"]
            percentile_97_5_reward = data["percentile_97_5_reward"]

            # Plot the mean cumulative regret with a shaded area for the 95% confidence interval
            label_tmp = f"CUTS-{graph_type}-{list_size} stages"
            label = map_plot_labels(label_tmp)
            if plot_reward:
                data = avg_reward
            else:
                data = avg_cumulative_regret
            plt.plot(
                data,
                label=label,
                linestyle=next(linestyles),
                # linewidth=2,
            )
            if plot_percentile:
                if plot_reward:
                    plt.fill_between(
                        range(len(avg_reward)),
                        percentile_2_5_reward,
                        percentile_97_5_reward,
                        alpha=0.2,
                    )
                else:
                    plt.fill_between(
                        range(len(avg_cumulative_regret)),
                        percentile_2_5,
                        percentile_97_5,
                        alpha=0.2,
                    )
        else:
            print(f"File {filename} not found. Skipping this file.")

    plt.xlabel("Rounds")
    if plot_reward:
        plt.ylabel("Average Reward")
    else:
        plt.ylabel("Average Cumulative Regret")
    plt.title("Cumulative regret performance of CUTS")
    _plt_lbl_list = ["num_arms", "num_stages", "num_rounds", "num_mcrs"]
    plt.suptitle(
        {
            _plt_lbl_list[i]: e
            for i, e in enumerate(get_params_from_filename(filename))
            if i in [0, 2, 3]
        }
    )
    # plt.legend()
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid()
    if save_as is not None:
        plt.savefig(save_as)
    else:
        plt.show()


def write_csv_file_multiple(x_data, func_data, filename, header, fmt):
    data = np.column_stack([x_data] + func_data)
    np.savetxt(filename, data, delimiter="\t", header=header, fmt=fmt, comments="")


def write_graphdata_to_csv(filelist: List[str], save_as: str = None):
    labels = []
    y_data = []
    y_data_len = []

    for filename in filelist:
        # Check if the file exists
        if os.path.isfile(filename):
            (
                num_arms,
                list_size,
                num_rounds,
                num_simulations,
                graph_type,
            ) = get_params_from_filename(filename)
            data = np.load(filename, allow_pickle=True)
            avg_cumulative_regret = data["avg_cumulative_regret"]
            y_data.append(avg_cumulative_regret)
            y_data_len.append(len(avg_cumulative_regret))
            label_tmp = f"CUTS-{graph_type}-{list_size} stages"
            labels.append(map_plot_labels(label_tmp))
        else:
            print(f"File {filename} not found. Skipping this file.")

    # Write to csv
    write_csv_file_multiple(
        x_data=range(max(y_data_len)),
        func_data=y_data,
        filename=save_as,
        header="round\t" + "\t".join(labels),
        fmt=["%d"] + ["%f"] * len(labels),
    )


if __name__ == "__main__":
    # Fixed parameters
    num_rounds = 2000
    num_simulations = 10000
    num_arms = 232
    plot_percentile = True
    plot_reward = False

    # Graphs labels (in file names) to plot
    graph_types = [
        "p0.02",
        "full",
        "line",
    ]
    list_sizes = [1, 4]

    # Generate the simdata filenames to plot
    regret_base_file = "simdata/CUTS_rounds{num_rounds}_mcr{num_simulations}_arms{num_arms}_list{list_size}_{graph_type}.npz"
    filelist = []

    # DEBUG
    for graph_type, list_size in product(graph_types, list_sizes):
        filename = regret_base_file.format(
            num_rounds=num_rounds,
            num_simulations=num_simulations,
            num_arms=num_arms,
            list_size=list_size,
            graph_type=graph_type,
        )
        filelist.append(filename)

    # Plot the data, save the plot and plot data
    plot_from_multiple_simdatafiles(
        filelist=filelist, plot_percentile=plot_percentile, plot_reward=plot_reward
    )

    # Write the data to a csv file
    write_graphdata_to_csv(
        filelist,
        save_as=f"simdata/CUTS_rounds{num_rounds}_mcrs{num_simulations}_arms{num_arms}.csv",
    )
