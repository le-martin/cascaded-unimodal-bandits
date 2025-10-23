import marimo

__generated_with = "0.17.0"
app = marimo.App(width="columns")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # WiFi-CUTS: Rate Adaptation with Cascaded Unimodal Multi-Armed Bandits in IEEE 802.11ac Testbed Experiments

    _Authors:_ Martin Le (Institute for Communications Technology, TU Braunschweig), Bile Peng (Institute for Communications Technology, TU Braunschweig), Eduard A. Jorswieck (Institute for Communications Technology, TU Braunschweig)

    This [marimo](https://marimo.io) notebook presents the performance of the Cascaded Unimodal Thompson Sampling (CUTS) developed to exploit the inherent unimodal and cascaded properties of rate adaptation in Wi-Fi. The notebook makes it easier to run simple simulations and visualize the simulation results in an interactive way.

    This notebook consists of two parts:
    - **CUTS Numerical Simulation** (for running fast and light-weight simulations)
    - **Interactive Plotting GUI** (for interactive plotting of numerical simulations).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # CUTS Numerical Simulation

    This is the first part of the marimo notebook that allows fast and guided numerical simulations of CUTS.

    - In the following, you can set the bandit and graph parameters for a numerical simulation of CUTS.
    - After setting the parameters, please press the 'click to run' button, to start the simulation.
        - If the simulation was already done with the given parameter set, then the simulation results are loaded from the simulation files under the folder `simdata`. To force a rerun of the simulation and overwrite the old simulation results, please tick the rerun-checkbox.
    - When the simulation is done, the corresponding cumulative regret plot is shown.

    *Note: Only one line corresponding to the given parameter set can be shown here. To plot multiple lines, please refer to the second part of this marimo notebook.*
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    num_arms_number = mo.ui.number(2, 200, value=100, label=" Number of arms (min: 2, max: 200): ")
    num_rounds_number = mo.ui.number(10, 5000, value=2000, label="Number of rounds (min: 10, max: 5000): ")
    num_monte_carlo_runs_number = mo.ui.number(1, 100, value=10, label="Number of Monte Carlo runs (min: 1, max: 100): ")
    num_stages_slider = mo.ui.slider(1, 4, label="Number of stages in cascaded model (min: 1, max: 4): ")
    edge_prob_number = mo.ui.number(0.1, 1, 0.1, 0.2, label="Edge probability in random graph (min: 0.1, max: 1): ")
    graph_type_dropdown = mo.ui.dropdown(
        options = ["erdos-renyi", "line", "full"], 
        value = "erdos-renyi", 
        label = "Graph type (erdos-renyi/random, line, full/complete): "
    )
    run_simulation_button = mo.ui.run_button()
    rerun_checkbox = mo.ui.checkbox(label="Rerun simulation and overwrite old data")
    return (
        edge_prob_number,
        graph_type_dropdown,
        num_arms_number,
        num_monte_carlo_runs_number,
        num_rounds_number,
        num_stages_slider,
        rerun_checkbox,
        run_simulation_button,
    )


@app.cell(hide_code=True)
def _(
    edge_prob_number,
    graph_type_dropdown,
    mo,
    num_arms_number,
    num_monte_carlo_runs_number,
    num_rounds_number,
    num_stages_slider,
    rerun_checkbox,
    run_simulation_button,
):
    num_arms = num_arms_number.value
    num_rounds = num_rounds_number.value
    num_simulations = num_monte_carlo_runs_number.value
    list_size = num_stages_slider.value

    graph_type = graph_type_dropdown.value
    p = edge_prob_number.value

    rerun_sim = rerun_checkbox.value

    mo.md(
        f"""## Input Parameters for Numerical Simulation of CUTS:\n
        ### Bandit Parameters:\n
        {num_arms_number}\n
        {num_rounds_number}\n
        {num_monte_carlo_runs_number}\n
        {num_stages_slider} {list_size} stages\n
        ### Graph Parameters\n
        {graph_type_dropdown}\n
        {edge_prob_number}\n

        ### To **(re)start** the **simulation** with the above parameters, please **press** the following **button**:
        {run_simulation_button} {rerun_checkbox}
        """
    )
    return (
        graph_type,
        list_size,
        num_arms,
        num_rounds,
        num_simulations,
        p,
        rerun_sim,
    )


@app.cell(hide_code=True)
def _(
    CUTS,
    CascadingUnimodalBandit,
    assign_success_probabilities,
    create_or_load_graph,
    generate_filename,
    graph_type,
    list_size,
    load_and_plot_individual_simulation_data,
    mo,
    np,
    num_arms,
    num_rounds,
    num_simulations,
    os,
    p,
    rerun_sim,
    run_and_save_simulation_data,
    run_simulation_button,
):
    mo.stop(not run_simulation_button.value, "Click 'run' above to start the simulation with the above defined parameters.")

    # import asyncio
    # for _ in mo. status.progress_bar(
    #     range(num_stages), 
    #     title="Running simulation...", 
    #     subtitle="Please wait", 
    #     show_eta=True, 
    #     show_rate=True
    # ):
    #     await asyncio.sleep(0.5)
    #     # Run the CUTS simulation

    # Plot results in the following code
    mo.md(
        f"""# Output of simulation parameters:
        {num_arms} arms\n
        {num_rounds} rounds\n
        {num_simulations} Monte Carlo runs\n
        {list_size} stages\n
        p = {p}\n 
        """
    )
    # Set random seed for reproducibility
    rng = np.random.default_rng(2024)

    # List of algorithms to compare
    algorithms = [CUTS]
    algorithm_names = ["CUTS"]

    # Create or load the graph
    if graph_type == "erdos-renyi":
        suffix = f"p{p}"
    else:
        suffix = graph_type
    graph_filename = f'simdata/CUTS_rounds{num_rounds}_mcr{num_simulations}_arms{num_arms}_{suffix}.pickle'
    graph_labels = None

    G = create_or_load_graph(graph_filename, num_arms, graph_type, p, labels=graph_labels)

    # Assign success probabilities based on distances
    means = assign_success_probabilities(G)

    # Generate the base filename (without algorithm-specific suffix)
    base_filename = generate_filename("{name}", num_arms, list_size, num_rounds, num_simulations, suffix)

    # Remove the file extension to get the base filename
    base_filename = os.path.splitext(base_filename)[0]

    # Check if individual algorithm data files exist
    all_files_exist = True
    for name in algorithm_names:
        alg_filename = base_filename.format(name=name) + ".npz"
        if not os.path.isfile(alg_filename):
            all_files_exist = False
            break

    # Check if `simdata` folder exists
    if not os.path.exists("simdata"):
        os.makedirs("simdata")
        print("Folder created: simdata")

    if not all_files_exist or rerun_sim:
        # Run simulation and save data
        run_and_save_simulation_data(base_filename + ".npz", CascadingUnimodalBandit, algorithms, algorithm_names, num_arms, list_size, num_rounds, num_simulations, means, G, save=True, rng=rng)
    else:
        print("All individual simulation data files exist. Skipping simulation.")

    # Load and plot data from individual files
    save_as = graph_filename.replace(".pickle", f"_list{list_size}_regret.png")
    save_as = None
    mo.md(f"{mo.as_html(load_and_plot_individual_simulation_data(base_filename, algorithm_names, save_as=save_as))}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    # Interactive Plotting GUI

    This is the second part of the marimo notebook that allows plotting of different parameter sets and shows different lines.

    - As in the first part, to plot the regret/reward lines, there are bandit and graph parameters to choose from.
    - Upon selecting the parameters, the plot will be updated on-the-fly if simulation results are available.
        - If they are unavailable, a warning will be displayed for the missing configurations. Please simulate with the desired parameters and then try plotting them again.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    # Bandit parameters UI elements
    num_arms_plot_number = mo.ui.number(2, 1000, value=232, label="Number of arms (min: 2, max: 1000): ")
    num_rounds_plot_number = mo.ui.number(10, 5000, value=2000, label="Number of rounds (min: 10, max: 5000): ")
    num_monte_carlo_runs_plot_number = mo.ui.number(1, 10000, value=10000, label="Number of Monte Carlo runs (min: 1, max: 10000): ")
    stage_1_checkbox = mo.ui.checkbox(value=True, label="1")
    stage_2_checkbox = mo.ui.checkbox(label="2")
    stage_3_checkbox = mo.ui.checkbox(label="3")
    stage_4_checkbox = mo.ui.checkbox(label="4")
    graph_type_erdos_renyi_checkbox = mo.ui.checkbox(value=True, label="Erdos Renyi/Random")
    graph_type_line_checkbox = mo.ui.checkbox(value=True, label="Line")
    graph_type_full_checkbox = mo.ui.checkbox(value=True, label="Full/Complete")

    # Graph parameters UI elements
    edge_prob_plot_number = mo.ui.number(0.01, 1, 0.01, 0.02, label="Edge probability in Erdos-Renyi/random graph (min: 0.01, max: 1): ")
    plot_percentile_dropdown = mo.ui.dropdown(options=[True, False], value=False, label="Plot 95% confidence interval?")
    plot_reward_dropdown = mo.ui.dropdown(options=[True, False], value=False, label="Plot reward instead of regret?")
    return (
        edge_prob_plot_number,
        graph_type_erdos_renyi_checkbox,
        graph_type_full_checkbox,
        graph_type_line_checkbox,
        num_arms_plot_number,
        num_monte_carlo_runs_plot_number,
        num_rounds_plot_number,
        plot_percentile_dropdown,
        plot_reward_dropdown,
        stage_1_checkbox,
        stage_2_checkbox,
        stage_3_checkbox,
        stage_4_checkbox,
    )


@app.cell(hide_code=True)
def _(
    edge_prob_plot_number,
    graph_type_erdos_renyi_checkbox,
    graph_type_full_checkbox,
    graph_type_line_checkbox,
    mo,
    num_arms_plot_number,
    num_monte_carlo_runs_plot_number,
    num_rounds_plot_number,
    plot_percentile_dropdown,
    plot_reward_dropdown,
    stage_1_checkbox,
    stage_2_checkbox,
    stage_3_checkbox,
    stage_4_checkbox,
):
    mo.md(
        f"""
    ## Plot multiple lines
    ### Bandit Parameters:\n
    {num_arms_plot_number}\n
    {num_rounds_plot_number}\n
    {num_monte_carlo_runs_plot_number}\n
    Number of stages to plot:
    {stage_1_checkbox}{stage_2_checkbox}{stage_3_checkbox}{stage_4_checkbox}\n


    ### Graph Parameters
    Graph types: {graph_type_erdos_renyi_checkbox} {graph_type_line_checkbox} {graph_type_full_checkbox} \n
    {edge_prob_plot_number}\n
    {plot_percentile_dropdown} {plot_reward_dropdown} \n
    """
    )
    return


@app.cell(hide_code=True)
def _(filelist, get_params_from_filename, mo, os):
    missing_sims = [_file for _file in filelist if not os.path.isfile(_file)]
    attention_str = "/// attention | Attention!\n The following CUTS simulations are missing: \n\n"
    for _file in missing_sims:
        arms, stages, rounds, sims, suff = get_params_from_filename(_file)
        attention_str += f"Arms: {arms}, Stages: {stages}, Rounds: {rounds}, Monte Carlo runs: {sims}, Graph information: {suff}\n\n"
    attention_str += "///"
    if not missing_sims:
        attention_str = "/// details | All lines were successfully plotted.\n"
        for _file in filelist:
            arms, stages, rounds, sims, suff = get_params_from_filename(_file)
            attention_str += f"Arms: {arms}, Stages: {stages}, Rounds: {rounds}, Monte Carlo runs: {sims}, Graph information: {suff}\n\n"
        attention_str += "///"
    mo.md(attention_str)
    return


@app.cell(hide_code=True)
def _(
    edge_prob_plot_number,
    graph_type_erdos_renyi_checkbox,
    graph_type_full_checkbox,
    graph_type_line_checkbox,
    mo,
    num_arms_plot_number,
    num_monte_carlo_runs_plot_number,
    num_rounds_plot_number,
    plot_from_multiple_simdatafiles,
    plot_percentile_dropdown,
    plot_reward_dropdown,
    product,
    stage_1_checkbox,
    stage_2_checkbox,
    stage_3_checkbox,
    stage_4_checkbox,
):
    # Bandit parameters
    num_arms_plot = num_arms_plot_number.value
    num_rounds_plot = num_rounds_plot_number.value
    num_simulations_plot = num_monte_carlo_runs_plot_number.value

    stage_1 = stage_1_checkbox.value
    stage_2 = stage_2_checkbox.value
    stage_3 = stage_3_checkbox.value
    stage_4 = stage_4_checkbox.value

    # Graph parameters
    graph_type_erdos_renyi = graph_type_erdos_renyi_checkbox.value
    graph_type_line = graph_type_line_checkbox.value
    graph_type_full = graph_type_full_checkbox.value

    edge_prob_plot = edge_prob_plot_number.value

    plot_percentile = plot_percentile_dropdown.value
    plot_reward = plot_reward_dropdown.value

    graph_types = []
    if graph_type_erdos_renyi:
        graph_types.append(f"p{edge_prob_plot}")
    if graph_type_line:
        graph_types.append("line")
    if graph_type_full:
        graph_types.append("full")

    list_sizes = []
    if stage_1:
        list_sizes.append(1)
    if stage_2:
        list_sizes.append(2)
    if stage_3:
        list_sizes.append(3)
    if stage_4:
        list_sizes.append(4)

    # Generate the simdata filenames to plot
    regret_base_file = str(mo.notebook_dir() / "simdata" / "CUTS_rounds{num_rounds}_mcr{num_simulations}_arms{num_arms}_list{list_size}_{graph_type}.npz")
    filelist = []

    # DEBUG
    for graph_type_tmp, list_size_tmp in product(graph_types, list_sizes):
        filename = regret_base_file.format(
            num_rounds=num_rounds_plot,
            num_simulations=num_simulations_plot,
            num_arms=num_arms_plot,
            list_size=list_size_tmp,
            graph_type=graph_type_tmp,
        )
        filelist.append(filename)

    mo.md(
        f"{mo.as_html(plot_from_multiple_simdatafiles(filelist=filelist, plot_percentile=plot_percentile, plot_reward=plot_reward))}"
    )
    return (filelist,)


@app.cell(hide_code=True)
def _(best_4_pairs, write_csv_file_multiple):
    from functools import cache
    import heapq
    from itertools import product, cycle
    import os
    import pickle
    import random
    import time
    import tqdm
    from typing import List

    import marimo as mo
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np


    @mo.cache
    def load_and_plot_individual_simulation_data(base_filename, algorithm_names, save_as=None):
        plt.figure(figsize=(12, 8))
        for name in algorithm_names:
            alg_filename = base_filename.format(name=name) + ".npz"
            # Check if the file exists
            if os.path.isfile(mo.notebook_dir() / alg_filename):
                # Load data
                data = np.load(alg_filename, allow_pickle=True)
                avg_cumulative_regret = data['avg_cumulative_regret']
                percentile_2_5 = data['percentile_2_5']
                percentile_97_5 = data['percentile_97_5']

                # Plot the mean cumulative regret with a shaded area for the 95% confidence interval
                plt.plot(avg_cumulative_regret, label=f"{name} Mean Cumulative Regret")
                plt.fill_between(range(len(avg_cumulative_regret)), percentile_2_5, percentile_97_5, alpha=0.2)
            else:
                print(f"File {alg_filename} not found. Skipping this algorithm.")

        plt.xlabel("Rounds")
        plt.ylabel("Cumulative Regret")
        plt.title("Cumulative regret performance of CUTS")
        plt.suptitle(get_params_from_filename(alg_filename))
        plt.legend()
        if save_as is not None:
            plt.savefig(save_as)
        return plt.gca()


    @mo.cache
    def plot_from_multiple_simdatafiles(filelist, save_as=None, plot_percentile=True, plot_reward=False):
        plt.figure(figsize=(12, 8), dpi=300)
        ax = plt.gca()
        linestyles = cycle(['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (5, (10, 3)), (0, (5, 10)), (0, (3, 10, 1, 10))])
        for filename in filelist:
            # Check if the file exists
            if os.path.isfile(mo.notebook_dir() / filename):
                num_arms, list_size, num_rounds, num_simulations, graph_type = get_params_from_filename(filename)
                # Load data
                data = np.load(filename, allow_pickle=True)
                avg_cumulative_regret = data['avg_cumulative_regret']
                percentile_2_5 = data['percentile_2_5']
                percentile_97_5 = data['percentile_97_5']
                avg_reward = data['avg_reward']
                percentile_2_5_reward = data['percentile_2_5_reward']
                percentile_97_5_reward = data['percentile_97_5_reward']

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
                        plt.fill_between(range(len(avg_reward)), percentile_2_5_reward, percentile_97_5_reward, alpha=0.2)
                    else:
                        plt.fill_between(range(len(avg_cumulative_regret)), percentile_2_5, percentile_97_5, alpha=0.2)
            else:
                print(f"File {filename} not found. Skipping this file.")

        plt.xlabel("Rounds")
        if plot_reward:
            plt.ylabel("Average Reward")
        else:
            plt.ylabel("Average Cumulative Regret")
        plt.title("Cumulative regret performance of CUTS")
        _plt_lbl_list = ['num_arms', 'num_stages', 'num_rounds', 'num_mcrs']
        plt.suptitle({_plt_lbl_list[i]: e for i, e in enumerate(get_params_from_filename(filename)) if i in [0, 2, 3]})
        # plt.legend()
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid()
        if save_as is not None:
            plt.savefig(save_as)
        return plt.gca()


    @mo.cache
    def print_missing_sim_warning(missing_sims):
        attention_str = "/// attention | Attention!\n The following CUTS simulations are missing: \n\n"
        for _filename in missing_sims:
            arms, stages, rounds, sims, suff = get_params_from_filename(_filename)
            attention_str += f"Arms: {arms}, Stages: {stages}, Rounds: {rounds}, Monte Carlo runs: {sims}, Graph information: {suff}\n\n"
        attention_str += "///"
        if missing_sims:
            mo.md(attention_str)


    class CascadingUnimodalBandit:
        def __init__(self, G, num_items, list_size, means=None, rng=None):
            self.G = G
            self.num_items = num_items
            self.list_size = list_size
            self.rate_to_node = {rate: node for node, rate in enumerate(G.nodes)}
            if means is None:
                self.means = np.array(
                    rng.random(num_items)
                )  # True but unknown click probabilities
                self.given_means = False
            elif isinstance(means, list):
                self.given_means = True
                self.means = np.array(means)
            elif isinstance(means, dict):
                self.given_means = True
                self.means = np.zeros(G.number_of_nodes())
                for node, mean in means.items():
                    self.means[self.rate_to_node[node]] = mean


    class CUTS:
        def __init__(self, G, num_items, list_size, reward_scaling=None):
            self.G = G
            self.num_items = num_items
            self.list_size = list_size
            # Initialize Beta parameters for each item
            self.alpha = np.ones(
                num_items
            )  # Alpha parameter of Beta distribution (success count)
            self.beta = np.ones(
                num_items
            )  # Beta parameter of Beta distribution (failure count)
            self.leader_counts = {node: 0 for node in G.nodes}
            self.counts = np.zeros(num_items)
            self.empirical_means = np.zeros(num_items)
            self.reward_scaling = reward_scaling

        def thompson_sample(self, node):
            """Draw a sample from the Beta distribution for a given node."""
            return np.random.beta(self.alpha[node], self.beta[node])

        @cache
        def get_extended_neighbors(self, start, depth):
            visited = {start}
            frontier = {start}

            for _ in range(depth):
                next_frontier = set()
                for node in frontier:
                    for neighbor in self.G.neighbors(node):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            next_frontier.add(neighbor)
                # If there are no more nodes to explore, break early.
                if not next_frontier:
                    break
                frontier = next_frontier
            return list(visited)

        def select_arms(self, *args, **kwargs):
            single_hop_exploration = kwargs.get("single_hop_exploration", False)
            selected_arms = []
            samples_values = {node: self.thompson_sample(node) for node in self.G.nodes}
            # Select the top K items with the highest sampled values from the leader's neighborhood
            if self.reward_scaling is not None:
                samples_values = {
                    node: samples_values[node] * self.reward_scaling[node]
                    for node in self.G.nodes
                }
            else:
                samples_values = {node: samples_values[node] for node in self.G.nodes}
            if self.counts.sum() == 0:
                # If no arms have been selected yet, select the leader based on sampled values
                leader = select_n_largest(samples_values, 1)[0]
            else:
                # Select the leader based on empirical means
                leader = select_n_largest(self.empirical_means, 1)[0]
            if single_hop_exploration:
                neighborhood = list(self.G.neighbors(leader)) + [leader]
            else:
                neighborhood = self.get_extended_neighbors(leader, self.list_size)
            samples_values_neigborhood = {
                node: samples_values[node] for node in neighborhood
            }
            # Select the top K items with the highest sampled values
            # selected_arms = neighborhood[np.argsort(theta_samples_filtered)[-self.list_size:]]
            selected_arms = select_n_largest(samples_values_neigborhood, self.list_size)
            return selected_arms

        def update(self, selected_items, feedback, reward_scaling=None):
            # Update Beta parameters based on observed feedback
            for idx, item in enumerate(selected_items):
                self.counts[item] += 1
                if reward_scaling is not None:
                    reward = (
                        reward_scaling[selected_items[feedback]] if feedback != -1 else 0
                    )
                else:
                    reward = 1 if feedback != -1 else 0
                if feedback == idx:  # User clicked on this item
                    self.alpha[item] += 1
                    self.empirical_means[item] = (
                        self.empirical_means[item] * (self.counts[item] - 1) + reward
                    ) / self.counts[item]
                    break
                else:  # User skipped this item
                    self.beta[item] += 1
                    # self.empirical_means = self.alpha / (self.alpha + self.beta)
                    self.empirical_means[item] = (
                        self.empirical_means[item] * (self.counts[item] - 1)
                    ) / self.counts[item]

        def reset(self):
            """Reset the algorithm parameters for a fresh run."""
            self.alpha = np.ones(self.num_items)
            self.beta = np.ones(self.num_items)
            self.empirical_means = {node: 0 for node in self.G.nodes}
            self.counts = np.zeros(self.num_items)


    def create_or_load_graph(filename, num_nodes, graph_type='erdos-renyi', p=0.5, labels=None, new_graph=False):
        """
        Create a graph or load from a file if it already exists.

        Parameters:
        - filename (str): The file name to save or load the graph.
        - num_nodes (int): The number of nodes in the graph.
        - graph_type (str): The type of graph to create ('line' or 'erdos-renyi').
        - p (float): Probability for edge creation in Erdős-Rényi graph.
        - new_graph (bool): If True, create a new graph even if the file exists.

        Returns:
        - G (networkx.Graph): The generated or loaded graph.
        """
        if os.path.exists(filename) and not new_graph:
            print(f"Loading graph from {filename}")
            with open(filename, 'rb') as f:
                G = pickle.load(f)
        else:
            print(f"Creating a new {graph_type} graph with {num_nodes} nodes")
            if (graph_type == 'line') or (graph_type == 'line-csv'):
                G = nx.path_graph(num_nodes)
            elif graph_type == 'erdos-renyi':
                G = nx.erdos_renyi_graph(num_nodes, p)
                while not nx.is_connected(G):
                    G = nx.erdos_renyi_graph(num_nodes, p)
            elif graph_type == 'full':
                G = nx.complete_graph(num_nodes)
            else:
                raise ValueError("Unsupported graph type. Use 'line' or 'erdos-renyi'.")

            # Check if folder does not exist
            folder_path = os.path.dirname(filename)
            if not os.path.exists(folder_path):
                # Create the folder (including any necessary parent directories)
                os.makedirs(folder_path)
                print(f"Folder created: {folder_path}")

            # Save the graph to a file
            with open(filename, 'wb') as f:
                pickle.dump(G, f)
            print(f"Graph saved to {filename}")

        return G


    def assign_success_probabilities(G, max_prob=0.9, min_prob=0.1):
        """
        Assign success probabilities to each node in the graph based on its distance
        from an optimal arm.

        Parameters:
        - G (networkx.Graph): The graph representing the MAB structure.
        - max_prob (float): Maximum success probability for the optimal arm.
        - min_prob (float): Minimum success probability for the farthest arm.

        Returns:
        - success_probabilities (dict): A dictionary with nodes as keys and success probabilities as values.
        """
        # Randomly select the optimal arm
        optimal_arm = random.choice(list(G.nodes))
        nx.set_node_attributes(G, {optimal_arm: {"optimal": True}})
        print(f"Optimal Arm: {optimal_arm}")

        # Calculate shortest path distances from the optimal arm
        shortest_paths = nx.single_source_shortest_path_length(G, optimal_arm)
        max_distance = max(shortest_paths.values())

        # Calculate success probabilities
        success_probabilities = {}
        for node, distance in shortest_paths.items():
            # Linear decay from max_prob to min_prob based on distance
            success_prob = max_prob - (distance / max_distance) * (max_prob - min_prob)
            success_probabilities[node] = success_prob
            G.nodes[node]["success_prob"] = (
                success_prob  # Add as a node attribute for easy access
            )

        return success_probabilities


    def generate_filename(
        prefix, num_items, list_size, num_rounds, num_simulations, suffix
    ):
        return f"simdata/{prefix}_rounds{num_rounds}_mcr{num_simulations}_arms{num_items}_list{list_size}_{suffix}.npz"


    def run_and_save_simulation_data(
        filename,
        bandit_class,
        algorithms,
        algorithm_names,
        num_items,
        list_size,
        num_rounds=1000,
        num_simulations=100,
        means=None,
        G=None,
        reward_scaling=None,
        single_hop_exploration=False,
        save=True,
        rng=None,
    ):
        data = {}
        # Record start time
        start_time = time.time()

        # Remove file extension to get base filename
        base_filename = os.path.splitext(filename)[0]

        for algorithm_class, name in zip(algorithms, algorithm_names):
            # Run the Monte Carlo simulation for each algorithm
            (
                avg_cumulative_regret,
                percentile_2_5,
                percentile_97_5,
                means,
                avg_reward,
                percentile_2_5_reward,
                percentile_97_5_reward,
            ) = monte_carlo_simulation(
                G,
                bandit_class,
                algorithm_class,
                num_items,
                list_size,
                num_rounds,
                num_simulations,
                means=means,
                reward_scaling=reward_scaling,
                single_hop_exploration=single_hop_exploration,
                rng=rng,
            )

            # Store data in a dictionary
            data[name] = {
                "means": means,
                "avg_cumulative_regret": avg_cumulative_regret,
                "percentile_2_5": percentile_2_5,
                "percentile_97_5": percentile_97_5,
                "avg_reward": avg_reward,
                "percentile_2_5_reward": percentile_2_5_reward,
                "percentile_97_5_reward": percentile_97_5_reward,
            }

            # Save individual algorithm data to a separate file
            alg_filename = base_filename.format(name=name) + ".npz"
            if save:
                np.savez(
                    alg_filename,
                    avg_cumulative_regret=avg_cumulative_regret,
                    percentile_2_5=percentile_2_5,
                    percentile_97_5=percentile_97_5,
                    means=means,
                    avg_reward=avg_reward,
                    percentile_2_5_reward=percentile_2_5_reward,
                    percentile_97_5_reward=percentile_97_5_reward,
                )
            print(f"Simulation data for {name} saved to {alg_filename}")

        # Record end time
        end_time = time.time()
        print(f"Simulation completed in {end_time - start_time:.2f} seconds.")


    def get_params_from_filename(filename):
        filename = os.path.splitext(filename)[0]
        parts = filename.split("_")
        num_arms = int(parts[3][4:])
        list_size = int(parts[4][4:])
        num_rounds = int(parts[1][6:])
        num_simulations = int(parts[2][3:])
        suffix = parts[-1]
        return num_arms, list_size, num_rounds, num_simulations, suffix


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


    def monte_carlo_simulation(
        G,
        bandit_class,
        algorithm_class,
        num_items,
        list_size,
        num_rounds=1000,
        num_simulations=100,
        means=None,
        reward_scaling=None,
        single_hop_exploration=False,
        rng=None,
    ):
        all_regrets = []
        all_rewards = []
        # Initialize a new bandit and algorithm instance for each simulation run
        bandit = bandit_class(
            G=G, num_items=num_items, list_size=list_size, means=means, rng=rng
        )
        algorithm = algorithm_class(G, num_items, list_size, reward_scaling=reward_scaling)
        if reward_scaling is None:
            best_arm_ind = np.argsort(bandit.means)[-list_size:]
            optimal_reward = expected_instant_reward(
                means=bandit.means, selected_items=best_arm_ind
            )
        else:
            optimal_reward, best_arm_ind = best_4_pairs(bandit.means, reward_scaling)

        for sim in tqdm.tqdm(range(num_simulations), desc="Monte Carlo Runs"):
            algorithm.reset()
            # Run the simulation for the specified number of rounds and record cumulative regret
            regrets = []
            rewards = []
            arm_history = []
            for t in range(1, num_rounds + 1):
                selected_items = algorithm.select_arms(
                    t, single_hop_exploration=single_hop_exploration
                )
                feedback = simulate_click(bandit.means, selected_items, rng=rng)
                arm_history.append(selected_items[feedback] if feedback != -1 else -1)
                algorithm.update(selected_items, feedback, reward_scaling)

                # Calculate cumulative regret for this round
                if reward_scaling is None:
                    observed_reward = 1 if feedback != -1 else 0
                else:
                    observed_reward = reward_scaling[selected_items[feedback]] if feedback != -1 else 0
                # regret = optimal_reward - observed_reward
                regret = optimal_reward - expected_instant_reward(bandit.means, selected_items)
                regrets.append(regret)
                rewards.append(observed_reward)

            # Store cumulative regret for this simulation
            cumulative_regret = np.cumsum(regrets)
            all_regrets.append(cumulative_regret)
            all_rewards.append(np.array(rewards))

        # Convert all regrets and observed rewards to a NumPy array for percentile calculation
        all_regrets = np.array(all_regrets)
        all_rewards = np.array(all_rewards)

        # Compute the mean cumulative regret
        mean_regret = np.mean(all_regrets, axis=0)
        # Compute the 2.5th and 97.5th percentiles for 95% confidence interval
        lower_percentile = np.percentile(all_regrets, 2.5, axis=0)
        upper_percentile = np.percentile(all_regrets, 97.5, axis=0)

        # Compute the mean observed rewards
        mean_rewards = np.mean(all_rewards, axis=0)
        # Compute the 2.5th and 97.5th percentiles for observed rewards
        lower_percentile_rewards = np.percentile(all_rewards, 2.5, axis=0)
        upper_percentile_rewards = np.percentile(all_rewards, 97.5, axis=0)

        return (
            mean_regret,
            lower_percentile,
            upper_percentile,
            bandit.means,
            mean_rewards,
            lower_percentile_rewards,
            upper_percentile_rewards,
        )


    def expected_instant_reward(means: np.ndarray, selected_items: np.ndarray):
        return 1 - np.prod(1 - means[selected_items])


    def select_n_largest(data, n):
        """
        Find the indices (for lists) or keys (for dictionaries) of the largest `n` elements,
        resolving ties randomly.

        Parameters
        ----------
        data : list or dict
            The input data structure. Can be a list of values or a dictionary where values are compared.
        n : int
            The number of largest elements to find.

        Returns
        -------
        list
            A list of indices (for lists) or keys (for dictionaries) corresponding to the largest `n` elements.
            Ties are resolved randomly.
        """
        if not isinstance(data, (list, dict)):
            raise TypeError("Input must be a list or a dictionary.")
        if n < 0:
            raise ValueError("`n` must be non-negative.")
        if n == 0:
            return []

        # Instead of creating a full sorted list, we “decorate” each item with a random value
        # to randomize tie resolution, then use heapq.nlargest to select the top n.
        if isinstance(data, list):
            # Each element becomes a tuple: (value, random_tiebreaker, index)
            items = ((value, random.random(), index) for index, value in enumerate(data))
        else:
            # For dictionaries, each element is: (value, random_tiebreaker, key)
            items = ((value, random.random(), key) for key, value in data.items())

        # heapq.nlargest returns the n largest items (ordered in descending order)
        # using the key (value, random_tiebreaker) to decide the ordering.
        top_n = heapq.nlargest(n, items, key=lambda item: (item[0], item[1]))

        # Extract and return the key (or index) from each tuple.
        return [item[2] for item in top_n]


    def simulate_click(attraction_probs, selected_items, rng=None):
        for i, item in enumerate(selected_items):
            if rng.binomial(1, attraction_probs[item]):
                return i  # Clicked on item i
        return -1  # No clicks
    return (
        CUTS,
        CascadingUnimodalBandit,
        assign_success_probabilities,
        create_or_load_graph,
        generate_filename,
        get_params_from_filename,
        load_and_plot_individual_simulation_data,
        mo,
        np,
        os,
        plot_from_multiple_simdatafiles,
        product,
        run_and_save_simulation_data,
    )


if __name__ == "__main__":
    app.run()
