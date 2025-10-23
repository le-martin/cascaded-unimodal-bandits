import marimo

__generated_with = "0.17.0"
app = marimo.App(width="columns")


@app.cell
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


@app.cell
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
def _():
    from itertools import product, cycle
    import os

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from cuts import (
        CascadingUnimodalBandit,
        CUTS,
        create_or_load_graph,
        assign_success_probabilities,
        generate_filename,
        run_and_save_simulation_data,
        get_params_from_filename
    )
    from plot_multi import (
        write_graphdata_to_csv,
        map_plot_labels
    )

    @mo.cache
    def load_and_plot_individual_simulation_data(base_filename, algorithm_names, save_as=None):
        plt.figure(figsize=(12, 8))
        for name in algorithm_names:
            alg_filename = base_filename.format(name=name) + ".npz"
            # Check if the file exists
            if os.path.isfile(alg_filename):
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
            if os.path.isfile(filename):
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
        for filename_tmp3 in missing_sims:
            arms, stages, rounds, sims, suff = get_params_from_filename(filename_tmp3)
            attention_str += f"Arms: {arms}, Stages: {stages}, Rounds: {rounds}, Monte Carlo runs: {sims}, Graph information: {suff}\n\n"
        attention_str += "///"
        if missing_sims:
            mo.md(attention_str)
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


@app.cell
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
    num_arms_plot_number = mo.ui.number(2, 1000, value=232, label="Number of arms (min:2, max: 1000): ")
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
    regret_base_file = "simdata/CUTS_rounds{num_rounds}_mcr{num_simulations}_arms{num_arms}_list{list_size}_{graph_type}.npz"
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


if __name__ == "__main__":
    app.run()
