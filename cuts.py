import argparse
from functools import cache
import heapq
import math
import os.path
from pathlib import Path
import pickle
import random
import time
import tqdm

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


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


def simulate_click(attraction_probs, selected_items, rng=None):
    for i, item in enumerate(selected_items):
        if rng.binomial(1, attraction_probs[item]):
            return i  # Clicked on item i
    return -1  # No clicks


# Updated Monte Carlo Simulation Function to return all regrets
# Monte Carlo Simulation Function - same as before, but returns all results for saving
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


# Function to save simulation data to files
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


# Function to load data from a file and plot
def load_and_plot_simulation_data(filename):
    # Load data
    data = np.load(filename, allow_pickle=True)

    plt.figure(figsize=(12, 8))
    for name in data:
        avg_cumulative_regret = data[name].item().get("avg_cumulative_regret")
        percentile_2_5 = data[name].item().get("percentile_2_5")
        percentile_97_5 = data[name].item().get("percentile_97_5")

        # Plot the mean cumulative regret with a shaded area for the 95% confidence interval
        plt.plot(avg_cumulative_regret, label=f"{name} Mean Cumulative Regret")
        plt.fill_between(
            range(len(avg_cumulative_regret)),
            percentile_2_5,
            percentile_97_5,
            alpha=0.2,
            label=f"{name} 95% CI",
        )

    plt.xlabel("Rounds")
    plt.ylabel("Cumulative Regret")
    plt.title("Comparison of Algorithm Performance in Cascading Bandits")
    plt.legend()
    plt.show()


# Function to load data from individual files and plot
def load_and_plot_individual_simulation_data(
    base_filename, algorithm_names, save_as=None
):
    plt.figure(figsize=(12, 8))
    for name in algorithm_names:
        alg_filename = base_filename.format(name=name) + ".npz"
        # Check if the file exists
        if os.path.isfile(alg_filename):
            # Load data
            data = np.load(alg_filename, allow_pickle=True)
            avg_cumulative_regret = data["avg_cumulative_regret"]
            percentile_2_5 = data["percentile_2_5"]
            percentile_97_5 = data["percentile_97_5"]

            # Plot the mean cumulative regret with a shaded area for the 95% confidence interval
            plt.plot(avg_cumulative_regret, label=f"{name} Mean Cumulative Regret")
            plt.fill_between(
                range(len(avg_cumulative_regret)),
                percentile_2_5,
                percentile_97_5,
                alpha=0.2,
            )
        else:
            print(f"File {alg_filename} not found. Skipping this algorithm.")

    plt.xlabel("Rounds")
    plt.ylabel("Cumulative Regret")
    plt.title("Cumulative regret performance of CUTS")
    plt.suptitle(get_params_from_filename(alg_filename))
    plt.legend()
    if save_as is not None:
        plt.savefig(save_as)
    else:
        plt.show()


def expected_instant_reward(means: np.ndarray, selected_items: np.ndarray):
    return 1 - np.prod(1 - means[selected_items])


def generate_filename(
    prefix, num_items, list_size, num_rounds, num_simulations, suffix
):
    return f"simdata/{prefix}_rounds{num_rounds}_mcr{num_simulations}_arms{num_items}_list{list_size}_{suffix}.npz"


# @profile
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


def get_params_from_filename(filename):
    filename = os.path.splitext(filename)[0]
    parts = filename.split("_")
    num_arms = int(parts[3][4:])
    list_size = int(parts[4][4:])
    num_rounds = int(parts[1][6:])
    num_simulations = int(parts[2][3:])
    suffix = parts[-1]
    return num_arms, list_size, num_rounds, num_simulations, suffix


def best_4_pairs(p_list, r_list):
    """
    Given two lists p_list, r_list of length N,
    each p[i] in [0,1], each r[i] > 0,
    find i, j, k, l (distinct) maximizing:
       p[i]*r[i]
       + (1-p[i])*p[j]*r[j]
       + (1-p[i])*(1-p[j])*p[k]*r[k]
       + (1-p[i])*(1-p[j])*(1-p[k])*p[l]*r[l]
    Returns (best_value, (i, j, k, l)).
    """

    N = len(p_list)
    # Precompute T[n] = p[n]*r[n]
    T = [p_list[n] * r_list[n] for n in range(N)]

    # Sort indices by descending T-value
    # e.g. sorted_indices[0] is index of the largest T
    sorted_indices = sorted(range(N), key=lambda x: T[x], reverse=True)

    best_val = -math.inf
    best_quad = None

    for i in range(N):
        for j in range(N):
            if j == i:
                continue
            for k in range(N):
                if k == i or k == j:
                    continue

                # partial for the first three picks
                partial_val = T[i] + (1 - p_list[i]) * (T[j] + (1 - p_list[j]) * T[k])
                factor = (1 - p_list[i]) * (1 - p_list[j]) * (1 - p_list[k])

                # Among all possible l != i,j,k, pick the one with largest T[l].
                # Just scan sorted_indices to find the first index not in (i,j,k).
                for candidate in sorted_indices:
                    if candidate != i and candidate != j and candidate != k:
                        val = partial_val + factor * T[candidate]
                        if val > best_val:
                            best_val = val
                            best_quad = (i, j, k, candidate)
                        break  # Found the best l for this i,j,k

    return best_val, best_quad


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


# Function to create or load a graph
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


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Monte Carlo simulations for cascading unimodal Thompson sampling.")

    # Add arguments for the simulation
    parser.add_argument("--num_arms", type=int, default=232, help="Number of arms in the cascading bandit.")
    parser.add_argument("--list_size", type=int, default=1, help="Size of the recommendation list.")
    parser.add_argument("--num_rounds", type=int, default=2000, help="Number of rounds in each simulation.")
    parser.add_argument("--num_simulations", type=int, default=100, help="Number of Monte Carlo simulations.")
    parser.add_argument("--graph_type", type=str, default="full", help="Type of graph structure (line, erdos-renyi, full).")
    parser.add_argument("--p", type=float, default=0.02, help="Probability of edge creation in the Erdos-Renyi graph.")
    parser.add_argument("--wifimeans_csvfile", type=str, default=None, help="CSV file with WiFi means.")
    parser.add_argument("--wifi_scaling", action="store_true", help="Scale means with data rates.")
    parser.add_argument("--single_hop_exploration", action="store_true", help="Enable single-hop exploration.")
    parser.add_argument("--rerun_all", type=int, default=0, help="Force re-running of the simulation even if the simulation results exist.")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed for reproducibility.")

    # Parse the arguments
    args = parser.parse_args()

    # Extract the arguments
    num_arms = args.num_arms
    list_size = args.list_size
    num_rounds = args.num_rounds
    num_simulations = args.num_simulations
    graph_type = args.graph_type
    p = args.p
    wifimeans_csvfile = args.wifimeans_csvfile
    wifi_scaling = args.wifi_scaling
    single_hop_exploration = args.single_hop_exploration

    # Check parameters
    if wifimeans_csvfile is not None:
        mean_csvdata = pd.read_csv(wifimeans_csvfile)
    if wifimeans_csvfile is not None and wifi_scaling:
        # Check if the number of entries in the CSV file is equal to `num_arms`
        if mean_csvdata.shape[0] != num_arms:
            raise ValueError(
                f"The number of means in {wifimeans_csvfile} is {mean_csvdata.shape[0]}. It is not equal to `num_arms`={num_arms}."
            )
        # Check if there is a column named 'mean_scaled' in the CSV file
        if "mean_scaled_with_goodput" not in mean_csvdata.columns:
            raise ValueError(
                f'The CSV file {wifimeans_csvfile} does not have a column named "mean_scaled".'
            )
    if wifi_scaling:
        # Check if there is a column named 'data_rate' in the CSV file
        if "data_rate_Mbps" not in mean_csvdata.columns:
            raise ValueError(
                f'The CSV file {wifimeans_csvfile} does not have a column named "data_rate_Mbps".'
            )

    # Set random seed for reproducibility
    rng = np.random.default_rng(args.seed)

    # List of algorithms to compare
    algorithms = [CUTS]
    algorithm_names = ["CUTS"]

    # Create or load the graph
    if graph_type == "erdos-renyi":
        suffix = f"p{p}"
    else:
        suffix = graph_type
    if wifi_scaling:
        suffix = f"{suffix}-wifi"
    graph_filename = f"simdata/CUTS_rounds{num_rounds}_mcr{num_simulations}_arms{num_arms}_{suffix}.pickle"
    graph_labels = None
    if (wifimeans_csvfile is not None) and Path(wifimeans_csvfile).exists():# and graph_type == 'line-csv':
        graph_labels = pd.read_csv(wifimeans_csvfile)['rate']
        
    G = create_or_load_graph(graph_filename, num_arms, graph_type, p, labels=graph_labels)

    # Assign success probabilities based on distances
    data_rates = None
    if (wifimeans_csvfile is not None) and Path(wifimeans_csvfile).exists():
        # Load the means from the CSV file
        means_list = pd.read_csv(wifimeans_csvfile).sort_values(
            by="mean_scaled_with_data_rate"
        )
        means = dict()
        if not wifi_scaling:
            for i, (_, row) in enumerate(means_list.iterrows()):
                means[i] = row["mean_scaled_with_data_rate"]
        else:
            data_rates = np.zeros(G.number_of_nodes())
            for i, (_, row) in enumerate(means_list.iterrows()):
                data_rates[i] = row["data_rate_Mbps"]
                means[i] = row["successes_rel"]
    else:
        if wifi_scaling:
            data_rates = np.random.uniform(1, 100, G.number_of_nodes())
            means = np.random.uniform(0, 1, G.number_of_nodes())
            means_scaled = means * data_rates
            sorted_indices = np.argsort(means_scaled)
            means_scaled = means_scaled[sorted_indices]
            data_rates = data_rates[sorted_indices]
            means_list = means[sorted_indices]
            means = {i: mean for i, mean in enumerate(means_list)}
        else:
            means = assign_success_probabilities(G)

    # Generate the base filename (without algorithm-specific suffix)
    base_filename = generate_filename(
        "{name}", num_arms, list_size, num_rounds, num_simulations, suffix
    )

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

    if not all_files_exist or args.rerun_all:
    # Run simulation and save data
        run_and_save_simulation_data(base_filename + ".npz", CascadingUnimodalBandit, algorithms, algorithm_names, num_arms, list_size, num_rounds, num_simulations, means, G, reward_scaling=data_rates, single_hop_exploration=single_hop_exploration,
        save=True, rng=rng)
    else:
        print("All individual simulation data files exist. Skipping simulation.")

    # Load and plot data from individual files
    save_as = graph_filename.replace(".pickle", f"_list{list_size}_regret.png")
    load_and_plot_individual_simulation_data(base_filename, algorithm_names, save_as=save_as)


if __name__ == "__main__":
    main()
