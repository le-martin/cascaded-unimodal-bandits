#!/bin/bash

# Define bandit parameters
num_arms=232  # Number of arms
num_rounds=2000  # Number of rounds
num_simulations=10000  # Number of simulations

edge_prob=0.02  # Edge probability for Erdos-Renyi graph

rerun_all=0  # Set to 1 to rerun all simulations even if simulation results exist

seed=2024  # Random seed for reproducibility

# For graph types 'full', 'line' and 'erdos-renyi'
for graph in "full" "line" "erdos-renyi"; do
 for list_size in 1 4; do
   echo "Running: list_size=${list_size}, graph_type=${graph}, p=${edge_prob}"
   python cuts.py \
     --num_arms ${num_arms} \
     --list_size ${list_size} \
     --num_rounds ${num_rounds} \
     --num_simulations ${num_simulations} \
     --graph_type ${graph} \
     --p ${edge_prob} \
     --rerun_all ${rerun_all} \
     --seed ${seed}
 done
done
