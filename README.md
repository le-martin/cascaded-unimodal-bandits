# WiFi-CUTS: Rate Adaptation with Cascaded Unimodal Multi-Armed Bandits in IEEE 802.11ac Testbed Experiments

[![Open with marimo](https://marimo.io/shield.svg)](https://marimo.app/l/ea2b05)
![GitHub](https://img.shields.io/github/license/le-martin/cascaded-unimodal-bandits)
<!-- [![DOI](https://img.shields.io/badge/doi-10.1109/TWC.2022.3172760-informational)]() -->


```diff
! Update badge information/links
```

This repository is accompanying the paper *"WiFi-CUTS: Rate Adaptation with Cascaded Unimodal Multi-Armed Bandits in IEEE 802.11ac Testbed Experiments"* (M. Le, B. Peng, E. A. Jorswieck. MONTH, IEEE Transaction on Communications. DOI-LINK).

It provides an interactive and guided version of running the numerical simulations and plotting of the regret/reward performance of the proposed Cascaded Unimodal Thompson Sampling (CUTS).


## File List
The following files are provided in this repository:

- `run.sh`: Bash script that runs numerical simulations CUTS with user-defined parameters. The results are saved in the `simdata` folder and can be plotted with the `plot_multi.py` script.
- `plot-multi.py`: Python script for plotting CUTS with different parameter sets for comparison.
- `simdata`: Folder containing all simulation data including realized regret and graph data.
- `Interactive.py`: Marimo notebook for fast, guided and interactive numerical simulation of CUTS and performance simulation for comparison of CUTS variants.


## Usage
### Running it online
The easiest way is to use the official [marimo](https://marimo.app/) playground
to run the notebook online. Simply navigate to [https://marimo.app/l/ea2b05](https://marimo.app/l/ea2b05)
to run the notebooks in your browser without setting everything up locally.

### Local Installation
If you want to run it locally on your machine, Python3 and marimo are needed.
The present code was developed and tested with the following versions:
```diff
- Python 3.12.9
- numpy 2.3.4
- pandas 2.3.3
- networkx 3.5
- matplotlib 3.10.7
- tqdm 4.67.1
```

Make sure you have [Python3](https://www.python.org/downloads/) installed on
your computer.
You can then install the required packages by either running
```bash
pip3 install -r requirements.txt
```
This will install all the needed packages which are listed in the requirements 
file.


For fast, guided and interactive CUTS simulations and visualization of their reward/regret performance, you can run the Marimo notebooks with
```bash
marimo run Interactive.py
```

You can also manually run numerical simulation of CUTS by running
```bash
bash run.sh
```
The used bandit and graph parameters are defined in `run.sh`. Adjust them to obtain CUTS performance results for different parameter sets.

The simulation results are saved in the folder `simdata`. To plot the regret/reward performance of CUTS for different parameter sets, adjust the parameter sets / scenarios to compare with each other and run
```bash
python3 plot_multi.py
```


## Acknowledgements
This research was supported by the Federal Ministry of Education and Research of Germany as part of the joint project “Software-Driven Urban plus Rural Area Communication Networks (SupraCoNeX)”, project identification number: 16KIS1193, the project ML4RIS funded by German Research Foundation (DFG) under grant 566937681, and partly by the Federal Ministry of Education and Research (BMBF), Germany, through the Program of Souverän, Digital, and Vernetzt Joint Project 6G-RIC under Grant 16KISK031.


## License and Referencing
This program is licensed under the MIT license. If you in any way use this
code for research that results in publications, please cite our original
article listed above.

<!-- You can use the following BibTeX entry
```bibtex
@article{...,
  author = {...},
  title = {...},
  ...
}
``` -->
```diff
! Add bibtex entry of the published paper
```
