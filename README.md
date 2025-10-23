# WiFi-CUTS: Rate Adaptation with Cascaded Unimodal Multi-Armed Bandits in IEEE 802.11ac Testbed Experiments

[![Marimo](https://img.shields.io/badge/Launch-Marimo_notebook-hsl(168%2C61%25%2C28%25))](https://marimo.app/?src=https%3A%2F%2Fraw.githubusercontent.com%2Fklb2%2Freproducible-paper-python-template%2Frefs%2Fheads%2Fmaster%2FInteractive.py)
![GitHub](https://img.shields.io/github/license/klb2/reproducible-paper-python-template)
[![DOI](https://img.shields.io/badge/doi-10.1109/TWC.2022.3172760-informational)](https://doi.org/10.1109/TWC.2022.3172760)


```diff
! Update badge information/links
```

This repository is accompanying the paper "WiFi-CUTS: Rate Adaptation with Cascaded Unimodal Multi-Armed Bandits in IEEE 802.11ac Testbed Experiments" (M. Le, B. Peng, E. A. Jorswieck. MONTH, IEEE Transaction on Communications. DOI-LINK).

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
to run the notebook online. Simply navigate to [https://marimo.app/?src=https%3A%2F%2Fraw.githubusercontent.com%2Fklb2%2Freproducible-paper-python-template%2Frefs%2Fheads%2Fmaster%2FInteractive.py](https://marimo.app/?src=https%3A%2F%2Fraw.githubusercontent.com%2Fklb2%2Freproducible-paper-python-template%2Frefs%2Fheads%2Fmaster%2FInteractive.py)
```diff
! Add Marimo app link to notebook.
```
to run the notebooks in your browser without setting everything up locally.

### Local Installation
If you want to run it locally on your machine, Python3 and marimo are needed.
The present code was developed and tested with the following versions:
```diff
- Python 3.13
- numpy 2.2
- scipy 1.15
```

Make sure you have [Python3](https://www.python.org/downloads/) installed on
your computer.
You can then install the required packages by running
```bash
pip3 install -r requirements.txt
```
This will install all the needed packages which are listed in the requirements 
file.


Finally, you can run the Marimo notebooks with
```bash
marimo run Interactive.py
```

You can also recreate the figures from the paper by running
```bash
bash run.sh
```


## Acknowledgements
This research was supported by
```diff
! Add funding information
```


## License and Referencing
This program is licensed under the MIT license. If you in any way use this
code for research that results in publications, please cite our original
article listed above.

You can use the following BibTeX entry
```bibtex
@article{...,
  author = {...},
  title = {...},
  ...
}
```
```diff
! Add bibtex entry of the published paper
```
