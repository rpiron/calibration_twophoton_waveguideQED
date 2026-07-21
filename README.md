# Two-Photon Renormalization

Numerical code and data for two-photon scattering simulations with cutoff-dependent renormalization.

The repository is organized around three layers:

- `src/`: core numerical routines and experiment abstractions.
- `experiments/`: Python functions that run parameter sweeps and generate CSV data.
- `notebooks/`: analysis notebooks used to launch experiments and generate figures.
- `results/`: stored CSV outputs and generated figures.

## Repository Layout

```text
.
├── src/
│   ├── experiment.py              # Experiment class: state initialization, propagation, observables
│   ├── xp_config.py               # Dataclass for experiment parameters
│   ├── rk_integrator.py           # Runge-Kutta time evolution
│   ├── bare_param.py              # Bare-parameter inversion / renormalization helpers
│   ├── coincidence_theory.py      # Analytical coincidence estimate
│   └── ed_integrator.py           # Optional exact-diagonalization/Qutip integrator
├── experiments/
│   ├── coincidence_vs_bandwidth.py # Coincidence versus spectral bandwidth
│   ├── coincidence_vs_frequency.py# Coincidence versus photon frequency
│   └── coincidence_vs_n.py        # Coincidence versus renormalization order
├── notebooks/
│   ├── coincidence_vs_bandwidth_results.ipynb
│   ├── coincidence_vs_frequency_results.ipynb
│   ├── coincidence_vs_n_results.ipynb
│   ├── figure_dispersion.ipynb
│   ├── renorm_inversion.ipynb
│   └── experiment_class_example.ipynb
└── results/
    ├── csv_files/
    │   ├── coincidence_vs_bandwidth/
    │   ├── coincidence_vs_frequency/
    │   └── coincidence_vs_n/
    └── fig/
```

## Python Environment

The core scripts use:

- Python 3
- `numpy`
- `scipy`
- `pandas`
- `tqdm`
- `matplotlib` for notebooks and figure generation
- `qutip` only for the optional `src/ed_integrator.py`

Example setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy scipy pandas tqdm matplotlib jupyter qutip
```

## Usage

The notebooks are the recommended entry point. They are intended to be run from the `notebooks/` directory, so that relative paths such as `../results/...` resolve to the repository-level `results/` directory.

```bash
cd notebooks
jupyter lab
```

Typical workflow:

1. Open the relevant notebook in `notebooks/`.
2. Run the cells that define physical parameters and launch the corresponding function from `experiments/`.
3. Generated CSV files are written under `results/csv_files/`.
4. Plotting cells read the stored CSV files and write figures under `results/fig/`.

The experiment functions can also be imported directly:

```python
from experiments.coincidence_vs_n import run_coincidence_vs_n
from experiments.coincidence_vs_frequency import run_coincidence_vs_frequency
from experiments.coincidence_vs_bandwidth import run_coincidence_vs_bandwidth
```

For lightweight tests or exploratory runs, pass `store_results=False` to avoid writing CSV files.

## Data Organization

Generated CSV files are stored in:

```text
results/csv_files/
├── coincidence_vs_bandwidth/
│   ├── monochr/<experiment_index>/
│   └── non_monochr/<experiment_index>/
├── coincidence_vs_frequency/
│   ├── monochr/
│   └── non_monochr/
└── coincidence_vs_n/
```

Figures are stored in:

```text
results/fig/
```

## Reproducibility Notes

- `src/` contains the reusable simulation code.
- `experiments/` contains sweep-level functions used by the notebooks.
- `notebooks/` records the parameter choices used to generate the stored data and figures.
- `results/` contains the generated outputs that are referenced by the plotting notebooks.

For public archival use, the repository should avoid committing generated Python caches, operating-system metadata, and obsolete backup directories.
