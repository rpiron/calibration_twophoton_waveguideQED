import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Local imports.
from src.xp_config import ExperimentConfig
from src.experiment import Experiment

def run_coincidence_vs_frequency(param_photons_bis, param_cavity, param_time_evol, frequency_values, cutoffs, index_experiment=0, 
                                correction:bool=False, monochr:bool=True, store_results:bool=True, progress:bool=True):

    """
    Runs the coincidence against frequency experiment for different photon central frequencies.
    
    Parameters:
    param_photon_bis (Dict) : Dictionary containing only {'sigma_w', 'x_0'} for both photons
    param_cavity (Dict) : Dictionary containing {'omega_0', 'gamma_0', 'L'}
    param_time_evol (Dict) : Dictionary containing {'T', 'dt'}
    cutoffs (Dict) : Dictionary containing {'ir_cutoff', 'uv_cutoff'} 
    frequency_values (np.array) : Array of photon central frequency values
    index_experiment (int) : Index of the experiment if multiple are run in sequence.
    store_results (bool) : Whether to store the results in a CSV file.
    progress (bool) : Whether to display a progress bar.

    Returns:
    frequency_values (np.array) : Array of photon central frequency values used.
    coincidence_tab (np.array) : Final coincidence probability for each frequency.

    """

    # Allocate the output array.
    coincidence_tab = np.zeros(len(frequency_values))

    for i in tqdm(range(len(frequency_values)), disable=not progress):
        omega_p = frequency_values[i]
        param_photons_current = param_photons_bis.copy()
        param_photons_current['omega_p'] = [omega_p, omega_p]

        config = ExperimentConfig(param_photons=param_photons_current,
                                  param_cavity=param_cavity,
                                  param_time_evol=param_time_evol,
                                  cutoffs=cutoffs,
                                  store_state=False)

        experiment = Experiment(config)
        experiment.propagate_state(progress=True)
        # Compute the coincidence only at the final time.
        n_modes = experiment.n_modes
        P12_final = np.sum(np.abs(experiment.c_array[:n_modes, n_modes:])**2)
        P21_final = np.sum(np.abs(experiment.c_array[n_modes:, :n_modes])**2)

        coincidence_tab[i] = P12_final + P21_final

        del experiment
    
    if store_results:
        data_to_save = {'photon_frequency_tab': frequency_values, 'final_reflection_tab': coincidence_tab}
        df = pd.DataFrame(data_to_save)

        if index_experiment:
            if correction and monochr:
                df.to_csv(project_root / 'results' / 'csv_files' / 'coincidence_vs_frequency' / 'monochr' /f'coincidence_vs_frequency_{index_experiment}_corrected.csv', index=False)
            elif correction and not monochr:
                df.to_csv(project_root / 'results' / 'csv_files' / 'coincidence_vs_frequency' / 'non_monochr' /f'coincidence_vs_frequency_{index_experiment}_corrected.csv', index=False)
            elif not correction and monochr:
                df.to_csv(project_root / 'results' / 'csv_files' / 'coincidence_vs_frequency' / 'monochr' /f'coincidence_vs_frequency_{index_experiment}.csv', index=False)
            elif not correction and not monochr:
                df.to_csv(project_root / 'results' / 'csv_files' / 'coincidence_vs_frequency' / 'non_monochr' /f'coincidence_vs_frequency_{index_experiment}.csv', index=False)
        else:
            df.to_csv(project_root / 'results' / 'csv_files' /'coincidence_vs_frequency.csv', index=False)
    
    return frequency_values, coincidence_tab
