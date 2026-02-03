import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd

project_root = Path().resolve().parents[0]
sys.path.append(str(project_root))

#Local imports
from src.xp_config import ExperimentConfig
from src.experiment import Experiment
from src.bare_param import get_bare_param

pi = np.pi

def run_antiHOM_vs_bandwith(omega_ref, lbda_tab, index_experiment = 0, store_results:bool=True, progress:bool=True):
    """
    Computes the antiHOM probability at omega_q = omega_A - Gamma/2 for different bandwith
    
    Parameters:
    omega_ref (float) : Center of the frequency window
    lbda_tab (np.array) : Array of different bandwith, defining several frequency window to test

    Returns:
    antiHOM_proba_tab (np.array) : Array of measured anti HOM probabilities for each frequency window
    """

    #Parameterize the experiment

    omega_A = 10*pi
    Gamma = 5*pi

    antiHOM_proba_tab = np.zeros(len(lbda_tab))

    for i in tqdm(range(len(lbda_tab)), disable=not progress):

        #Frequency window
        cutoffs = {'ir_cutoff': omega_ref - lbda_tab[i] , 'uv_cutoff': omega_ref + lbda_tab[i]}

        #Bare parameters
        omega_0, gamma = get_bare_param(omega_A, Gamma, omega_ref, lbda_tab[i])

        #Parameters of the simulation
        L = 50

        param_cavity = {'omega_0': omega_0, 'gamma': gamma, 'L': L}

        param_time_evol = {'T': L/2, 'dt': 0.01}

        param_photons = {'omega_p': [omega_A - Gamma/2, omega_A - Gamma/2 ], 
                        'delta_k': [0.05*pi, 0.05*pi],
                        'x_0': [-L/4, -L/4]}

        #Run the scattering experiment
        config = ExperimentConfig(param_photons=param_photons,
                                  param_cavity=param_cavity,
                                  param_time_evol=param_time_evol,
                                  cutoffs=cutoffs,
                                  store_state=False)
        
        experiment = Experiment(config)
        experiment.propagate_state(progress=True)

        #Compute the coindicence only at final time to save computational resources
        n_modes = experiment.n_modes
        P12_final = np.sum(np.abs(experiment.c_array[:n_modes, n_modes:])**2)
        P21_final = np.sum(np.abs(experiment.c_array[n_modes:, :n_modes])**2)

        antiHOM_proba_tab[i] = P12_final + P21_final

        del experiment

    if store_results:
        data_to_save = {'lbda_tab': lbda_tab, 'antiHOM_proba_tab': antiHOM_proba_tab}
        df = pd.DataFrame(data_to_save)
        if index_experiment:
            df.to_csv(project_root / 'results' / 'csv_files' /f'antiHOM_vs_bandwith_{index_experiment}.csv', index=False)
        else:
            df.to_csv(project_root / 'results' / 'csv_files' /'antiHOM_vs_bandwith.csv', index=False)
    
    return lbda_tab, antiHOM_proba_tab