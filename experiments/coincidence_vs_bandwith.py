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

def run_coincidence_vs_bandwith(omega_q, ir_tab, uv_tab, index_omega_q = 0, index_experiment = 0, 
                                sym_variables:bool=True, store_results:bool=True, progress:bool=True):
    """
    Computes the coincidence probability at omega_q for different frequency windows
    
    Parameters:
    omega_q (float) : Photons incoming frequency
    ir_tab (np.array) : Array of different ir cutoffs
    uv_tab (np.array) : Array of different uv cutoffs

    Returns:
    antiHOM_proba_tab (np.array) : Array of measured anti HOM probabilities for each frequency window
    """

    #Parameterize the experiment

    omega_A = 10*pi
    Gamma = 5*pi
    coincidence_tab = np.zeros(len(ir_tab))
    for i in tqdm(range(len(ir_tab)), disable=not progress):

        #Frequency window
        cutoffs = {'ir_cutoff': ir_tab[i] , 'uv_cutoff': uv_tab[i]}

        #Sanity check
        if omega_q < cutoffs['ir_cutoff'] or omega_q > cutoffs['uv_cutoff'] :
            print("WARNING : The photon frequency is not included in the frequency window. Returning NaN")
            coincidence_tab[i] = np.nan
        
        else:
            #Bare parameters
            omega_0, gamma = get_bare_param(omega_A, Gamma, ir_tab[i], uv_tab[i])

            #Parameters of the simulation
            L = 50

            param_cavity = {'omega_0': omega_0, 'gamma': gamma, 'L': L}

            param_time_evol = {'T': L/2, 'dt': 0.05}

            param_photons = {'omega_p': [omega_q, omega_q], 
                            'delta_k': [0.05*pi, 0.05*pi],
                            'x_0': [-L/4, -L/4]}

            #Run the scattering experiment
            config = ExperimentConfig(param_photons=param_photons,
                                    param_cavity=param_cavity,
                                    param_time_evol=param_time_evol,
                                    cutoffs=cutoffs,
                                    store_state=False)
            
            experiment = Experiment(config)
            experiment.propagate_state(progress=False)

            #Compute the coindicence only at final time to save computational resources
            n_modes = experiment.n_modes
            P12_final = np.sum(np.abs(experiment.c_array[:n_modes, n_modes:])**2)
            P21_final = np.sum(np.abs(experiment.c_array[n_modes:, :n_modes])**2)

            coincidence_tab[i] = P12_final + P21_final

            del experiment

    if store_results:
        data_to_save = {'ir_tab': ir_tab, 'uv_tab': uv_tab, 'coincidence_tab': coincidence_tab}
        df = pd.DataFrame(data_to_save)

        if index_experiment:
            if sym_variables:
                df.to_csv(project_root / 'results' / 'csv_files' / 'sym_variables' /f'coincidence_vs_bandwith_{index_omega_q}_{index_experiment}.csv', index=False)
            else:
                df.to_csv(project_root / 'results' / 'csv_files' / 'iruv_variables' /f'coincidence_vs_bandwith_{index_omega_q}_{index_experiment}.csv', index=False)
        
        else:
            df.to_csv(project_root / 'results' / 'csv_files' /'coincidence_vs_bandwith.csv', index=False)
    
    return ir_tab, uv_tab, coincidence_tab