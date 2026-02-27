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
from src.bare_param import get_bare_param_n

pi = np.pi

def run_coincidence_vs_n(param_photons, param_cavity_physical, param_time_evol, cutoffs, n_tab,
                         index_omega_q = 0, index_experiment=0, store_results:bool=True, progress:bool=True):
    """
    To complete
    """

    coincidence_tab = np.zeros(len(n_tab))

    for i in tqdm(range(len(n_tab)), disable=not progress):

        #Try to get the bare parameters
        try:
            omega_0, gamma = get_bare_param_n(param_cavity_physical['omega_A'], 
                                              param_cavity_physical['Gamma'], 
                                              cutoffs['ir_cutoff'], 
                                              cutoffs['uv_cutoff'], 
                                              n=n_tab[i])

            #Parameters of the simulation
            param_cavity = {'omega_0': omega_0, 'gamma': gamma, 'L': param_cavity_physical['L']}

            #Run the scattering experiment
            config = ExperimentConfig(param_photons=param_photons,
                                      param_cavity=param_cavity,
                                      param_time_evol=param_time_evol,
                                      cutoffs=cutoffs,
                                      store_state=False)
                
            experiment = Experiment(config)
            experiment.propagate_state(progress=False)

            #Compute the coindicence only at final time to save computational resources
            _, _, P12n_array, P21n_array, _ = experiment.compute_observables()

            coincidence_tab[i] = P12n_array[-1] + P21n_array[-1]

            del experiment

        except Exception:
            print("WARNING : Bare parameters not found. Returning NaN")
            coincidence_tab[i] = np.nan

    if store_results:
        data_to_save = {'n_tab': n_tab, 'coincidence_tab': coincidence_tab}
        df = pd.DataFrame(data_to_save)
        df.to_csv(project_root / 'results' / 'csv_files' / f'coincidence_vs_n_omega_{index_omega_q}_window_{index_experiment}.csv', index=False)
    
    return n_tab, coincidence_tab