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

def run_coincidence_vs_rkstep(param_photons, param_cavity_physical, T, N_step_tab, cutoffs, n,
                              index_experiment = 0, store_results:bool=True, progress:bool=True):
    
    coincidence_tab = np.zeros(len(N_step_tab))
    
    for i in tqdm(range(len(N_step_tab)), disable=not progress):

        dt = T / N_step_tab[i]
        param_time_evol = {'T': T, 'dt': dt}
    
        #get the bare parameters : n=-1 serves as a baseline (no renormalization)
        omega_0, gamma = get_bare_param_n(param_cavity_physical['omega_A'], 
                                          param_cavity_physical['Gamma'], 
                                          cutoffs['ir_cutoff'], 
                                          cutoffs['uv_cutoff'], n=n)
        
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
    
    if store_results:
        data_to_save = {'N_step_tab' : N_step_tab, 'coincidence_tab': coincidence_tab}
        df = pd.DataFrame(data_to_save)
        if n == -1:
            df.to_csv(project_root / 'results' / 'csv_files' / f'coincidence_vs_rkstep_norenorm_{index_experiment}.csv', index=False)
        else:
            df.to_csv(project_root / 'results' / 'csv_files' / f'coincidence_vs_rkstep_n{n}_{index_experiment}.csv', index=False)
        
    return N_step_tab, coincidence_tab