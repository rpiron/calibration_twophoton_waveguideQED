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

def run_coincidence_vs_n(omega_q, ir_tab, uv_tab, index_omega_q = 0, n=1, 
                         store_results:bool=True, progress:bool=True):
    """
    To complete
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
            try:
                omega_0, gamma = get_bare_param_n(omega_A, Gamma, ir_tab[i], uv_tab[i], n=n)

                #Parameters of the simulation
                L = 50

                param_cavity = {'omega_0': omega_0, 'gamma': gamma, 'L': L}

                param_time_evol = {'T': L/2, 'dt': 0.01}

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
                _, _, P12n_array, P21n_array, _ = experiment.compute_observables()

                coincidence_tab[i] = P12n_array[-1] + P21n_array[-1]

                del experiment

            except Exception:
                print("WARNING : Bare parameters not found. Returning NaN")
                coincidence_tab[i] = np.nan

    if store_results:
        data_to_save = {'ir_tab': ir_tab, 'uv_tab': uv_tab, 'coincidence_tab': coincidence_tab}
        df = pd.DataFrame(data_to_save)
        df.to_csv(project_root / 'results' / 'csv_files' / f'coincidence_vs_n{n}_ir{int(ir_tab[0]/pi)}_{index_omega_q}.csv', index=False)
    
    return ir_tab, uv_tab, coincidence_tab