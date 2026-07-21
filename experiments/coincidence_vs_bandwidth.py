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
from src.bare_param import get_bare_param_n

pi = np.pi

def run_coincidence_vs_bandwidth(param_photons, param_cavity_physical, param_time_evol, ir_tab, uv_tab, 
                                index_omega_q = 0, index_experiment = 0, n= 1, 
                                monochr:bool = True, store_results:bool=True, progress:bool=True):
    """Compute the coincidence probability as a function of the cutoff window.

    The physical cavity parameters are converted to cutoff-dependent bare
    parameters for each infrared/ultraviolet cutoff pair.  The final
    coincidence probability is then evaluated by propagating the two-photon
    state.  Results can optionally be written to the corresponding CSV file.
    """

    # Allocate the output array.
    coincidence_tab = np.zeros(len(ir_tab))

    for i in tqdm(range(len(ir_tab)), disable=not progress):

        # Define the frequency window.
        cutoffs = {'ir_cutoff': ir_tab[i] , 'uv_cutoff': uv_tab[i]}

        # Check that the input photon frequency lies inside the window.
        if param_photons['omega_p'][0] < cutoffs['ir_cutoff'] or param_photons['omega_p'][0] > cutoffs['uv_cutoff'] :
            print("WARNING : The photon frequency is not included in the frequency window. Returning NaN")
            coincidence_tab[i] = np.nan
        
        else:
            # Compute the cutoff-dependent bare parameters.
            try:
                omega_0, gamma_0 = get_bare_param_n(param_cavity_physical['omega_A'], 
                                                  param_cavity_physical['gamma_A'], 
                                                  cutoffs['ir_cutoff'], 
                                                  cutoffs['uv_cutoff'], n=n)

                # Assemble the simulation parameters.
                param_cavity = {'omega_0': omega_0, 'gamma_0': gamma_0, 'L': param_cavity_physical['L']}

                # Run the scattering experiment.
                config = ExperimentConfig(param_photons=param_photons,
                                        param_cavity=param_cavity,
                                        param_time_evol=param_time_evol,
                                        cutoffs=cutoffs,
                                        store_state=False)
                
                experiment = Experiment(config)
                experiment.propagate_state(progress=False)

                # Compute the coincidence only at the final time.
                _, _, P12n_array, P21n_array, _ = experiment.compute_observables()

                coincidence_tab[i] = P12n_array[-1] + P21n_array[-1]

                del experiment

            except Exception:
                print("WARNING : Bare parameters not found. Returning NaN")
                coincidence_tab[i] = np.nan

    if store_results:
        data_to_save = {'ir_tab': ir_tab, 'uv_tab': uv_tab, 'coincidence_tab': coincidence_tab}
        df = pd.DataFrame(data_to_save)

        if monochr:
            df.to_csv(project_root / 'results' / 'csv_files' / 'coincidence_vs_bandwidth' / 'monochr' / f'{index_experiment}' / f'coincidence_vs_bandwidth_omega{index_omega_q}_xp{index_experiment}_n{n}.csv', index=False)
        else:
            df.to_csv(project_root / 'results' / 'csv_files' / 'coincidence_vs_bandwidth' / 'non_monochr' / f'{index_experiment}' / f'coincidence_vs_bandwidth_omega{index_omega_q}_xp{index_experiment}_n{n}.csv', index=False)
        
    return ir_tab, uv_tab, coincidence_tab
