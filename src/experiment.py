from src.rg_integrator import rg_propagator
from src.xp_config import ExperimentConfig
import numpy as np
from tqdm import tqdm
#from src.ed_integrator import  # IGNORE

class Experiment:

    ###Initialize the experiment with the given configuration
    def __init__(self, config:ExperimentConfig):

        #integrator can be rg_propagator or ed_propagator (default: rg_propagator)
        self.integrator_func = config.integrator_func

        #Indicate whether all the states towards the evolution are stored
        self.store_state = config.store_state
        
        #parameters of the photon wavepackets and of the cavity
        self.param_photons = config.param_photons
        self.param_cavity = config.param_cavity
        self.param_time_evol = config.param_time_evol

        #ir and uv cutoffs for the frequency modes
        self.ir_cutoff = config.cutoffs['ir_cutoff']
        self.uv_cutoff = config.cutoffs['uv_cutoff']

        #Dictionnary to store potential error messages or warnings
        self.messages = {}

        #Frequency modes array assoiciated with the experiment
        omega_tab_half = np.array([2*np.pi*n/self.param_cavity['L'] for n in range(1000000) \
                                   if (2*np.pi*n/self.param_cavity['L'] <= self.uv_cutoff and \
                                       2*np.pi*n/self.param_cavity['L'] >= self.ir_cutoff)])
        
        self.omega_tab = np.concatenate((omega_tab_half, omega_tab_half))
        self.n_modes = len(omega_tab_half)

        #initialize arrays to store the time evolution of the state
        if self.store_state:
            self.c_array = np.zeros((int(self.param_time_evol['T']/self.param_time_evol['dt']), 2*self.n_modes, 2*self.n_modes), dtype=complex)
            self.b1_array =  np.zeros((int(self.param_time_evol['T']/self.param_time_evol['dt']), 2*self.n_modes), dtype=complex)
            self.b2_array =  np.zeros((int(self.param_time_evol['T']/self.param_time_evol['dt']), 2*self.n_modes), dtype=complex)

            #initialize arrays to store the excited state population and the photon's repartition
            self.An_array = np.zeros(int(self.param_time_evol['T']/self.param_time_evol['dt']), dtype=float)
            self.P11n_array = np.zeros(int(self.param_time_evol['T']/self.param_time_evol['dt']), dtype=float)
            self.P12n_array = np.zeros(int(self.param_time_evol['T']/self.param_time_evol['dt']), dtype=float)
            self.P21n_array = np.zeros(int(self.param_time_evol['T']/self.param_time_evol['dt']), dtype=float)
            self.P22n_array = np.zeros(int(self.param_time_evol['T']/self.param_time_evol['dt']), dtype=float)

        else:
            self.c_array = np.zeros((2*self.n_modes, 2*self.n_modes), dtype=complex)
            self.b1_array =  np.zeros(2*self.n_modes, dtype=complex)
            self.b2_array =  np.zeros(2*self.n_modes, dtype=complex)
        
            #initialize arrays to store the excited state population and the photon's repartition
            self.An_array = np.zeros(1, dtype=float)
            self.P11n_array = np.zeros(1, dtype=float)
            self.P12n_array = np.zeros(1, dtype=float)
            self.P21n_array = np.zeros(1, dtype=float)
            self.P22n_array = np.zeros(1, dtype=float)


        #test monochromatic limit

        for i in range(2): #test for both photons
            if self.param_photons['delta_k'][i] / (self.param_cavity['gamma'] / 2) > 0.1:
                self.messages['monochromatic_limit_'+str(i)] = "Warning: Photon "+str(i)+"'s wavepacket is not in the monochromatic limit."
    
    def check_parameters(self):
        #to be completed: check that the config
        return True

    def propagate_state(self, progress:bool=False):

        #initialize the state
        b1_init = np.zeros(2*self.n_modes)
        b2_init = np.zeros(2*self.n_modes)

        #joint wavefunction for the two-photon state 
        c1 = np.exp(-(self.omega_tab - self.param_photons['omega_p'][0])**2 /(4*self.param_photons['delta_k'][0]**2)) \
                 * np.exp(-1j * self.omega_tab * self.param_photons['x_0'][0]) \
                    * np.concatenate((np.ones(self.n_modes), np.zeros(self.n_modes)))
        c1 = c1 / np.linalg.norm(c1)
        
        c2 = np.exp(-(self.omega_tab - self.param_photons['omega_p'][1])**2 /(4*self.param_photons['delta_k'][1]**2)) \
                 * np.exp(-1j * self.omega_tab * self.param_photons['x_0'][1]) \
                    * np.concatenate((np.zeros(self.n_modes), np.ones(self.n_modes)))
        c2 = c2 / np.linalg.norm(c2)


        c_init = 1/np.sqrt(2) * (c1[:, np.newaxis] * c2[np.newaxis, :] + c2[:, np.newaxis] * c1[np.newaxis, :])

        #propagate the state using the selected integrator
        c_array, b1_array, b2_array = self.integrator_func(c_init, b1_init, b2_init, self.omega_tab, self.param_cavity, self.param_time_evol, 
                                                           progress=progress, store_state = self.store_state)

        self.c_array = c_array
        self.b1_array = b1_array
        self.b2_array = b2_array

        return c_array, b1_array, b2_array
    
    def compute_observables(self, progress:bool=False):

        #compute the excited state population and photon number at each time step

        if self.store_state:
            for i in tqdm(range(len(self.c_array)), disable=not progress):
                self.An_array[i] = np.sum(np.abs(self.b1_array[i])**2) + np.sum(np.abs(self.b2_array[i])**2)
                self.P11n_array[i] = np.sum(np.abs(self.c_array[i, :self.n_modes, :self.n_modes])**2)
                self.P12n_array[i] = np.sum(np.abs(self.c_array[i, :self.n_modes, self.n_modes:])**2)
                self.P21n_array[i] = np.sum(np.abs(self.c_array[i, self.n_modes:, :self.n_modes])**2)
                self.P22n_array[i] = np.sum(np.abs(self.c_array[i, self.n_modes:, self.n_modes:])**2)
        else:
            self.An_array[-1] = np.sum(np.abs(self.b1_array)**2) + np.sum(np.abs(self.b2_array)**2)
            self.P11n_array[-1] = np.sum(np.abs(self.c_array[:self.n_modes, :self.n_modes])**2)
            self.P12n_array[-1] = np.sum(np.abs(self.c_array[:self.n_modes, self.n_modes:])**2)
            self.P21n_array[-1] = np.sum(np.abs(self.c_array[self.n_modes:, :self.n_modes])**2)
            self.P22n_array[-1] = np.sum(np.abs(self.c_array[self.n_modes:, self.n_modes:])**2)

        #test probability conservation at final time
        if not np.isclose(self.P11n_array[-1] + self.P12n_array[-1] + self.P21n_array[-1] + self.P22n_array[-1] + self.An_array[-1], 1.0, atol=1e-3) :
            self.messages['probability_conservation'] = f"Error: Probability not conserved at final time: P11 + P12 + P21 + P22 + A = \
            {self.P11n_array[-1] + self.P12n_array[-1] + self.P21n_array[-1] + self.P22n_array[-1] + self.An_array[-1]:.6f} != 1."

        #Check population of excited atomic state at final time
        if self.An_array[-1] > 1e-2:
            self.messages['final_excited_state'] = f"Warning: The population of the excited state at final time is {self.An_array[-1]:.2e}, which is significant."

        return self.An_array, self.P11n_array, self.P12n_array, self.P21n_array, self.P22n_array

    def get_messages(self):
        if self.messages:
            print("Current messsages:")
            for msg in self.messages:
                print("-", msg)
                print(self.messages[msg])

        return self.messages
        
