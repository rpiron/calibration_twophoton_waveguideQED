from src.rk_integrator import rk_propagator
from src.xp_config import ExperimentConfig
import numpy as np
from tqdm import tqdm
# Optional exact-diagonalization integrator.

class Experiment:

    """Represent and propagate a two-photon scattering experiment."""

    def __init__(self, config:ExperimentConfig):
        """Initialize the experiment from an :class:`ExperimentConfig`."""

        # The integrator can be replaced by an alternative propagator.
        self.integrator_func = config.integrator_func

        # Whether the full state history is stored.
        self.store_state = config.store_state
        
        # Parameters of the photon wave packets and the cavity.
        self.param_photons = config.param_photons
        self.param_cavity = config.param_cavity
        self.param_time_evol = config.param_time_evol

        # Infrared and ultraviolet cutoffs for the frequency modes.
        self.ir_cutoff = config.cutoffs['ir_cutoff']
        self.uv_cutoff = config.cutoffs['uv_cutoff']

        # Messages and warnings generated during the calculation.
        self.messages = {}

        # Frequency-mode array associated with the experiment.
        omega_tab_half = np.array([2*np.pi*n/self.param_cavity['L'] for n in range(-1000000,1000000) \
                                   if (2*np.pi*n/self.param_cavity['L'] <= self.uv_cutoff and \
                                       2*np.pi*n/self.param_cavity['L'] >= self.ir_cutoff)])
        
        self.omega_tab = np.concatenate((omega_tab_half, omega_tab_half))
        self.n_modes = len(omega_tab_half)

            # Allocate arrays for the full time evolution.
        if self.store_state:
            self.c_array = np.zeros((int(self.param_time_evol['T']/self.param_time_evol['dt']), 2*self.n_modes, 2*self.n_modes), dtype=complex)
            self.b_array =  np.zeros((int(self.param_time_evol['T']/self.param_time_evol['dt']), 2*self.n_modes), dtype=complex)

            # Allocate arrays for the excited-state population and photon-number sectors.
            self.An_array = np.zeros(int(self.param_time_evol['T']/self.param_time_evol['dt']), dtype=float)
            self.P11n_array = np.zeros(int(self.param_time_evol['T']/self.param_time_evol['dt']), dtype=float)
            self.P12n_array = np.zeros(int(self.param_time_evol['T']/self.param_time_evol['dt']), dtype=float)
            self.P21n_array = np.zeros(int(self.param_time_evol['T']/self.param_time_evol['dt']), dtype=float)
            self.P22n_array = np.zeros(int(self.param_time_evol['T']/self.param_time_evol['dt']), dtype=float)

        else:
            self.c_array = np.zeros((2*self.n_modes, 2*self.n_modes), dtype=complex)
            self.b_array =  np.zeros(2*self.n_modes, dtype=complex)
        
            # Allocate arrays for the final excited-state population and photon-number sectors.
            self.An_array = np.zeros(1, dtype=float)
            self.P11n_array = np.zeros(1, dtype=float)
            self.P12n_array = np.zeros(1, dtype=float)
            self.P21n_array = np.zeros(1, dtype=float)
            self.P22n_array = np.zeros(1, dtype=float)


        # Check the monochromatic-limit condition for both photons.

        for i in range(2):  # Apply the check to both photons.
            if self.param_photons['sigma_w'][i] / (self.param_cavity['gamma_0'] / 2) > 0.1:
                self.messages['monochromatic_limit_'+str(i)] = "Warning: Photon "+str(i)+"'s wavepacket is not in the monochromatic limit."
    
    def check_parameters(self):
        """Validate the experiment configuration.

        The configuration is currently accepted without additional checks.
        The method is kept as an extension point for future parameter validation.
        """
        return True

    def propagate_state(self, progress:bool=False):
        """Initialize and propagate the two-photon state.

        Returns the propagated two-photon and atomic coefficient arrays.  The
        returned arrays contain the full history when ``store_state`` is true,
        and only the final state otherwise.
        """

        # Initialize the atomic component.
        b_init = np.zeros(2*self.n_modes)

        # Construct the joint two-photon wave function.
        c1 = np.exp(-(self.omega_tab - self.param_photons['omega_p'][0])**2 /(4*self.param_photons['sigma_w'][0]**2)) \
                 * np.exp(-1j * self.omega_tab * self.param_photons['x_0'][0]) \
                    * np.concatenate((np.ones(self.n_modes), np.zeros(self.n_modes)))
        c1 = c1 / np.linalg.norm(c1)
        
        c2 = np.exp(-(self.omega_tab - self.param_photons['omega_p'][1])**2 /(4*self.param_photons['sigma_w'][1]**2)) \
                 * np.exp(-1j * self.omega_tab * self.param_photons['x_0'][1]) \
                    * np.concatenate((np.zeros(self.n_modes), np.ones(self.n_modes)))
        c2 = c2 / np.linalg.norm(c2)


        c_init = 1/np.sqrt(2) * (c1[:, np.newaxis] * c2[np.newaxis, :] + c2[:, np.newaxis] * c1[np.newaxis, :])

        # Propagate the state with the selected integrator.
        c_array, b_array = self.integrator_func(c_init, b_init, self.omega_tab, self.param_cavity, self.param_time_evol, 
                                                           progress=progress, store_state = self.store_state)

        self.c_array = c_array
        self.b_array = b_array

        return c_array, b_array
    
    def compute_observables(self, progress:bool=False):
        """Compute atomic and photon-sector populations from the propagated state."""

        # Compute the excited-state population and photon-sector populations.

        if self.store_state:
            for i in tqdm(range(len(self.c_array)), disable=not progress):
                self.An_array[i] = np.sum(np.abs(self.b_array[i])**2)
                self.P11n_array[i] = np.sum(np.abs(self.c_array[i, :self.n_modes, :self.n_modes])**2)
                self.P12n_array[i] = np.sum(np.abs(self.c_array[i, :self.n_modes, self.n_modes:])**2)
                self.P21n_array[i] = np.sum(np.abs(self.c_array[i, self.n_modes:, :self.n_modes])**2)
                self.P22n_array[i] = np.sum(np.abs(self.c_array[i, self.n_modes:, self.n_modes:])**2)
        else:
            self.An_array[-1] = np.sum(np.abs(self.b_array)**2)
            self.P11n_array[-1] = np.sum(np.abs(self.c_array[:self.n_modes, :self.n_modes])**2)
            self.P12n_array[-1] = np.sum(np.abs(self.c_array[:self.n_modes, self.n_modes:])**2)
            self.P21n_array[-1] = np.sum(np.abs(self.c_array[self.n_modes:, :self.n_modes])**2)
            self.P22n_array[-1] = np.sum(np.abs(self.c_array[self.n_modes:, self.n_modes:])**2)

        # Check probability conservation at the final time.
        if not np.isclose(self.P11n_array[-1] + self.P12n_array[-1] + self.P21n_array[-1] + self.P22n_array[-1] + self.An_array[-1], 1.0, atol=1e-3) :
            self.messages['probability_conservation'] = f"Error: Probability not conserved at final time: P11 + P12 + P21 + P22 + A = \
            {self.P11n_array[-1] + self.P12n_array[-1] + self.P21n_array[-1] + self.P22n_array[-1] + self.An_array[-1]:.6f} != 1."

        # Check the residual excited-state population at the final time.
        if self.An_array[-1] > 1e-2:
            self.messages['final_excited_state'] = f"Warning: The population of the excited state at final time is {self.An_array[-1]:.2e}, which is significant."

        return self.An_array, self.P11n_array, self.P12n_array, self.P21n_array, self.P22n_array

    def get_messages(self):
        """Print and return messages generated during the experiment."""
        if self.messages:
            print("Current messsages:")
            for msg in self.messages:
                print("-", msg)
                print(self.messages[msg])

        return self.messages
        
