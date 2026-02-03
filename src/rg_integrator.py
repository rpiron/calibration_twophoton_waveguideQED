import numpy as np
from tqdm import tqdm

def increment_state(c,b1,b2,omega_tab,param_cavity,param_time_evol, t):
    """
    Computes the infinitesimal increment (c_new, b1_new, b2_new) = -i * dt * V_I(t) (c,b1,b2)
    
    Parameters:
    c (np array): Coefficients in front of the states |1_k, 1_k', g>
    b1 (np array): Coefficients in front of the states |1_k, 0, e >
    b2 (np array): Coefficients in front of the states |0, 1_k', e >
    omega_tab (np array): Array of containing two copies of the frequency modes.
    param_cavity: dictionnary of parameters {omega_0, gamma, L, T, dt}
    param_time_evol: dictionnary of parameters {T, dt, n_time_step}
    
    Returns:
    c_new (np array): Updated coefficients.
    b1_new (float): Updated atomic coefficient.
    b2_new (float): Updated atomic coefficient.
    """
    omega_0 = param_cavity['omega_0']
    gamma = param_cavity['gamma']
    L = param_cavity['L']

    dt = param_time_evol['dt']

    V_matrix = -1j * np.sqrt(gamma / (2*L)) * np.exp(1j * (omega_tab - omega_0) * t)

    c_new = b1[:,np.newaxis] @ V_matrix[np.newaxis,:] + V_matrix[:,np.newaxis] @ b2[np.newaxis,:] 
    b1_new = c @ np.conjugate(V_matrix)
    b2_new = np.conjugate(V_matrix) @ c 
    

    return -1j*dt*c_new, -1j*dt*b1_new, -1j*dt*b2_new


def rg_propagator(c_init, b1_init, b2_init, omega_tab, param_cavity, param_time_evol, progress:bool=False, store_state:bool=True):
    """
    Propagates the state of the system using the RK4 scheme.
    
    Parameters:
    c_init (np array): initial coefficients in front of the states |1_k, 1_k', g>
    b1_init (np array): initial atomic coefficients in front of the state |1_k, 0, e>
    b12init (np array): initial atomic coefficients in front of the state |0, 1_k', e>
    omega_tab (array): Array of containing two copies of the frequency modes.
    param_time_evol: dictionnary of parameters {T, dt, n_time_step}
    
    Returns:
    c_array (np array): Array of c coefficients at each time step.
    b1_array (np array): Array of b1 coefficients at each time step.
    b2_array (np array): Array of b2 coefficients at each time step.
    """
    
    dt = param_time_evol['dt']
    n_time_step = int(param_time_evol['T'] / dt)

    n_modes = len(c_init)//2

    c_current = c_init
    b1_current = b1_init
    b2_current = b2_init

    if store_state:
        c_array = np.zeros((n_time_step, 2*n_modes, 2*n_modes), dtype=complex)
        b1_array = np.zeros((n_time_step, 2*n_modes), dtype=complex)
        b2_array = np.zeros((n_time_step, 2*n_modes), dtype=complex)
        # Set the initial conditions
        c_array[0] = c_init
        b1_array[0] = b1_init
        b2_array[0] = b2_init

    # Time evolution loop
    for i in tqdm(range(1, n_time_step), disable=not progress):
        t = i * dt

        c_n1, b1_n1, b2_n1 = increment_state(c_current, b1_current, b2_current, omega_tab, param_cavity, param_time_evol, t)
        c_n2, b1_n2, b2_n2 = increment_state(c_current + c_n1/2, b1_current + b1_n1/2, b2_current + b2_n1/2, omega_tab, param_cavity, param_time_evol, t + dt/2)
        c_n3, b1_n3, b2_n3 = increment_state(c_current + c_n2/2, b1_current + b1_n2/2, b2_current + b2_n2/2, omega_tab, param_cavity, param_time_evol, t + dt/2)
        c_n4, b1_n4, b2_n4 = increment_state(c_current + c_n3, b1_current + b1_n3, b2_current + b2_n3, omega_tab, param_cavity, param_time_evol, t + dt)

        c_new = c_current + (c_n1 + 2*c_n2 + 2*c_n3 + c_n4) / 6
        b1_new = b1_current + (b1_n1 + 2*b1_n2 + 2*b1_n3 + b1_n4) / 6
        b2_new = b2_current + (b2_n1 + 2*b2_n2 + 2*b2_n3 + b2_n4) / 6


        if store_state:
            c_array[i] = c_new
            b1_array[i] = b1_new
            b2_array[i] = b2_new
        
        c_current = c_new
        b1_current = b1_new
        b2_current = b2_new

    if store_state:
        return c_array, b1_array, b2_array 
    else:
        return c_current, b1_current, b2_current
    