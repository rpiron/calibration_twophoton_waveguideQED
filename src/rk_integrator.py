import numpy as np
from tqdm import tqdm

def increment_state(c,b,omega_tab,param_cavity,param_time_evol, t):
    """
    Computes the infinitesimal increment (c_new, b_new) = -i * dt * V_I(t) (c,b)
    
    Parameters:
    c (np array): Coefficients in front of the states |alpha k, beta p, g>
    b (np array): Coefficients in front of the states |alpha k, e >
    omega_tab (np array): Array of containing two copies of the frequency modes.
    param_cavity: dictionnary of parameters {omega_0, gamma, L, T, dt}
    param_time_evol: dictionnary of parameters {T, dt, n_time_step}
    
    Returns:
    c_new (np array): Updated coefficients.
    b_new (np array): Updated atomic coefficients.
    """
    omega_0 = param_cavity['omega_0']
    gamma_0 = param_cavity['gamma']
    L = param_cavity['L']

    dt = param_time_evol['dt']

    V_matrix = -1j * np.sqrt(gamma_0 / (2*L)) * np.exp(1j * (omega_tab - omega_0) * t)

    c_new = 1/2*(b[:,np.newaxis] @ V_matrix[np.newaxis,:] + V_matrix[:,np.newaxis] @ b[np.newaxis,:])
    b_new = 2 * c @ np.conjugate(V_matrix) 
    

    return -1j*dt*c_new, -1j*dt*b_new


def rk_propagator(c_init, b_init, omega_tab, param_cavity, param_time_evol, progress:bool=False, store_state:bool=True):
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
    b_current = b_init

    if store_state:
        c_array = np.zeros((n_time_step, 2*n_modes, 2*n_modes), dtype=complex)
        b_array = np.zeros((n_time_step, 2*n_modes), dtype=complex)
        # Set the initial conditions
        c_array[0] = c_init
        b_array[0] = b_init

    # Time evolution loop
    for i in tqdm(range(1, n_time_step), disable=not progress):
        t = i * dt

        c_n1, b_n1 = increment_state(c_current, b_current, omega_tab, param_cavity, param_time_evol, t)
        c_n2, b_n2 = increment_state(c_current + c_n1/2, b_current + b_n1/2, omega_tab, param_cavity, param_time_evol, t + dt/2)
        c_n3, b_n3 = increment_state(c_current + c_n2/2, b_current + b_n2/2, omega_tab, param_cavity, param_time_evol, t + dt/2)
        c_n4, b_n4 = increment_state(c_current + c_n3, b_current + b_n3, omega_tab, param_cavity, param_time_evol, t + dt)

        c_new = c_current + (c_n1 + 2*c_n2 + 2*c_n3 + c_n4) / 6
        b_new = b_current + (b_n1 + 2*b_n2 + 2*b_n3 + b_n4) / 6


        if store_state:
            c_array[i] = c_new
            b_array[i] = b_new
        
        c_current = c_new
        b_current = b_new

    if store_state:
        return c_array, b_array
    else:
        return c_current, b_current
    