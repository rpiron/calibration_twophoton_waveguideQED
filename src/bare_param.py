import numpy as np
from scipy.optimize import root

pi = np.pi

def get_bare_param_n2(omega_A, gamma_A, ir, uv, precision=1e-4):
    """
    Computes the bare parameters by inverting the renormalization relations
    (omega_0, gamma_0) = F(omega_A, gamma_A, omega_ref, lbda)
    
    Parameters:
    omega_A : physical transition frequency of the TLS
    gamma_A : physical decay rate of the TLS
    ir: infrared cutoff of the spectral density
    uv: ultraviolet cutoff of the spectral density
    precision : convergence tolerance for the bisection search

    Returns:
    omega_0 : bare frequency to parameterize the Hamiltonian
    gamma_0 : bare decay rate to parameterize the Hamiltonian
    """
    
    # Express the bare decay rate as a function of the bare frequency.
    def adjust_gamma_0(x):
        return gamma_A / (1 - gamma_A/(2*pi) * ((ir + omega_A - 2*x)/(ir - x)**2 - (uv + omega_A - 2*x)/(uv - x)**2))

    # Obtain the bare frequency with a bisection search.
    w0_inf = 1.01*ir
    w0_sup = 0.99*uv
    residual_diff = np.inf
    N_iter = 0

    while np.abs(residual_diff) > precision and N_iter < 1e4:  # Bisection search.

        w0_guess = 0.5*(w0_inf + w0_sup)
        gamma_0_guess = adjust_gamma_0(w0_guess)
        residual_diff = -1*gamma_0_guess/(4*pi) * (1/(ir - w0_guess)**2 - 1/(uv - w0_guess)**2) * (gamma_A**2 / 4 - (w0_guess - omega_A)**2) \
                        - (1 + gamma_0_guess/(2*pi) * (1/(ir - w0_guess) - 1/(uv - w0_guess))) * (w0_guess - omega_A) \
                        + gamma_0_guess / (2*pi) * np.log((uv - w0_guess)/(w0_guess - ir))

        if residual_diff > 0:
            w0_inf = w0_guess
        else:
            w0_sup = w0_guess
        N_iter += 1

    # Update the bare decay rate using the final frequency estimate.
    gamma_0_guess = adjust_gamma_0(w0_guess)

    if N_iter <= 1e4:
        return w0_guess, gamma_0_guess
    else:
        return np.nan, np.nan


def get_bare_param_n1(omega_A, gamma_A, ir, uv, precision=1e-4):
    """
    Computes the bare parameters by inverting the renormalization relations
    (omega_0, gamma_0) = F(omega_A, gamma_A, omega_ref, lbda)
    
    Parameters:
    omega_A : physical transition frequency of the TLS
    gamma_A : physical decay rate of the TLS
    ir: infrared cutoff of the spectral density
    uv: ultraviolet cutoff of the spectral density
    precision : convergence tolerance for the bisection search

    Returns:
    omega_0 : bare frequency to parameterize the Hamiltonian
    gamma_0 : bare decay rate to parameterize the Hamiltonian
    """
    
    # Obtain the bare frequency with a bisection search.
    w0_inf = 1.01*ir
    w0_sup = 0.99*uv
    residual_diff = np.inf
    N_iter = 0

    while np.abs(residual_diff) > precision and N_iter < 1e4:  # Bisection search.

        w0_guess = 0.5*(w0_inf + w0_sup)
        residual_diff = omega_A - w0_guess + gamma_A /(2*pi) * np.log((uv - w0_guess) / (w0_guess - ir))

        if residual_diff > 0:
            w0_inf = w0_guess
        else:
            w0_sup = w0_guess
        N_iter += 1

    # Deduce the bare decay rate from the final frequency estimate.
    gamma_0 = gamma_A / (1 - gamma_A/(2*pi)*(1/(ir - w0_guess) - 1/(uv - w0_guess)))
    
    if N_iter <= 1e4:
        return w0_guess, gamma_0
    else:
        return np.nan, np.nan


def get_bare_param_n(omega_A, gamma_A, ir, uv, n=1):
    """
    Computes the bare parameters by inverting the renormalization relations
    (omega_0, gamma_0) = F(omega_A, gamma_A, omega_ref, lbda)
    
    Parameters:
    omega_A : physical transition frequency of the TLS
    gamma_A : physical decay rate of the TLS
    ir: infrared cutoff of the spectral density
    uv: ultraviolet cutoff of the spectral density
    n : maximal n to keep in the alpha truncation

    Returns:
    omega_0 : bare frequency to parameterize the Hamiltonian
    gamma_0 : bare decay rate to parameterize the Hamiltonian
    """

    # n=-1 is the baseline with no correction to the bare parameters.
    if n == -1:
        omega_0 =  omega_A
        gamma_0 = gamma_A

    else:
        def F(omega_0_guess, gamma_0_guess):
            # Store the coefficients of the truncated expansion.
            polynom_tab = np.zeros(n+1, dtype=complex)

            X = (1j* (omega_0_guess - omega_A) - gamma_A/2)

            polynom_tab[0] = -gamma_0_guess/2 + 1j*gamma_0_guess/(2*pi)*np.log(np.abs((uv-omega_0_guess)/(omega_0_guess - ir)))

            for i in range(1, n+1):
                # This is the truncated expansion used for the numerical inversion.
                polynom_tab[i] = (-1j)**(i-1) * gamma_0_guess / (2*i*pi) * \
                                ((omega_0_guess - ir)**(-i) + (-1)**(i-1) * (uv - omega_0_guess)**(-i)) \
                                * X**i

            error_term = np.sum(polynom_tab) - X

            return error_term
        
        def F_real(vars):
            omega_0, gamma_0 = vars
            val = F(omega_0, gamma_0)
            return [val.real, val.imag]

        initial_guess = get_bare_param_n1(omega_A, gamma_A, ir, uv)
        sol = root(F_real, initial_guess, tol=1e-10)
            
        omega_0, gamma_0 = sol.x

    return omega_0, gamma_0
