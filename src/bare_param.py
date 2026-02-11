import numpy as np
pi = np.pi

def get_bare_param(omega_A, Gamma, ir, uv, precision=1e-4):
    """
    Computes the bare parameters by inverting the renormalization relations
    (omega_0, gamma) = F(omega_A, Gamma, omega_ref, lbda)
    
    Parameters:
    omega_A : physical transition frequency of the TLS
    Gamma : physical decay rate of the TLS
    ir: infrared cutoff of the spectral density
    uv: ultraviolet cutoff of the spectral density
    precision : precision parameter for the dichotomy research

    Returns:
    omega_0 : bare frequency to parameterize the Hamiltonian
    gamma : bare decay rate to parameterize the Hamiltonian
    """
    
    ##Express the bare decay as a function of omega_0
    def adjust_gamma(x):
        return Gamma / (1 - Gamma/(2*pi) * ((ir + omega_A - 2*x)/(ir - x)**2 - (uv + omega_A - 2*x)/(uv - x)**2))

    ## Obtain the bare frequenc by a dichtomy search
    w0_inf = 1.01*ir
    w0_sup = 0.99*uv
    residual_diff = np.inf

    while np.abs(residual_diff) > precision: #research by dichotomy

        w0_guess = 0.5*(w0_inf + w0_sup)
        gamma_guess = adjust_gamma(w0_guess)
        residual_diff = gamma_guess/(4*pi) * (1/(ir - w0_guess)**2 - 1/(uv - w0_guess)**2) * (Gamma**2 / 4 - (w0_guess - omega_A)**2) \
                        - (1 + gamma_guess/(2*pi) * (1/(ir - w0_guess) - 1/(uv - w0_guess))) * (w0_guess - omega_A) \
                        + gamma_guess / (2*pi) * np.log((uv - w0_guess)/(w0_guess - ir))

        if residual_diff > 0:
            w0_inf = w0_guess
        else:
            w0_sup = w0_guess

    #Last round needed to update gamma
    gamma_guess = adjust_gamma(w0_guess)

    return w0_guess, gamma_guess


def get_bare_param_first_order(omega_A, Gamma, ir, uv, precision=1e-4):
    """
    Computes the bare parameters by inverting the renormalization relations
    (omega_0, gamma) = F(omega_A, Gamma, omega_ref, lbda)
    
    Parameters:
    omega_A : physical transition frequency of the TLS
    Gamma : physical decay rate of the TLS
    ir: infrared cutoff of the spectral density
    uv: ultraviolet cutoff of the spectral density
    precision : precision parameter for the dichotomy research

    Returns:
    omega_0 : bare frequency to parameterize the Hamiltonian
    gamma : bare decay rate to parameterize the Hamiltonian
    """
    
    ## Obtain the bare frequenc by a dichtomy search
    w0_inf = 1.01*ir
    w0_sup = 0.99*uv
    residual_diff = np.inf

    while np.abs(residual_diff) > precision: #research by dichotomy

        w0_guess = 0.5*(w0_inf + w0_sup)
        residual_diff = omega_A - w0_guess + Gamma /(2*pi) * np.log((uv - w0_guess) / (w0_guess - ir))

        if residual_diff > 0:
            w0_inf = w0_guess
        else:
            w0_sup = w0_guess

    #Deduce gamma accordingly
    gamma = Gamma / (1 - Gamma/(2*pi)*(1/(ir - w0_guess) - 1/(uv - w0_guess)))
    
    return w0_guess, gamma