import numpy as np
pi = np.pi

def get_C(omega_A, gamma_A, omega_q, sigma_q = 0, nonlin_correction=False):
        
        R_theory_physical = 1 / (1 + ((omega_q- omega_A)/ (gamma_A/2))**2)

        theoretical_val = 1 - 4*R_theory_physical*(1-R_theory_physical)

        if nonlin_correction:
            try:
                non_monochr_ratio = gamma_A / (2*sigma_q)
            except Exception:
                 print("sigma_q should be non zero")
                 non_monochr_ratio + 0

            theoretical_val += non_monochr_ratio

        return theoretical_val