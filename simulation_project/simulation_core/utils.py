# File: simulation_project/simulation_core/utils.py

import numpy as np

def calc_quad_tw(alpha):
    """Calculates field quadratures for TW representation."""
    X1 = (alpha + np.conjugate(alpha)) / 2.0
    X2 = (alpha - np.conjugate(alpha)) / (2j)
    return np.real(X1), np.real(X2)

def generate_complex_gaussian_noise_tw(variance=0.5):
    """
    Generates complex Gaussian noise for TW initial conditions.
    For TW, <|eta|^2> should be 1/2, so variance here refers to <|eta|^2>.
    Var(Re(eta)) = Var(Im(eta)) = variance / 2.
    """
    std_dev_part = np.sqrt(variance / 2.0)
    real_part = np.random.normal(0, std_dev_part)
    imag_part = np.random.normal(0, std_dev_part)
    return real_part + 1j * imag_part