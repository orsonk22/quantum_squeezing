# File: simulation_project/simulation_core/pp_simulation.py
# MODIFIED to ensure dW is real in pdc_pplus_step_numba as per supervisor feedback

import numpy as np
import numba
import multiprocessing
from functools import partial
import time # For timing within the function if needed

@numba.jit(nopython=True, nogil=True) 
def pdc_pplus_step_numba(y, kappa, dt): # Name kept as per original file provided
    """
    Numba jitted step function for +P.
    MODIFIED to use REAL Wiener increments dW, which are then multiplied by complex diffusion coefficients.
    """
    re_a1, im_a1, re_a1p, im_a1p, re_a2, im_a2, re_a2p, im_a2p = y
    alpha1 = re_a1 + 1j * im_a1
    alpha1_plus = re_a1p + 1j * im_a1p
    alpha2 = re_a2 + 1j * im_a2
    alpha2_plus = re_a2p + 1j * im_a2p
    
    sqrt_dt = np.sqrt(dt)
    # Generate two independent REAL Gaussian random numbers N(0,1)
    z1 = np.random.normal(0.0, 1.0)
    z2 = np.random.normal(0.0, 1.0)

    # Form REAL Wiener increments
    dw1_real_increment = z1 * sqrt_dt
    dw2_real_increment = z2 * sqrt_dt

    # Drift terms
    drift_a1_complex = kappa * alpha1_plus * alpha2
    drift_a1p_complex = kappa * alpha1 * alpha2_plus
    drift_a2_complex = -(kappa / 2.0) * alpha1**2
    drift_a2p_complex = -(kappa / 2.0) * alpha1_plus**2
    
    # Complex diffusion coefficients
    diffusion_coeff1 = np.sqrt(kappa * alpha2 + 0j) 
    diffusion_coeff1_plus = np.sqrt(kappa * alpha2_plus + 0j)

    y_next = np.empty_like(y)

    # Stochastic term for alpha1: diffusion_coeff1 (complex) * dw1_real_increment (real)
    stochastic_product1 = diffusion_coeff1 * dw1_real_increment
    stoch_term1_re = np.real(stochastic_product1)
    stoch_term1_im = np.imag(stochastic_product1)
    
    y_next[0] = y[0] + np.real(drift_a1_complex) * dt + stoch_term1_re
    y_next[1] = y[1] + np.imag(drift_a1_complex) * dt + stoch_term1_im
    
    # Stochastic term for alpha1_plus: diffusion_coeff1_plus (complex) * dw2_real_increment (real)
    stochastic_product1_plus = diffusion_coeff1_plus * dw2_real_increment
    stoch_term2_re = np.real(stochastic_product1_plus)
    stoch_term2_im = np.imag(stochastic_product1_plus)

    y_next[2] = y[2] + np.real(drift_a1p_complex) * dt + stoch_term2_re
    y_next[3] = y[3] + np.imag(drift_a1p_complex) * dt + stoch_term2_im
    
    # Pump modes are deterministic
    y_next[4] = y[4] + np.real(drift_a2_complex) * dt
    y_next[5] = y[5] + np.imag(drift_a2_complex) * dt
    y_next[6] = y[6] + np.real(drift_a2p_complex) * dt
    y_next[7] = y[7] + np.imag(drift_a2p_complex) * dt
    
    return y_next

@numba.jit(nopython=True, nogil=True) 
def run_single_trajectory_numba_pplus(y0_traj, num_steps_traj, kappa_traj, dt_traj):
    """Numba jitted single trajectory runner for +P."""
    alpha1_hist = np.empty(num_steps_traj + 1, dtype=np.complex128)
    alpha1_plus_hist = np.empty(num_steps_traj + 1, dtype=np.complex128)
    alpha2_hist = np.empty(num_steps_traj + 1, dtype=np.complex128)
    alpha2_plus_hist = np.empty(num_steps_traj + 1, dtype=np.complex128)

    y_curr = y0_traj.copy()
    alpha1_hist[0] = y_curr[0] + 1j * y_curr[1]
    alpha1_plus_hist[0] = y_curr[2] + 1j * y_curr[3]
    alpha2_hist[0] = y_curr[4] + 1j * y_curr[5]
    alpha2_plus_hist[0] = y_curr[6] + 1j * y_curr[7]

    for step in range(num_steps_traj):
        y_curr = pdc_pplus_step_numba(y_curr, kappa_traj, dt_traj) # Calls the now-modified step function
        alpha1_hist[step + 1] = y_curr[0] + 1j * y_curr[1]
        alpha1_plus_hist[step + 1] = y_curr[2] + 1j * y_curr[3]
        alpha2_hist[step + 1] = y_curr[4] + 1j * y_curr[5]
        alpha2_plus_hist[step + 1] = y_curr[6] + 1j * y_curr[7]
        
    return alpha1_hist, alpha1_plus_hist, alpha2_hist, alpha2_plus_hist

def worker_simulate_trajectory_pplus(traj_idx_seed_tuple, y0_worker, num_steps_worker, kappa_worker, dt_worker):
    """Worker function for multiprocessing in +P. Handles individual trajectory seeding."""
    _, seed_for_traj = traj_idx_seed_tuple 
    if seed_for_traj is not None:
        np.random.seed(seed_for_traj) 
    return run_single_trajectory_numba_pplus(y0_worker, num_steps_worker, kappa_worker, dt_worker)

def run_pplus_simulation_parallel(n_current, num_trajectories, kappa, tspan_current_regime, dt_current_regime, num_workers=None, current_seed=None):
    """
    Runs +P simulation in parallel for a given regime and global_seed for the run.
    Returns a dictionary of results.
    (Quadrature definitions remain as they were in the provided version of this file)
    """
    if current_seed is not None:
        np.random.seed(current_seed) 

    trajectory_seeds = None
    if current_seed is not None:
        seed_seq = np.random.SeedSequence(current_seed)
        trajectory_seeds = seed_seq.generate_state(num_trajectories)

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    t_start, t_end = tspan_current_regime
    num_steps = max(1, int(round((t_end - t_start) / dt_current_regime)))
    t_eval_current = np.linspace(t_start, t_start + num_steps * dt_current_regime, num_steps + 1)

    alpha1_0 = 0 + 0j
    alpha1_plus_0 = 0 + 0j
    alpha2_0 = np.sqrt(n_current) + 0j
    alpha2_plus_0 = np.sqrt(n_current) + 0j
    y0_pp = np.array([
        np.real(alpha1_0), np.imag(alpha1_0), np.real(alpha1_plus_0), np.imag(alpha1_plus_0),
        np.real(alpha2_0), np.imag(alpha2_0), np.real(alpha2_plus_0), np.imag(alpha2_plus_0)
    ], dtype=np.float64)

    _ = run_single_trajectory_numba_pplus(y0_pp, 1, kappa, dt_current_regime if dt_current_regime > 0 else 0.001)
    
    worker_partial_pplus = partial(worker_simulate_trajectory_pplus, y0_worker=y0_pp, 
                                   num_steps_worker=num_steps, kappa_worker=kappa, dt_worker=dt_current_regime)
    
    map_args = [(i, trajectory_seeds[i] if trajectory_seeds is not None else None) for i in range(num_trajectories)]

    with multiprocessing.Pool(processes=num_workers) as pool:
        results_pplus = pool.map(worker_partial_pplus, map_args)
        
    alpha1_all_traj = np.array([res[0] for res in results_pplus])
    alpha1_plus_all_traj = np.array([res[1] for res in results_pplus])
    alpha2_all_traj = np.array([res[2] for res in results_pplus])
    alpha2_plus_all_traj = np.array([res[3] for res in results_pplus])

    alpha1_avg_pp = np.mean(alpha1_all_traj, axis=0)
    alpha1_plus_avg_pp = np.mean(alpha1_plus_all_traj, axis=0)
    alpha2_avg_pp = np.mean(alpha2_all_traj, axis=0)
    alpha2_plus_avg_pp = np.mean(alpha2_plus_all_traj, axis=0)

    # Quadrature calculations remain as they were in the version of this file you provided
    X1_signal_avg_pp = 0.5 * np.real(alpha1_avg_pp + alpha1_plus_avg_pp)
    X2_signal_avg_pp = 0.5 * np.imag(alpha1_avg_pp - alpha1_plus_avg_pp) 
    X1_pump_avg_pp = 0.5 * np.real(alpha2_avg_pp + alpha2_plus_avg_pp)
    X2_pump_avg_pp = 0.5 * np.imag(alpha2_avg_pp - alpha2_plus_avg_pp)

    X1_signal_sq_avg_pp = 0.25 * np.real(np.mean(alpha1_all_traj**2 + alpha1_plus_all_traj**2 + 
                                            2 * alpha1_all_traj * alpha1_plus_all_traj + 1.0, axis=0)) # Ensured 1.0
    X1_signal_var_pp = X1_signal_sq_avg_pp - X1_signal_avg_pp**2
    
    X2_signal_sq_avg_pp = 0.25 * np.real(np.mean(-(alpha1_all_traj**2 + alpha1_plus_all_traj**2 -
                                             2 * alpha1_all_traj * alpha1_plus_all_traj - 1.0), axis=0)) # Ensured 1.0
    X2_signal_var_pp = X2_signal_sq_avg_pp - X2_signal_avg_pp**2

    X1_pump_sq_avg_pp = 0.25 * np.real(np.mean(alpha2_all_traj**2 + alpha2_plus_all_traj**2 +
                                          2 * alpha2_all_traj * alpha2_plus_all_traj + 1.0, axis=0)) # Ensured 1.0
    X1_pump_var_pp = X1_pump_sq_avg_pp - X1_pump_avg_pp**2
    
    X2_pump_sq_avg_pp = 0.25 * np.real(np.mean(-(alpha2_all_traj**2 + alpha2_plus_all_traj**2 - 
                                           2 * alpha2_all_traj * alpha2_plus_all_traj - 1.0), axis=0)) # Ensured 1.0
    X2_pump_var_pp = X2_pump_sq_avg_pp - X2_pump_avg_pp**2
    
    n1_proxy_avg_pp = np.abs(alpha1_avg_pp)**2
    n2_proxy_avg_pp = np.abs(alpha2_avg_pp)**2

    n1_actual_avg_pp = np.real(np.mean(alpha1_all_traj * alpha1_plus_all_traj, axis=0))
    n2_actual_avg_pp = np.real(np.mean(alpha2_all_traj * alpha2_plus_all_traj, axis=0))
    
    return {
        "time_physical": t_eval_current,
        "alpha1_avg": alpha1_avg_pp, "alpha2_avg": alpha2_avg_pp,
        "alpha1_plus_avg": alpha1_plus_avg_pp, "alpha2_plus_avg": alpha2_plus_avg_pp,
        "X1_signal_avg": X1_signal_avg_pp, "X2_signal_avg": X2_signal_avg_pp,
        "X1_signal_var": X1_signal_var_pp, "X2_signal_var": X2_signal_var_pp,
        "X1_pump_avg": X1_pump_avg_pp, "X2_pump_avg": X2_pump_avg_pp,
        "X1_pump_var": X1_pump_var_pp, "X2_pump_var": X2_pump_var_pp,
        "n1_proxy_avg": n1_proxy_avg_pp, "n2_proxy_avg": n2_proxy_avg_pp,
        "n1_actual_avg": n1_actual_avg_pp, "n2_actual_avg": n2_actual_avg_pp,
    }