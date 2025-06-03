# File: simulation_project/simulation_core/tw_simulation.py

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from .utils import calc_quad_tw, generate_complex_gaussian_noise_tw

def pdc_full_system_tw(t, y, kappa):
    """ODE system for Truncated Wigner (TW) simulation."""
    re_alpha1, im_alpha1, re_alpha2, im_alpha2 = y
    alpha1 = re_alpha1 + 1j * im_alpha1
    alpha2 = re_alpha2 + 1j * im_alpha2
    dalpha1_dt = kappa * np.conjugate(alpha1) * alpha2
    dalpha2_dt = -(kappa / 2.0) * alpha1**2
    return [
        np.real(dalpha1_dt), np.imag(dalpha1_dt),
        np.real(dalpha2_dt), np.imag(dalpha2_dt)
    ]


def run_full_simulation_tw(n_current, num_trajectories, kappa, tspan_current_regime, 
                           current_seed=None, t_eval_hint=None): # Added t_eval_hint
    """
    Runs TW simulation for a given regime and seed.
    Returns a dictionary of results.
    If t_eval_hint is provided, it's used for the output time points.
    """
    if current_seed is not None:
        np.random.seed(current_seed)

    # Use t_eval_hint if provided, otherwise default to 500 points
    if t_eval_hint is not None and len(t_eval_hint) > 1:
        # Ensure t_eval_hint starts from tspan_current_regime[0] and ends at tspan_current_regime[1]
        # This assumes t_eval_hint is already correctly constructed to match the tspan.
        # If t_eval_hint might be slightly off due to linspace, ensure first/last points match tspan
        # For robustness, we could reconstruct it if only num_points was passed,
        # but run_simulations.py now passes the full array.
        t_eval_current = np.array(t_eval_hint, dtype=float)
        if not np.isclose(t_eval_current[0], tspan_current_regime[0]) or \
           not np.isclose(t_eval_current[-1], tspan_current_regime[1]):
            print("Warning: t_eval_hint in run_full_simulation_tw does not perfectly match tspan_current_regime.")
            print(f"t_eval_hint: [{t_eval_current[0]}, ..., {t_eval_current[-1]}]")
            print(f"tspan_current_regime: {tspan_current_regime}")
            # For safety, ensure the t_eval used by solve_ivp is within the tspan
            t_eval_for_solver = t_eval_current[(t_eval_current >= tspan_current_regime[0]) & (t_eval_current <= tspan_current_regime[1])]
            if len(t_eval_for_solver) < 2 : # not enough points
                 t_eval_for_solver = np.linspace(tspan_current_regime[0], tspan_current_regime[1], 2) # min 2 points
        else:
            t_eval_for_solver = t_eval_current

    else:
        t_eval_for_solver = np.linspace(tspan_current_regime[0], tspan_current_regime[1], 500) # Fallback
        t_eval_current = t_eval_for_solver # Ensure t_eval_current is defined for output structure size

    num_output_points = len(t_eval_current)

    alpha1_all_traj = np.zeros((num_trajectories, num_output_points), dtype=complex)
    alpha2_all_traj = np.zeros((num_trajectories, num_output_points), dtype=complex)
    
    X1_signal_all_traj = np.zeros((num_trajectories, num_output_points))
    X2_signal_all_traj = np.zeros((num_trajectories, num_output_points))
    X1_pump_all_traj = np.zeros((num_trajectories, num_output_points))
    X2_pump_all_traj = np.zeros((num_trajectories, num_output_points))

    n1_tw_all_traj = np.zeros((num_trajectories, num_output_points))
    n2_tw_all_traj = np.zeros((num_trajectories, num_output_points))

    for i in range(num_trajectories):
        eta1 = generate_complex_gaussian_noise_tw(variance=0.5) 
        eta2 = generate_complex_gaussian_noise_tw(variance=0.5)
        alpha1_0 = 0 + eta1
        alpha2_0 = np.sqrt(n_current) + eta2
        y0_tw = np.array([np.real(alpha1_0), np.imag(alpha1_0), np.real(alpha2_0), np.imag(alpha2_0)])
        
        solution = solve_ivp(
            pdc_full_system_tw, tspan_current_regime, y0_tw, args=(kappa,),
            method='RK45', t_eval=t_eval_for_solver, # Use t_eval_for_solver here
            rtol=1e-8, atol=1e-8,
            max_step=max(0.001, (tspan_current_regime[1] - tspan_current_regime[0]) / 100.0) 
        )
        
        t_sol, y_sol = solution.t, solution.y
        alpha1_traj_points_current_solver = y_sol[0] + 1j * y_sol[1]
        alpha2_traj_points_current_solver = y_sol[2] + 1j * y_sol[3]

        # Interpolate solution onto t_eval_current if solve_ivp didn't use exactly those points
        # This is crucial if t_eval_for_solver had to be clipped or if solve_ivp chose different steps.
        if not np.allclose(t_sol, t_eval_current):
            # print(f"Trajectory {i}: Interpolating TW results to {num_output_points} desired time points.")
            interp_alpha1 = interp1d(t_sol, alpha1_traj_points_current_solver, kind='cubic', bounds_error=False, fill_value="extrapolate")
            interp_alpha2 = interp1d(t_sol, alpha2_traj_points_current_solver, kind='cubic', bounds_error=False, fill_value="extrapolate")
            alpha1_this_traj = interp_alpha1(t_eval_current)
            alpha2_this_traj = interp_alpha2(t_eval_current)
        else: # If t_sol perfectly matches t_eval_current
            alpha1_this_traj = alpha1_traj_points_current_solver
            alpha2_this_traj = alpha2_traj_points_current_solver
        
        alpha1_all_traj[i, :] = alpha1_this_traj
        alpha2_all_traj[i, :] = alpha2_this_traj

        X1_s, X2_s = calc_quad_tw(alpha1_this_traj)
        X1_p, X2_p = calc_quad_tw(alpha2_this_traj)
        X1_signal_all_traj[i,:] = X1_s
        X2_signal_all_traj[i,:] = X2_s
        X1_pump_all_traj[i,:] = X1_p
        X2_pump_all_traj[i,:] = X2_p

        n1_tw_all_traj[i,:] = np.abs(alpha1_this_traj)**2 - 0.5
        n2_tw_all_traj[i,:] = np.abs(alpha2_this_traj)**2 - 0.5
            
    alpha1_avg_tw = np.mean(alpha1_all_traj, axis=0)
    alpha2_avg_tw = np.mean(alpha2_all_traj, axis=0)
    
    X1_signal_avg_tw = np.mean(X1_signal_all_traj, axis=0)
    X2_signal_avg_tw = np.mean(X2_signal_all_traj, axis=0)
    X1_signal_var_tw = np.var(X1_signal_all_traj, axis=0)
    X2_signal_var_tw = np.var(X2_signal_all_traj, axis=0)
    
    X1_pump_avg_tw = np.mean(X1_pump_all_traj, axis=0)
    X2_pump_avg_tw = np.mean(X2_pump_all_traj, axis=0)
    X1_pump_var_tw = np.var(X1_pump_all_traj, axis=0)
    X2_pump_var_tw = np.var(X2_pump_all_traj, axis=0)
    
    n1_proxy_avg_tw = np.abs(alpha1_avg_tw)**2
    n2_proxy_avg_tw = np.abs(alpha2_avg_tw)**2

    n1_actual_avg_tw = np.mean(n1_tw_all_traj, axis=0)
    n2_actual_avg_tw = np.mean(n2_tw_all_traj, axis=0)
            
    return {
        "time_physical": t_eval_current, # Return the actual time points used for the output arrays
        "alpha1_avg": alpha1_avg_tw, "alpha2_avg": alpha2_avg_tw,
        "X1_signal_avg": X1_signal_avg_tw, "X2_signal_avg": X2_signal_avg_tw,
        "X1_signal_var": X1_signal_var_tw, "X2_signal_var": X2_signal_var_tw,
        "X1_pump_avg": X1_pump_avg_tw, "X2_pump_avg": X2_pump_avg_tw,
        "X1_pump_var": X1_pump_var_tw, "X2_pump_var": X2_pump_var_tw,
        "n1_proxy_avg": n1_proxy_avg_tw, "n2_proxy_avg": n2_proxy_avg_tw,
        "n1_actual_avg": n1_actual_avg_tw, "n2_actual_avg": n2_actual_avg_tw,
    }