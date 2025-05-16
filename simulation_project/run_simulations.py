# File: simulation_project/run_simulations.py

import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import argparse
import time
import multiprocessing

from simulation_core.tw_simulation import run_full_simulation_tw
from simulation_core.pp_simulation import run_pplus_simulation_parallel
from data_handling.storage import (
    generate_filename,
    save_simulation_data,
    load_simulation_data
)
from plotting.plot_squeezing import create_squeezing_plot
from plotting.plot_pump_depletion import create_pump_depletion_plot
from plotting.plot_photon_tracker import create_photon_tracker_plot
from plotting.plot_seed_variance import create_seed_variance_plot

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main(config_path):
    config = load_config(config_path)

    sim_params = config['simulation_params']
    photon_regimes_config = config['photon_regimes']
    data_conf = config['data_handling']
    plot_conf = config['plotting']

    kappa = sim_params['kappa']
    num_traj_main = sim_params['num_trajectories_main']
    num_seed_runs_plot4 = sim_params['num_seed_runs_plot4']
    global_seeds_plot4 = sim_params['global_seeds_plot4']
    if len(global_seeds_plot4) < num_seed_runs_plot4:
        print(f"Warning: Not enough seeds for {num_seed_runs_plot4} runs. Using {len(global_seeds_plot4)}.")
        num_seed_runs_plot4 = len(global_seeds_plot4)

    regime_physical_t_ends = sim_params['regime_physical_t_end']
    num_points = sim_params['num_simulation_points']
    if num_points <= 1:
        raise ValueError("num_simulation_points must be greater than 1.")
    
    num_workers = sim_params.get('num_workers_pplus', None)

    data_dir = data_conf['data_dir']
    force_rerun = data_conf['force_rerun_all']
    plot_dir = plot_conf['plot_dir']
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    plt.style.use(plot_conf.get('plot_style', 'seaborn-v0_8-whitegrid'))

    print(f"--- Simulation Configuration ---")
    print(f"Kappa: {kappa}, Main Trajectories: {num_traj_main}")
    print(f"Number of Simulation Points: {num_points} (i.e., {num_points - 1} steps)")
    print(f"Data Directory: {os.path.abspath(data_dir)}")
    print(f"Plot Directory: {os.path.abspath(plot_dir)}")
    print(f"Force Rerun All: {force_rerun}")
    print(f"Regime Physical End Times: {regime_physical_t_ends}")
    print("-" * 30)

    all_results_data = {}
    simulation_methods = {"TW": run_full_simulation_tw, "PP": run_pplus_simulation_parallel}
    overall_start_time = time.time()

    for regime_key, n_val in photon_regimes_config.items():
        all_results_data[regime_key] = {}

        current_physical_t_end = regime_physical_t_ends.get(regime_key)
        if current_physical_t_end is None:
            print(f"ERROR: Physical end time for regime '{regime_key}' not defined. Skipping.")
            continue
        if current_physical_t_end <= 0:
            print(f"ERROR: Physical end time for regime '{regime_key}' must be positive. Skipping.")
            continue
            
        # Calculate physical dt for this regime to achieve num_points
        current_physical_dt = current_physical_t_end / (num_points - 1)
        
        print(f"\nProcessing: Regime='{regime_key}' (n={n_val})")
        print(f"  Physical t_end={current_physical_t_end}, Physical dt={current_physical_dt:.4e} ({num_points-1} steps)")

        current_tspan_sim = (0, current_physical_t_end)
        # t_eval for solve_ivp and for storing results should use num_points
        t_eval_physical = np.linspace(0, current_physical_t_end, num_points)


        for method_key, sim_function in simulation_methods.items():
            print(f"  Method='{method_key}'")
            all_results_data[regime_key][method_key] = {}
            all_results_data[regime_key][method_key]['seed_runs_plot4'] = []

            seeds_to_run = global_seeds_plot4[:num_seed_runs_plot4]
            if not seeds_to_run :
                seeds_to_run = [0]
                if num_seed_runs_plot4 > 0:
                     print("  Warning: num_seed_runs_plot4 > 0 but global_seeds_plot4 empty. Using default seed [0].")

            for seed_iter_idx, current_run_seed_val in enumerate(seeds_to_run):
                is_main_run_for_plots123 = (seed_iter_idx == 0)

                params_dict_run = {
                    'method': method_key, 'regime': regime_key, 'n': n_val,
                    'kappa': kappa, 'num_traj': num_traj_main,
                    'physical_t_end': current_physical_t_end, # For filename hash
                    'num_points': num_points,                 # For filename hash
                    'seed_run_type': f'seed_{current_run_seed_val}',
                    'global_seed': current_run_seed_val
                }
                if method_key == "PP":
                     # Add physical_dt to params if it's crucial for hash,
                     # though it's derived from t_end and num_points.
                     # params_dict_run['physical_dt_pplus'] = current_physical_dt
                     pass


                filepath_run = generate_filename(params_dict_run, data_dir)
                run_type_msg = "Main/Seed" if is_main_run_for_plots123 else "Seed"
                print(f"    {run_type_msg} run (seed {current_run_seed_val}). File: {os.path.basename(filepath_run)}")
                
                current_run_data = None
                if not force_rerun: current_run_data = load_simulation_data(filepath_run)

                if current_run_data is None:
                    print(f"      Running simulation for {run_type_msg.lower()} run (seed {current_run_seed_val})...")
                    run_start_time = time.time()
                    
                    if method_key == "TW":
                        # run_full_simulation_tw expects tspan and produces its own t_eval based on 500 points.
                        # We need to modify run_full_simulation_tw to accept t_eval_physical
                        # OR adjust its internal linspace to use num_points.
                        # For now, assuming run_full_simulation_tw will be adapted or its internal
                        # t_eval is sufficient, and we use t_eval_physical for consistency if needed later.
                        # Let's assume for now it's called as before but we note the discrepancy.
                        # A better way would be to modify run_full_simulation_tw to accept num_points or t_eval.
                        # For this example, let's pass t_eval_physical to the function.
                        # This implies run_full_simulation_tw needs to be updated to use it.
                        current_run_data = sim_function(n_val, num_traj_main, kappa, current_tspan_sim, 
                                                        current_seed=current_run_seed_val, 
                                                        t_eval_hint=t_eval_physical) # Pass t_eval_physical as a hint
                    elif method_key == "PP":
                        current_run_data = sim_function(n_val, num_traj_main, kappa, 
                                                        current_tspan_sim, current_physical_dt, # Pass calculated physical dt
                                                        num_workers=num_workers, current_seed=current_run_seed_val)
                    
                    # Ensure the 'time_physical' key in current_run_data aligns with t_eval_physical
                    # This might require adjustment in the simulation functions themselves to return
                    # results at these specific time points.
                    if current_run_data is not None and 'time_physical' in current_run_data:
                        if not np.allclose(current_run_data['time_physical'], t_eval_physical):
                            print("      Warning: Output time points from simulation may not exactly match desired t_eval_physical.")
                            # Potentially interpolate results here if necessary, though ideally sim functions handle it.
                        current_run_data['time_physical'] = t_eval_physical # Standardize for consistency if possible
                    
                    save_simulation_data(filepath_run, current_run_data)
                    run_end_time = time.time()
                    print(f"      Sim for {run_type_msg.lower()} (seed {current_run_seed_val}) done: {run_end_time - run_start_time:.2f}s")
                
                all_results_data[regime_key][method_key]['seed_runs_plot4'].append(current_run_data)
                if is_main_run_for_plots123:
                    all_results_data[regime_key][method_key]['main'] = current_run_data
            
            if 'main' not in all_results_data[regime_key][method_key] and all_results_data[regime_key][method_key]['seed_runs_plot4']:
                all_results_data[regime_key][method_key]['main'] = all_results_data[regime_key][method_key]['seed_runs_plot4'][0]
            elif 'main' not in all_results_data[regime_key][method_key]:
                 print(f"Error: No simulation data for {method_key} in {regime_key} to assign to 'main'.")

    overall_end_time = time.time()
    print(f"\nAll simulations/data loading finished in {overall_end_time - overall_start_time:.2f} seconds.")
    print("-" * 30)
    print("Generating plots...")

    # Pass the whole config to plotting functions as they might need various parts of it
    if plot_conf.get('generate_plot1_squeezing', False):
        create_squeezing_plot(all_results_data, config, plot_dir)
    if plot_conf.get('generate_plot2_pump_depletion', False):
        create_pump_depletion_plot(all_results_data, config, plot_dir)
    if plot_conf.get('generate_plot3_photon_tracker', False):
        create_photon_tracker_plot(all_results_data, config, plot_dir)
    if plot_conf.get('generate_plot4_seed_variance', False):
        create_seed_variance_plot(all_results_data, config, plot_dir)

    print("-" * 30)
    print("Script finished successfully.")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="Run TW and +P simulations for PDC.")
    parser.add_argument('--config', type=str, default="config.yaml",
                        help="Path to the configuration YAML file. Default: config.yaml in script directory.")
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_path = args.config
    if not os.path.isabs(config_file_path):
        config_file_path = os.path.join(script_dir, config_file_path)
    if not os.path.exists(config_file_path):
        print(f"Error: Configuration file '{config_file_path}' not found.")
        exit(1)
    main(config_file_path)