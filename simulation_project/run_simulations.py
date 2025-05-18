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
from plotting.plot_max_squeezing_vs_n import create_max_squeezing_vs_n_plot


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
    
    # Seeds for original plots (Plot4 and error band in Plot1 for +P)
    num_seeds_for_pp_plot4 = sim_params.get('num_seed_runs_plot4', 1) # Primarily for +P
    num_seeds_for_tw_plot4 = sim_params.get('num_seed_runs_plot4_tw', 1) # Specific for TW
    
    global_seeds_config_list = sim_params.get('global_seeds_plot4', [0]) # Default to one seed if not provided
    if not global_seeds_config_list: # Ensure it's at least [0] if empty list provided
        global_seeds_config_list = [0]

    # Validate seed numbers against available seeds
    max_seeds_needed_for_orig_regimes = max(num_seeds_for_pp_plot4, num_seeds_for_tw_plot4)
    if len(global_seeds_config_list) < max_seeds_needed_for_orig_regimes:
        print(f"Warning: Not enough seeds in 'global_seeds_plot4' ({len(global_seeds_config_list)}) "
              f"for requested runs (up to {max_seeds_needed_for_orig_regimes}). "
              f"Will use available seeds.")
    
    # Effective number of seeds to process for +P and TW
    effective_num_seeds_pp = min(num_seeds_for_pp_plot4, len(global_seeds_config_list))
    effective_num_seeds_tw = min(num_seeds_for_tw_plot4, len(global_seeds_config_list))
    if effective_num_seeds_tw == 0 and len(global_seeds_config_list) > 0: # Ensure TW runs at least once if seeds are available
        effective_num_seeds_tw = 1


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
    print(f"Kappa: {kappa}, Main Trajectories (per seed run): {num_traj_main}")
    print(f"Number of Simulation Points: {num_points}")
    print(f"Data Directory: {os.path.abspath(data_dir)}")
    print(f"Plot Directory: {os.path.abspath(plot_dir)}")
    print(f"Force Rerun All: {force_rerun}")
    print(f"Effective seeds for +P (Plot1 error band/Plot4): {effective_num_seeds_pp}")
    print(f"Effective seeds for TW (Plot4): {effective_num_seeds_tw}")
    print("-" * 30)

    all_results_data = {}
    simulation_methods = {"TW": run_full_simulation_tw, "PP": run_pplus_simulation_parallel}
    overall_start_time = time.time()

    # --- Original Regime Simulations (for Plots 1-4) ---
    print("\n--- Starting Original Regime Simulations (for Plots 1-4) ---")
    for regime_key, n_val in photon_regimes_config.items():
        all_results_data[regime_key] = {}

        current_physical_t_end = regime_physical_t_ends.get(regime_key)
        if current_physical_t_end is None: print(f"ERROR: No t_end for {regime_key}. Skipping."); continue
        if current_physical_t_end <= 0: print(f"ERROR: t_end for {regime_key} must be > 0. Skipping."); continue
            
        current_physical_dt = current_physical_t_end / (num_points - 1)
        
        print(f"\nProcessing: Regime='{regime_key}' (n={n_val})")
        print(f"  Physical t_end={current_physical_t_end}, Physical dt={current_physical_dt:.4e}")

        current_tspan_sim = (0, current_physical_t_end)
        t_eval_physical = np.linspace(0, current_physical_t_end, num_points)

        for method_key, sim_function in simulation_methods.items():
            print(f"  Method='{method_key}'")
            all_results_data[regime_key][method_key] = {}
            all_results_data[regime_key][method_key]['seed_runs_plot4'] = [] 

            seeds_to_iterate_for_this_method = []
            if method_key == "TW":
                seeds_to_iterate_for_this_method = global_seeds_config_list[:effective_num_seeds_tw]
                print(f"    (TW method: processing {len(seeds_to_iterate_for_this_method)} seed(s))")
            elif method_key == "PP":
                seeds_to_iterate_for_this_method = global_seeds_config_list[:effective_num_seeds_pp]
                print(f"    ({method_key} method: processing {len(seeds_to_iterate_for_this_method)} seed(s))")
            
            if not seeds_to_iterate_for_this_method:
                 print(f"    Warning: No seeds to process for {method_key} in {regime_key}. Ensuring 'main' is handled if possible or skipped.")
                 # Ensure 'main' key exists with None if no runs, to prevent KeyErrors in plotting
                 all_results_data[regime_key][method_key]['main'] = None # Or some default empty structure
                 continue


            for seed_iter_idx, current_run_seed_val in enumerate(seeds_to_iterate_for_this_method):
                is_main_run = (seed_iter_idx == 0) 

                params_dict_run = {
                    'method': method_key, 'regime': regime_key, 'n': n_val,
                    'kappa': kappa, 'num_traj': num_traj_main,
                    'physical_t_end': current_physical_t_end,
                    'num_points': num_points,
                    'seed_run_type': f'seed_{current_run_seed_val}',
                    'global_seed': current_run_seed_val
                }
                filepath_run = generate_filename(params_dict_run, data_dir)
                
                run_type_msg = "Main" if is_main_run else f"{method_key}_SeedRun{seed_iter_idx+1}"
                print(f"    {run_type_msg} (seed {current_run_seed_val}). File: {os.path.basename(filepath_run)}")
                
                current_run_data = None
                if not force_rerun: current_run_data = load_simulation_data(filepath_run)

                if current_run_data is None:
                    print(f"      Running simulation for {run_type_msg.lower()} (seed {current_run_seed_val})...")
                    run_start_time_iter = time.time()
                    if method_key == "TW":
                        current_run_data = sim_function(n_val, num_traj_main, kappa, current_tspan_sim, 
                                                        current_seed=current_run_seed_val, t_eval_hint=t_eval_physical)
                    elif method_key == "PP":
                        current_run_data = sim_function(n_val, num_traj_main, kappa, current_tspan_sim, 
                                                        current_physical_dt, num_workers=num_workers, 
                                                        current_seed=current_run_seed_val)
                    
                    if current_run_data and 'time_physical' in current_run_data:
                        if not np.allclose(current_run_data['time_physical'], t_eval_physical):
                            print("      Warning: Sim output time points mismatch desired t_eval.")
                        current_run_data['time_physical'] = t_eval_physical 
                    save_simulation_data(filepath_run, current_run_data)
                    run_end_time_iter = time.time()
                    print(f"      Sim for {run_type_msg.lower()} done: {run_end_time_iter - run_start_time_iter:.2f}s")
                
                if current_run_data:
                    all_results_data[regime_key][method_key]['seed_runs_plot4'].append(current_run_data)
                    if is_main_run:
                        all_results_data[regime_key][method_key]['main'] = current_run_data
            
            if 'main' not in all_results_data[regime_key][method_key]:
                if all_results_data[regime_key][method_key]['seed_runs_plot4']:
                    all_results_data[regime_key][method_key]['main'] = all_results_data[regime_key][method_key]['seed_runs_plot4'][0]
                else:
                    print(f"Error: No simulation data for {method_key} in {regime_key} for 'main'.")
                    all_results_data[regime_key][method_key]['main'] = None # Prevent KeyError
    print("--- Original Regime Simulations Finished ---")

    # --- N-Scan simulations for Max Squeezing Plot ---
    if sim_params.get('n_scan_active', False):
        print("\n--- Starting N-Scan Simulations for Max Squeezing Plot ---")
        n_scan_values_conf = sim_params.get('n_scan_values', [])
        n_scan_seeds_conf = sim_params.get('n_scan_seeds', [])
        n_scan_t_ends_map_conf = sim_params.get('n_scan_t_ends', {})
        
        if not (n_scan_values_conf and n_scan_seeds_conf and n_scan_t_ends_map_conf):
            print("   N-Scan configuration missing or incomplete. Skipping N-Scan.")
        else:
            if 'n_scan_results' not in all_results_data:
                all_results_data['n_scan_results'] = {}

            for n_val_scan in n_scan_values_conf:
                if n_val_scan not in all_results_data['n_scan_results']:
                     all_results_data['n_scan_results'][n_val_scan] = [None] * len(n_scan_seeds_conf)

                current_physical_t_end_scan = n_scan_t_ends_map_conf.get(n_val_scan)
                if current_physical_t_end_scan is None:
                    print(f"    Skipping N-Scan for n={n_val_scan}: t_end not defined in n_scan_t_ends_map_conf.")
                    continue
                
                current_tspan_sim_scan = (0, current_physical_t_end_scan)
                t_eval_physical_scan = np.linspace(0, current_physical_t_end_scan, num_points)
                current_physical_dt_scan = current_physical_t_end_scan / (num_points - 1)

                print(f"  Processing N-Scan: n={n_val_scan}, t_end={current_physical_t_end_scan}")

                for seed_idx, current_run_seed_val_scan in enumerate(n_scan_seeds_conf):
                    print(f"    Seed {seed_idx+1}/{len(n_scan_seeds_conf)} (Global Seed: {current_run_seed_val_scan})")
                    params_dict_run_scan = {
                        'method': 'PP', 'regime': f'n_scan_{n_val_scan}', 'n': n_val_scan,
                        'kappa': kappa, 'num_traj': num_traj_main,
                        'physical_t_end': current_physical_t_end_scan, 'num_points': num_points,
                        'seed_run_type': f'seed_{current_run_seed_val_scan}', 
                        'global_seed': current_run_seed_val_scan
                    }
                    filepath_run_scan = generate_filename(params_dict_run_scan, data_dir)
                    current_run_data_scan = None
                    if not force_rerun: current_run_data_scan = load_simulation_data(filepath_run_scan)

                    if current_run_data_scan is None:
                        print(f"      Running +P simulation for n={n_val_scan}, seed={current_run_seed_val_scan}...")
                        run_start_time_iter = time.time()
                        current_run_data_scan = run_pplus_simulation_parallel(
                            n_val_scan, num_traj_main, kappa, current_tspan_sim_scan, 
                            current_physical_dt_scan, num_workers=num_workers, 
                            current_seed=current_run_seed_val_scan
                        )
                        if current_run_data_scan and 'time_physical' in current_run_data_scan:
                            if not np.allclose(current_run_data_scan['time_physical'], t_eval_physical_scan):
                                print("      Warning: N-scan sim output time points mismatch desired t_eval.")
                            current_run_data_scan['time_physical'] = t_eval_physical_scan
                        save_simulation_data(filepath_run_scan, current_run_data_scan)
                        run_end_time_iter = time.time()
                        print(f"      Sim for n-scan done: {run_end_time_iter - run_start_time_iter:.2f}s")
                    
                    if n_val_scan in all_results_data['n_scan_results']:
                        all_results_data['n_scan_results'][n_val_scan][seed_idx] = current_run_data_scan
                    if current_run_data_scan is None:
                        print(f"      Warning: Failed to load/run sim for n={n_val_scan}, seed={current_run_seed_val_scan}")
            print("--- N-Scan Simulations Finished ---")
    # --- End of N-Scan section ---

    overall_end_time = time.time()
    print(f"\nAll simulations/data loading finished in {overall_end_time - overall_start_time:.2f} seconds.")
    print("-" * 30)
    print("Generating plots...")

    if plot_conf.get('generate_plot1_squeezing', False):
        create_squeezing_plot(all_results_data, config, plot_dir)
    if plot_conf.get('generate_plot2_pump_depletion', False):
        create_pump_depletion_plot(all_results_data, config, plot_dir)
    if plot_conf.get('generate_plot3_photon_tracker', False):
        create_photon_tracker_plot(all_results_data, config, plot_dir)
    if plot_conf.get('generate_plot4_seed_variance', False):
        create_seed_variance_plot(all_results_data, config, plot_dir)
    
    if plot_conf.get('generate_plot_max_squeezing_vs_n', False):
        if 'n_scan_results' in all_results_data and \
           any(seed_list for seed_list in all_results_data.get('n_scan_results', {}).values() if seed_list and any(s is not None for s in seed_list) ):
            create_max_squeezing_vs_n_plot(all_results_data['n_scan_results'], config, plot_dir)
        else:
            print("   Skipping Max Squeezing vs N plot: No 'n_scan_results' data found or it's empty/all None.")

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