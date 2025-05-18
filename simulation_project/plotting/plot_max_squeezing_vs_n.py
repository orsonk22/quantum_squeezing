# File: simulation_project/plotting/plot_max_squeezing_vs_n.py

import numpy as np
import matplotlib.pyplot as plt
import os

def create_max_squeezing_vs_n_plot(n_scan_results_map, config, output_dir):
    """
    Plots the maximum squeezing (min V(X2) from +P) vs. initial pump photon number n.
    Uses a noise-based threshold to determine reliable data range.
    Also includes a diagnostic plot for the thresholding for one n_value.
    """
    print("--- Running create_max_squeezing_vs_n_plot ---")
    
    # n_scan_results_map is expected to be a dict: {n_val: [seed_run_data_1, ..., seed_run_data_10]}
    # Filter out n_values for which there's no data or all seed data is None
    n_values_with_any_data = sorted([
        n for n, seed_list in n_scan_results_map.items() if seed_list and any(s is not None for s in seed_list)
    ])

    if not n_values_with_any_data:
        print("   No valid n-scan results found to plot. Skipping.")
        return

    max_squeezing_db_list = []
    valid_n_for_plot = [] # n_values for which max squeezing could be determined

    reliability_std_threshold = config['plotting'].get('n_scan_reliability_std_threshold_linear', 0.05)
    vacuum_noise_var = config['plotting']['vacuum_noise_quad_variance']
    epsilon = config['plotting']['epsilon_db']
    num_simulation_points = config['simulation_params']['num_simulation_points']


    # --- Diagnostic plot setup ---
    # Pick a middle n_value that has data, for diagnostics
    n_for_diagnostic = None
    if n_values_with_any_data:
        n_for_diagnostic = n_values_with_any_data[len(n_values_with_any_data) // 2]
    
    diagnostic_data_collected_for_plot = False
    diag_time_axis, diag_mean_x2_db, diag_std_x2_lin, diag_cutoff_time_value = None, None, None, None
    
    for n_val in n_values_with_any_data:
        seed_data_list_raw = n_scan_results_map[n_val]
        # Filter out None entries from seed_data_list_raw
        seed_data_list = [run for run in seed_data_list_raw if run is not None and 
                          isinstance(run, dict) and 'X2_signal_var' in run and 
                          'time_physical' in run and len(run['time_physical']) == num_simulation_points]
        
        if not seed_data_list:
            print(f"   PlotMaxSq: Skipping n={n_val} due to no valid seed data after filtering.")
            continue

        # Assume all valid seed runs for a given n have the same time points defined by num_simulation_points
        # Use the time_physical from the first valid seed run
        time_physical = seed_data_list[0]['time_physical']
        
        x2_vars_linear_all_seeds = []
        for run_data in seed_data_list:
            # Basic check for consistent time array length (more robust checks could be added if needed)
            if len(run_data['time_physical']) == len(time_physical) and \
               np.allclose(run_data['time_physical'], time_physical): # Check values if lengths match
                x2_vars_linear_all_seeds.append(run_data['X2_signal_var'])
            else:
                print(f"   PlotMaxSq: Warning - Time array mismatch for n={n_val}, seed data for {run_data.get('global_seed', 'UnknownSeed')}. Skipping this seed.")

        if not x2_vars_linear_all_seeds: # If no seeds passed the time check
            print(f"   PlotMaxSq: No valid X2_signal_var data collected for n={n_val} after time alignment. Skipping.")
            continue
            
        x2_vars_linear_np = np.array(x2_vars_linear_all_seeds)
        
        mean_x2_lin_n_val = np.mean(x2_vars_linear_np, axis=0)
        std_x2_lin_n_val = np.std(x2_vars_linear_np, axis=0)

        cutoff_idx = len(time_physical) # Default to full range (index after last element)
        for idx, std_val in enumerate(std_x2_lin_n_val):
            if std_val > reliability_std_threshold:
                cutoff_idx = idx # Cutoff is at the first index where threshold is breached
                print(f"   PlotMaxSq: n={n_val}: Reliability threshold {reliability_std_threshold:.3f} breached at t={time_physical[idx]:.2f} (index {idx}). Std dev: {std_val:.4f}")
                break
        
        if cutoff_idx == 0:
            print(f"   PlotMaxSq: n={n_val}: Reliability threshold breached at t=0. No reliable squeezing data.")
            max_squeezing_db_list.append(np.nan) # Or handle as per desired output for unreliable n
            valid_n_for_plot.append(n_val)
            continue

        reliable_mean_x2_lin_n_val = mean_x2_lin_n_val[:cutoff_idx] # Data up to, but not including, cutoff_idx
        
        if len(reliable_mean_x2_lin_n_val) == 0: # Should not happen if cutoff_idx > 0
            print(f"   PlotMaxSq: n={n_val}: No reliable data points before cutoff index {cutoff_idx}. Skipping.")
            max_squeezing_db_list.append(np.nan)
            valid_n_for_plot.append(n_val)
            continue

        min_reliable_x2_lin_n_val = np.min(reliable_mean_x2_lin_n_val)
        current_max_squeezing_db = 10 * np.log10(np.maximum(min_reliable_x2_lin_n_val, epsilon) / vacuum_noise_var)
        
        max_squeezing_db_list.append(current_max_squeezing_db)
        valid_n_for_plot.append(n_val)
        print(f"   PlotMaxSq: n={n_val}: Max squeezing in reliable range (up to t={time_physical[cutoff_idx-1]:.2f} if cutoff_idx > 0 else 'start') = {current_max_squeezing_db:.2f} dB")

        if n_val == n_for_diagnostic and not diagnostic_data_collected_for_plot:
            diag_time_axis = time_physical
            diag_mean_x2_db = 10 * np.log10(np.maximum(mean_x2_lin_n_val, epsilon) / vacuum_noise_var)
            diag_std_x2_lin = std_x2_lin_n_val
            diag_cutoff_time_value = time_physical[cutoff_idx -1] if cutoff_idx > 0 else diag_time_axis[0] # Time of last reliable point
            if cutoff_idx == len(time_physical): # If all reliable
                 diag_cutoff_time_value = diag_time_axis[-1]
            diagnostic_data_collected_for_plot = True

    # --- Main Plot: Max Squeezing vs. n ---
    if valid_n_for_plot:
        fig_main, ax_main = plt.subplots(figsize=(10, 6))
        # Filter out NaNs for plotting if any n_val was deemed fully unreliable
        plot_n = [n for i, n in enumerate(valid_n_for_plot) if not np.isnan(max_squeezing_db_list[i])]
        plot_sq = [sq for sq in max_squeezing_db_list if not np.isnan(sq)]

        if plot_n: # Only plot if there's something to plot
            ax_main.plot(plot_n, plot_sq, marker='o', linestyle='-')
            ax_main.set_xlabel('Initial Pump Photon Number (n)')
            ax_main.set_ylabel('Maximum $V(X_2)$ Squeezing (+P) (dB)')
            ax_main.set_title('Maximum Squeezing vs. Pump Photon Number (+P Method)')
            ax_main.grid(True)
            
            plot_filename_main = os.path.join(output_dir, "plot_max_squeezing_vs_n.png")
            plt.savefig(plot_filename_main, dpi=300)
            print(f"   Plot (Max Squeezing vs n) saved to {plot_filename_main}")
        else:
            print("   PlotMaxSq: No data points to plot for Max Squeezing vs n after reliability checks.")
        plt.close(fig_main)
    else:
        print("   PlotMaxSq: No valid n values for plotting Max Squeezing vs n.")


    # --- Diagnostic Plot for Thresholding (for one n-value) ---
    if diagnostic_data_collected_for_plot:
        fig_diag, ax1_diag = plt.subplots(figsize=(12, 7))
        color1 = 'tab:blue'
        ax1_diag.set_xlabel(f'Time (physical units) for n={n_for_diagnostic}')
        ax1_diag.set_ylabel('Mean $V(X_2)$ (+P) (dB)', color=color1)
        ax1_diag.plot(diag_time_axis, diag_mean_x2_db, color=color1, label='Mean $V(X_2)$ (dB)')
        ax1_diag.tick_params(axis='y', labelcolor=color1)
        ax1_diag.grid(True, linestyle=':', alpha=0.7, axis='x') # Grid for x-axis from ax1

        ax2_diag = ax1_diag.twinx()
        color2 = 'tab:red'
        ax2_diag.set_ylabel('Std. Dev. of $V(X_2)$ (linear scale)', color=color2) # Clarify linear scale
        ax2_diag.plot(diag_time_axis, diag_std_x2_lin, color=color2, linestyle='--', label=f'Std. Dev. $V(X_2)$ (linear)')
        ax2_diag.tick_params(axis='y', labelcolor=color2)
        
        ax2_diag.axhline(y=reliability_std_threshold, color='k', linestyle=':', linewidth=1.5, label=f'Std. Dev. Threshold ({reliability_std_threshold:.3f})')
        ax1_diag.axvline(x=diag_cutoff_time_value, color='dimgray', linestyle='-.', linewidth=1.5, label=f'Est. Reliable Time Limit ({diag_cutoff_time_value:.2f})')

        fig_diag.suptitle(f'Diagnostic: $V(X_2)$ Mean & Std. Dev. for n={n_for_diagnostic} (+P Method)')
        
        # Combined legend
        lines1, labels1 = ax1_diag.get_legend_handles_labels()
        lines2, labels2 = ax2_diag.get_legend_handles_labels()
        ax1_diag.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=True)

        plt.tight_layout(rect=[0, 0.05, 1, 0.93]) # Adjust for suptitle and legend
        plot_filename_diag = os.path.join(output_dir, f"plot_diagnostic_threshold_n{n_for_diagnostic}.png")
        plt.savefig(plot_filename_diag, dpi=300)
        print(f"   Diagnostic plot for thresholding (n={n_for_diagnostic}) saved to {plot_filename_diag}")
        plt.close(fig_diag)
    elif n_for_diagnostic is not None: # If an n_for_diagnostic was chosen but no data collected for it specifically
        print(f"   PlotMaxSq: Diagnostic data for n={n_for_diagnostic} could not be collected/plotted (it might have been skipped or had issues).")
    
    print("--- Finished create_max_squeezing_vs_n_plot ---")