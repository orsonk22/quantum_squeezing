import numpy as np
import matplotlib.pyplot as plt
import os

def create_max_squeezing_vs_n_plot(n_scan_results_map, config, output_dir):
    print("--- Running create_max_squeezing_vs_n_plot (SEM + Clean Plot) ---")

    n_values_with_any_data = sorted([
        n for n, seed_list in n_scan_results_map.items() if seed_list and any(s is not None for s in seed_list)
    ])

    if not n_values_with_any_data:
        print("   PlotMaxSq: No valid n-scan results found to plot. Skipping.")
        return

    max_squeezing_db_list = []
    squeezing_db_lower_error = []
    squeezing_db_upper_error = []
    valid_n_for_plot = []

    reliability_std_threshold = config['plotting'].get('n_scan_reliability_std_threshold_linear', 0.035)
    vacuum_noise_var = config['plotting']['vacuum_noise_quad_variance']
    epsilon = config['plotting']['epsilon_db']
    num_simulation_points = config['simulation_params']['num_simulation_points']

    for n_val in n_values_with_any_data:
        seed_data_list_raw = n_scan_results_map[n_val]
        seed_data_list = [run for run in seed_data_list_raw if run is not None and 
                          isinstance(run, dict) and 'X2_signal_var' in run and 
                          'time_physical' in run and len(run['time_physical']) == num_simulation_points]

        if not seed_data_list:
            max_squeezing_db_list.append(np.nan)
            squeezing_db_lower_error.append(np.nan)
            squeezing_db_upper_error.append(np.nan)
            valid_n_for_plot.append(n_val)
            continue

        time_physical = seed_data_list[0]['time_physical']
        x2_vars_linear_all_seeds_np = np.array([
            run_data['X2_signal_var'] for run_data in seed_data_list 
            if len(run_data['time_physical']) == len(time_physical) and np.allclose(run_data['time_physical'], time_physical)
        ])

        if x2_vars_linear_all_seeds_np.ndim < 2 or x2_vars_linear_all_seeds_np.shape[0] < 1:
            max_squeezing_db_list.append(np.nan)
            squeezing_db_lower_error.append(np.nan)
            squeezing_db_upper_error.append(np.nan)
            valid_n_for_plot.append(n_val)
            continue

        mean_x2_lin_n_val = np.mean(x2_vars_linear_all_seeds_np, axis=0)
        std_x2_lin_n_val = np.std(x2_vars_linear_all_seeds_np, axis=0)

        reliable_indices = np.where(std_x2_lin_n_val <= reliability_std_threshold)[0]
        current_max_squeezing_db_for_n = np.nan
        current_lower_err_db = np.nan
        current_upper_err_db = np.nan

        if len(reliable_indices) > 0:
            reliable_mean_x2_values = mean_x2_lin_n_val[reliable_indices]

            if len(reliable_mean_x2_values) > 0:
                min_idx_in_reliable_array = np.argmin(reliable_mean_x2_values)
                mean_linear_at_min_time = reliable_mean_x2_values[min_idx_in_reliable_array]
                original_time_idx_of_min = reliable_indices[min_idx_in_reliable_array]
                time_of_min_for_n = time_physical[original_time_idx_of_min]

                current_max_squeezing_db_for_n = 10 * np.log10(np.maximum(mean_linear_at_min_time, epsilon) / vacuum_noise_var)

                values_at_optimal_time_from_all_seeds = x2_vars_linear_all_seeds_np[:, original_time_idx_of_min]
                sem_of_values_at_optimal_time = np.std(values_at_optimal_time_from_all_seeds, ddof=1) / np.sqrt(len(values_at_optimal_time_from_all_seeds))

                val_plus_sem_lin = mean_linear_at_min_time + sem_of_values_at_optimal_time
                val_minus_sem_lin = mean_linear_at_min_time - sem_of_values_at_optimal_time

                db_val_plus_sem = 10 * np.log10(np.maximum(val_plus_sem_lin, epsilon) / vacuum_noise_var)
                db_val_minus_sem = 10 * np.log10(np.maximum(val_minus_sem_lin, epsilon) / vacuum_noise_var)

                current_upper_err_db = db_val_plus_sem - current_max_squeezing_db_for_n
                current_lower_err_db = current_max_squeezing_db_for_n - db_val_minus_sem

                print(f"   PlotMaxSq: n={n_val}: Min reliable V(X2) = {current_max_squeezing_db_for_n:.2f} dB (+{current_upper_err_db:.2f}/-{current_lower_err_db:.2f} dB SEM) at t={time_of_min_for_n:.2f}")

        max_squeezing_db_list.append(current_max_squeezing_db_for_n)
        squeezing_db_lower_error.append(current_lower_err_db)
        squeezing_db_upper_error.append(current_upper_err_db)
        valid_n_for_plot.append(n_val)

    if valid_n_for_plot:
        fig_main, ax_main = plt.subplots(figsize=(10, 6))
        plot_n_indices = [i for i, sq_val in enumerate(max_squeezing_db_list) if not np.isnan(sq_val)]

        if plot_n_indices:
            plot_n = np.array(valid_n_for_plot)[plot_n_indices]
            plot_sq = np.array(max_squeezing_db_list)[plot_n_indices]
            plot_lower_err = np.maximum(np.array(squeezing_db_lower_error)[plot_n_indices], 0)
            plot_upper_err = np.maximum(np.array(squeezing_db_upper_error)[plot_n_indices], 0)
            errors = [plot_lower_err, plot_upper_err]

            ax_main.errorbar(plot_n, plot_sq, yerr=errors, fmt='o', capsize=5, label='Max Squeezing (+P) ± SEM')
            ax_main.set_xlabel('Initial Pump Photon Number (n)')
            ax_main.set_ylabel('Maximum $V(X_2)$ Squeezing (dB)')
            ax_main.set_title('Maximum Squeezing vs. Pump Photon Number (+P Method)')
            ax_main.set_ylim(bottom=-20)
            ax_main.grid(True)
            ax_main.legend()

            plot_filename_main = os.path.join(output_dir, "plot_max_squeezing_vs_n_errors.png")
            plt.savefig(plot_filename_main, dpi=300)
            print(f"   Plot (Max Squeezing vs n with SEM) saved to {plot_filename_main}")
            plt.close(fig_main)
        else:
            print("   PlotMaxSq: No data points to plot after filtering.")
    else:
        print("   PlotMaxSq: No valid n values for plotting.")

    print("--- Finished create_max_squeezing_vs_n_plot (SEM + Clean Plot) ---")
