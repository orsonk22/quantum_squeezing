# File: simulation_project/plotting/plot_squeezing.py

import numpy as np
import matplotlib.pyplot as plt
import os

def create_squeezing_plot(all_results, config, output_dir):
    """
    Plot 1: Signal Quadrature Variance (Squeezing) Comparison.
    Uses PHYSICAL TIME on x-axis.
    Allows for individual x and y axis limits per photon regime via config.
    MODIFIED to include shaded error for +P V(X2) based on seed_runs_plot4.
    """
    photon_regimes_config = config['photon_regimes']
    
    common_xlim_physical = config['plotting']['plot_xlim_physical']
    common_ylim = config['plotting']['plot_ylim_squeezing']
    
    vacuum_noise_var = config['plotting']['vacuum_noise_quad_variance']
    epsilon = config['plotting']['epsilon_db']

    plot1_regime_limits_config = config['plotting'].get('plot1_squeezing_regime_limits', {})
    
    regime_keys = list(photon_regimes_config.keys())
    
    fig, axes = plt.subplots(len(regime_keys), 1, 
                             figsize=(8, 5 * len(regime_keys)), 
                             sharex=False, sharey=False)
    if len(regime_keys) == 1: axes = [axes]

    fig.suptitle(f'Signal Quadrature Variance (Squeezing) Comparison', fontsize=16, y=0.99)

    for i, regime_key in enumerate(regime_keys):
        ax = axes[i]
        n_val = photon_regimes_config[regime_key]
        
        specific_limits_for_regime = plot1_regime_limits_config.get(regime_key, {})
        current_xlim = specific_limits_for_regime.get('xlim', common_xlim_physical)
        current_ylim = specific_limits_for_regime.get('ylim', common_ylim)

        # TW data - uses 'main' run
        # Ensure 'main' data exists and has the required keys
        tw_main_data = all_results.get(regime_key, {}).get('TW', {}).get('main')
        if tw_main_data and 'time_physical' in tw_main_data and \
           'X1_signal_var' in tw_main_data and 'X2_signal_var' in tw_main_data:
            t_physical_tw = tw_main_data['time_physical']
            X1_var_tw_db = 10 * np.log10(np.maximum(tw_main_data['X1_signal_var'], epsilon) / vacuum_noise_var)
            X2_var_tw_db = 10 * np.log10(np.maximum(tw_main_data['X2_signal_var'], epsilon) / vacuum_noise_var)
            ax.plot(t_physical_tw, X1_var_tw_db, label='$V(X_1)$ TW', linestyle='-')
            ax.plot(t_physical_tw, X2_var_tw_db, label='$V(X_2)$ TW', linestyle='--')
        else:
            print(f"Plot1: Warning - Missing TW 'main' data or required keys for regime {regime_key}")

        # +P data
        pp_data_for_regime = all_results.get(regime_key, {}).get('PP', {})
        pp_main_data = pp_data_for_regime.get('main')
        pp_seed_runs = pp_data_for_regime.get('seed_runs_plot4', [])
        
        t_physical_pp_fallback = None
        if pp_main_data and 'time_physical' in pp_main_data:
            t_physical_pp_fallback = pp_main_data['time_physical']

        # For X1_signal_var (+P), plot from main data
        if pp_main_data and 'X1_signal_var' in pp_main_data and t_physical_pp_fallback is not None:
            X1_var_pp_db_main = 10 * np.log10(np.maximum(pp_main_data['X1_signal_var'], epsilon) / vacuum_noise_var)
            ax.plot(t_physical_pp_fallback, X1_var_pp_db_main, label='$V(X_1)$ +P', linestyle='-.')
        else:
             print(f"Plot1: Warning - Missing +P 'main' data or 'X1_signal_var' for regime {regime_key}")

        # For X2_signal_var (+P), use seed runs to calculate mean and std dev
        # Filter out None entries from seed runs if any failed
        valid_pp_seed_runs = [run for run in pp_seed_runs if run is not None and 
                              isinstance(run, dict) and 'X2_signal_var' in run and 
                              'time_physical' in run]

        if valid_pp_seed_runs:
            # Assuming all valid seed runs have the same time points, use the first one's
            t_physical_pp_seeds = valid_pp_seed_runs[0]['time_physical']
            
            x2_vars_linear_seeds = []
            for seed_run_data in valid_pp_seed_runs:
                if np.array_equal(seed_run_data['time_physical'], t_physical_pp_seeds):
                    x2_vars_linear_seeds.append(seed_run_data['X2_signal_var'])
                else:
                    print(f"Plot1: Warning - time_physical mismatch in +P seed runs for regime {regime_key}. Skipping a seed run.")

            if x2_vars_linear_seeds:
                x2_vars_linear_seeds_np = np.array(x2_vars_linear_seeds)
                
                mean_x2_var_linear = np.mean(x2_vars_linear_seeds_np, axis=0)
                std_x2_var_linear = np.std(x2_vars_linear_seeds_np, axis=0)
                
                mean_x2_var_db = 10 * np.log10(np.maximum(mean_x2_var_linear, epsilon) / vacuum_noise_var)
                line_pp_x2, = ax.plot(t_physical_pp_seeds, mean_x2_var_db, label='$V(X_2)$ +P (mean)', linestyle=':')
                
                lower_bound_linear = mean_x2_var_linear - std_x2_var_linear
                upper_bound_linear = mean_x2_var_linear + std_x2_var_linear
                
                lower_bound_db = 10 * np.log10(np.maximum(lower_bound_linear, epsilon) / vacuum_noise_var)
                upper_bound_db = 10 * np.log10(np.maximum(upper_bound_linear, epsilon) / vacuum_noise_var)
                
                ax.fill_between(t_physical_pp_seeds, lower_bound_db, upper_bound_db, 
                                color=line_pp_x2.get_color(), alpha=0.2, label='Std. Dev. $V(X_2)$ +P')
            else: # Fallback if seed run data was problematic after filtering
                if pp_main_data and 'X2_signal_var' in pp_main_data and t_physical_pp_fallback is not None:
                    X2_var_pp_db_main = 10 * np.log10(np.maximum(pp_main_data['X2_signal_var'], epsilon) / vacuum_noise_var)
                    ax.plot(t_physical_pp_fallback, X2_var_pp_db_main, label='$V(X_2)$ +P (main)', linestyle=':')
                    print(f"Plot1: Info - Used main +P data for X2 in {regime_key} as seed data was insufficient.")
                else:
                    print(f"Plot1: Warning - Missing +P 'main' data for X2 for regime {regime_key} after seed run check.")
        
        elif pp_main_data and 'X2_signal_var' in pp_main_data and t_physical_pp_fallback is not None: # Fallback if no valid seed runs
            X2_var_pp_db_main = 10 * np.log10(np.maximum(pp_main_data['X2_signal_var'], epsilon) / vacuum_noise_var)
            ax.plot(t_physical_pp_fallback, X2_var_pp_db_main, label='$V(X_2)$ +P (main)', linestyle=':')
            print(f"Plot1: Info - Used main +P data for X2 in {regime_key} as no valid seed runs found.")
        else:
            print(f"Plot1: Warning - No +P data (main or seed) for X2 for regime {regime_key}")

        ax.axhline(y=0, color='k', linestyle=':', label='Vacuum (0 dB)')
        ax.set_title(f'{regime_key} Regime (n={n_val})')
        ax.set_xlabel(f'Time')
        if i == 0:
            ax.set_ylabel('Variance (dB rel. to Vacuum)')
        else:
            ax.set_ylabel('Variance (dB)')
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        legend = ax.legend(by_label.values(), by_label.keys(), fontsize='small')
        legend.set_frame_on(True)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('black')

        ax.grid(True)
        ax.set_xlim(current_xlim)
        ax.set_ylim(current_ylim)
        
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    plt.savefig(os.path.join(output_dir, "plot1_squeezing_comparison_with_error.png"), dpi=300)
    print("   Plot 1 (Squeezing Comparison - Physical Time with error) generated.")
    plt.close(fig)