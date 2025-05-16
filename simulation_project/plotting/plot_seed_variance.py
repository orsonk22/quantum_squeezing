# File: simulation_project/plotting/plot_seed_variance.py
import numpy as np
import matplotlib.pyplot as plt
import os

def create_seed_variance_plot(all_results, config, output_dir):
    """
    Plot 4: Signal X2 Variance (in dB) for Different Global Seeds. Uses PHYSICAL TIME.
    X-axis and Y-axis limits per regime are taken from plot1_squeezing_regime_limits.
    """
    photon_regimes = config['photon_regimes']
    global_seeds = config['simulation_params']['global_seeds_plot4']
    num_seed_runs_config = config['simulation_params']['num_seed_runs_plot4']
    num_available_seeds = len(global_seeds)
    num_seed_runs = min(num_seed_runs_config, num_available_seeds)

    if num_seed_runs_config > num_available_seeds:
        print(f"   Plot 4 Warning: Requested {num_seed_runs_config} seed runs, but only {num_available_seeds} unique global_seeds_plot4 are defined. Plotting for {num_available_seeds}.")

    default_xlim_physical = config['plotting']['plot_xlim_physical']
    default_ylim_squeezing = config['plotting']['plot_ylim_squeezing'] # Default y-limit
    
    vacuum_noise_var = config['plotting']['vacuum_noise_quad_variance']
    epsilon = config['plotting']['epsilon_db']

    # Get the master regime-specific limits defined for Plot 1 (used for both x and y here)
    plot1_regime_master_limits = config['plotting'].get('plot1_squeezing_regime_limits', {})

    regime_keys = list(photon_regimes.keys())
    methods = ["TW", "PP"]
    
    if not regime_keys or not methods:
        print("   No regimes or methods to plot for seed variance. Skipping Plot 4.")
        return
    if num_seed_runs == 0:
        print("   No seed runs to plot for Plot 4. Skipping Plot 4.")
        return

    fig, axes = plt.subplots(len(regime_keys), len(methods), 
                             figsize=(6 * len(methods), 5 * len(regime_keys)), 
                             sharex=False, 
                             sharey=False) # MODIFIED: sharey=False for individual y-limits per row
    
    if len(regime_keys) == 1 and len(methods) == 1: 
        axes = np.array([[axes]]) 
    elif len(regime_keys) == 1: 
        axes = np.array([axes])
    elif len(methods) == 1: 
        axes = np.array([[ax] for ax in axes])

    fig.suptitle(f'Signal $X_2$ Variance (dB) for Different Global Seeds - Physical Time', fontsize=16)
    
    colors_plot4_map = plt.cm.viridis(np.linspace(0, 0.85, max(1, num_seed_runs)))
    figure_legend_handles = {} 

    for i, regime_key in enumerate(regime_keys):
        regime_specific_master_limits = plot1_regime_master_limits.get(regime_key, {})
        current_xlim_for_regime = regime_specific_master_limits.get('xlim', default_xlim_physical)
        current_ylim_for_regime = regime_specific_master_limits.get('ylim', default_ylim_squeezing) # Get Y-LIMIT HERE

        for j, method_key in enumerate(methods):
            ax = axes[i, j]
            n_val = photon_regimes[regime_key]
            
            ax.set_title(f'{regime_key} (n={n_val}) - {method_key}')
            has_data_for_subplot = False

            if regime_key in all_results and \
               method_key in all_results[regime_key] and \
               'seed_runs_plot4' in all_results[regime_key][method_key] and \
               all_results[regime_key][method_key]['seed_runs_plot4']:

                seed_runs_data_list = all_results[regime_key][method_key]['seed_runs_plot4']
                actual_runs_to_plot = min(num_seed_runs, len(seed_runs_data_list))

                for seed_idx_in_plot in range(actual_runs_to_plot):
                    global_seeds_index = seed_idx_in_plot 
                    current_seed_data = seed_runs_data_list[seed_idx_in_plot]

                    if 'time_physical' in current_seed_data and 'X2_signal_var' in current_seed_data:
                        has_data_for_subplot = True
                        t_physical_seed = current_seed_data['time_physical']
                        X2_var_db_seed = 10 * np.log10(np.maximum(current_seed_data['X2_signal_var'], epsilon) / vacuum_noise_var)
                        
                        label_str = f'Seed {global_seeds[global_seeds_index]}'
                        
                        line, = ax.plot(t_physical_seed, X2_var_db_seed, linestyle='-', 
                                        color=colors_plot4_map[seed_idx_in_plot], 
                                        alpha=0.7)
                        
                        if label_str not in figure_legend_handles:
                            figure_legend_handles[label_str] = plt.Line2D([0], [0], linestyle='-', 
                                                                          color=colors_plot4_map[seed_idx_in_plot], 
                                                                          label=label_str)
                    else:
                        print(f"    Skipping seed run {global_seeds[global_seeds_index]} for {method_key} in {regime_key} (Plot 4) due to missing keys.")
            else:
                print(f"    No seed run data for {method_key} in {regime_key} for Plot 4.")

            if not has_data_for_subplot:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=9)

            ax.axhline(y=0, color='k', linestyle=':', alpha=0.8) 
            if i == len(regime_keys) - 1: 
                ax.set_xlabel(f'Time (physical units)')
            if j == 0: 
                ax.set_ylabel('$V(X_2)$ (dB rel. to Vacuum)')
            ax.grid(True)
            ax.set_xlim(current_xlim_for_regime) 
            ax.set_ylim(current_ylim_for_regime) # APPLY REGIME-SPECIFIC Y-LIMIT

    handles_for_legend = list(figure_legend_handles.values())
    if handles_for_legend: 
        vac_label = 'Vacuum (0 dB)'
        if vac_label not in figure_legend_handles:
             figure_legend_handles[vac_label] = plt.Line2D([0], [0], color='k', linestyle=':', label=vac_label)
             handles_for_legend.append(figure_legend_handles[vac_label])
        
        fig.legend(handles=handles_for_legend, loc='upper center', 
                   ncol=min(len(handles_for_legend), max(1,num_seed_runs // 2 + num_seed_runs % 2 +1)), # Dynamic ncol
                   bbox_to_anchor=(0.5, 0.03),
                   fontsize='small')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    fig_path = os.path.join(output_dir, "plot4_x2_variance_seeds_by_method_regime.png")
    plt.savefig(fig_path, dpi=300)
    print(f"   Plot 4 (X2 Variance for Seeds - Physical Time) saved to {fig_path}")
    plt.close(fig)