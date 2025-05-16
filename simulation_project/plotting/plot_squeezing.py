# File: simulation_project/plotting/plot_squeezing.py

import numpy as np
import matplotlib.pyplot as plt
import os

def create_squeezing_plot(all_results, config, output_dir):
    """
    Plot 1: Signal Quadrature Variance (Squeezing) Comparison.
    Uses PHYSICAL TIME on x-axis.
    Allows for individual x and y axis limits per photon regime via config.
    """
    photon_regimes_config = config['photon_regimes'] # From main config
    
    # Default/fallback limits are now for PHYSICAL time
    common_xlim_physical = config['plotting']['plot_xlim_physical']
    common_ylim = config['plotting']['plot_ylim_squeezing'] # Y-limits remain the same concept
    
    vacuum_noise_var = config['plotting']['vacuum_noise_quad_variance']
    epsilon = config['plotting']['epsilon_db']

    # Get the regime-specific limits configuration for Plot 1
    plot1_regime_limits_config = config['plotting'].get('plot1_squeezing_regime_limits', {})
    
    regime_keys = list(photon_regimes_config.keys())
    
    fig, axes = plt.subplots(1, len(regime_keys), 
                             figsize=(6 * len(regime_keys), 5), 
                             sharex=False, sharey=False) # Independent axes
    if len(regime_keys) == 1: axes = [axes]

    # Note: n_ref is no longer used for plot time scaling in this function's context
    # fig.suptitle(f'Signal Quadrature Variance (Squeezing) Comparison (Ref n={n_ref})', fontsize=16)
    # Consider a title without n_ref or adapt if n_ref has other meaning
    fig.suptitle(f'Signal Quadrature Variance (Squeezing) Comparison', fontsize=16)


    for i, regime_key in enumerate(regime_keys):
        ax = axes[i]
        n_val = photon_regimes_config[regime_key]
        # plot_time_scaling is REMOVED as we plot physical time directly

        # Get specific limits for this regime, or fall back to common limits
        specific_limits_for_regime = plot1_regime_limits_config.get(regime_key, {})
        current_xlim = specific_limits_for_regime.get('xlim', common_xlim_physical) # Use physical xlim
        current_ylim = specific_limits_for_regime.get('ylim', common_ylim)

        # TW data - use physical time directly
        data_tw = all_results[regime_key]['TW']['main']
        t_physical_tw = data_tw['time_physical'] # This is already physical time
        X1_var_tw_db = 10 * np.log10(np.maximum(data_tw['X1_signal_var'], epsilon) / vacuum_noise_var)
        X2_var_tw_db = 10 * np.log10(np.maximum(data_tw['X2_signal_var'], epsilon) / vacuum_noise_var)
        ax.plot(t_physical_tw, X1_var_tw_db, label='$V(X_1)$ TW', linestyle='-')
        ax.plot(t_physical_tw, X2_var_tw_db, label='$V(X_2)$ TW', linestyle='--')

        # +P data - use physical time directly
        data_pp = all_results[regime_key]['PP']['main']
        t_physical_pp = data_pp['time_physical'] # This is already physical time
        X1_var_pp_db = 10 * np.log10(np.maximum(data_pp['X1_signal_var'], epsilon) / vacuum_noise_var)
        X2_var_pp_db = 10 * np.log10(np.maximum(data_pp['X2_signal_var'], epsilon) / vacuum_noise_var)
        ax.plot(t_physical_pp, X1_var_pp_db, label='$V(X_1)$ +P', linestyle='-.')
        ax.plot(t_physical_pp, X2_var_pp_db, label='$V(X_2)$ +P', linestyle=':')
        
        ax.axhline(y=0, color='k', linestyle=':', label='Vacuum (0 dB)')
        ax.set_title(f'{regime_key} Regime (n={n_val})')
        ax.set_xlabel(f'Time (physical units)') # Updated X-axis label
        if i == 0:
            ax.set_ylabel('Variance (dB rel. to Vacuum)')
        
        ax.legend(fontsize='small')
        ax.grid(True)
        
        ax.set_xlim(current_xlim) # These are now physical time limits
        ax.set_ylim(current_ylim)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "plot1_squeezing_comparison.png"), dpi=300)
    print("   Plot 1 (Squeezing Comparison - Physical Time) generated.")
    plt.close(fig)