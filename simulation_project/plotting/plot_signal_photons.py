# File: simulation_project/plotting/plot_signal_photons.py

import numpy as np
import matplotlib.pyplot as plt
import os

def create_signal_photons_plot(all_results, config, output_dir):
    """
    Signal Photon Number Proxy Comparison. Uses PHYSICAL TIME.
    X-axis limits per regime are taken from plot1_squeezing_regime_limits.
    """
    photon_regimes = config['photon_regimes']
    default_xlim_physical = config['plotting']['plot_xlim_physical']

    # Get the master regime-specific limits defined for Plot 1
    plot1_regime_master_limits = config['plotting'].get('plot1_squeezing_regime_limits', {})

    regime_keys = list(photon_regimes.keys())
    fig, axes = plt.subplots(1, len(regime_keys), 
                             figsize=(6 * len(regime_keys), 5), 
                             sharex=False, sharey=False)
    if len(regime_keys) == 1: axes = [axes]

    fig.suptitle(f'Signal Photon Number Proxy Comparison ($|\\langle \\alpha_1 \\rangle|^2$)', fontsize=16)

    for i, regime_key in enumerate(regime_keys):
        ax = axes[i]
        n_val = photon_regimes[regime_key]
        
        # Determine xlim for this subplot using Plot 1's master config
        regime_specific_master_limits = plot1_regime_master_limits.get(regime_key, {})
        current_xlim = regime_specific_master_limits.get('xlim', default_xlim_physical)

        data_tw = all_results[regime_key]['TW']['main']
        t_physical_tw = data_tw['time_physical']
        ax.plot(t_physical_tw, data_tw['n1_proxy_avg'], label='TW')
        
        data_pp = all_results[regime_key]['PP']['main']
        t_physical_pp = data_pp['time_physical']
        ax.plot(t_physical_pp, data_pp['n1_proxy_avg'], label='+P', linestyle='--')
        
        ax.set_title(f'{regime_key} Regime (n={n_val})')
        ax.set_xlabel(f'Time (physical units)')
        if i == 0:
            ax.set_ylabel('Proxy Signal Photon Number $|\\langle \\alpha_1 \\rangle|^2$')
        ax.legend()
        ax.grid(True)
        ax.set_xlim(current_xlim) # Apply the determined x-limit
        # ax.set_ylim(bottom=0) # Optional y-limit
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "plot_signal_photon_proxy.png"), dpi=300)
    print("   Plot (Signal Photon Proxy - Physical Time) generated.")
    plt.close(fig)