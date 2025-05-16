# File: simulation_project/plotting/plot_pump_depletion.py

import numpy as np
import matplotlib.pyplot as plt
import os

def create_pump_depletion_plot(all_results, config, output_dir):
    """
    Plot 2: Pump Photon Number Proxy Comparison. Uses PHYSICAL TIME.
    X-axis limits per regime are taken from plot1_squeezing_regime_limits.
    Y-axis limits per regime are taken from plot2_pump_depletion_regime_ylimits.
    """
    print("--- Running create_pump_depletion_plot (Plot 2) ---") # DIAGNOSTIC
    photon_regimes = config['photon_regimes']
    default_xlim_physical = config['plotting']['plot_xlim_physical']
    
    # Get default y-limit for this specific plot type
    default_ylim_pump_depletion = config['plotting'].get('plot_ylim_pump_depletion_default', [None, None]) 
    print(f"   PLOT 2: Default y-limit (plot_ylim_pump_depletion_default): {default_ylim_pump_depletion}") # DIAGNOSTIC

    plot1_regime_master_xlims = config['plotting'].get('plot1_squeezing_regime_limits', {})
    plot2_regime_ylims_config = config['plotting'].get('plot2_pump_depletion_regime_ylimits', {})
    print(f"   PLOT 2: Loaded plot2_pump_depletion_regime_ylimits: {plot2_regime_ylims_config}") # DIAGNOSTIC


    regime_keys = list(photon_regimes.keys())
    fig, axes = plt.subplots(1, len(regime_keys), 
                             figsize=(6 * len(regime_keys), 5), 
                             sharex=False, sharey=False) 
    if len(regime_keys) == 1: axes = [axes]

    fig.suptitle(f'Pump Photon Number Proxy Comparison ($|\\langle \\alpha_2 \\rangle|^2$)', fontsize=16)

    for i, regime_key in enumerate(regime_keys):
        ax = axes[i]
        n_val = photon_regimes[regime_key]
        print(f"   PLOT 2: Processing regime '{regime_key}'") # DIAGNOSTIC

        regime_specific_master_xlims = plot1_regime_master_xlims.get(regime_key, {})
        current_xlim = regime_specific_master_xlims.get('xlim', default_xlim_physical)
        print(f"     PLOT 2 ({regime_key}): Using xlim: {current_xlim}") # DIAGNOSTIC

        regime_specific_plot2_ylims = plot2_regime_ylims_config.get(regime_key, {})
        current_ylim = regime_specific_plot2_ylims.get('ylim', default_ylim_pump_depletion)
        print(f"     PLOT 2 ({regime_key}): Found specific ylim config: {regime_specific_plot2_ylims}") # DIAGNOSTIC
        print(f"     PLOT 2 ({regime_key}): Using ylim: {current_ylim}") # DIAGNOSTIC
        
        if not (isinstance(current_ylim, list) and len(current_ylim) == 2):
            print(f"     PLOT 2 ({regime_key}): Warning - current_ylim is not a list of two elements. Will autoscale Y. Current_ylim: {current_ylim}")
            current_ylim = [None, None]


        data_tw = all_results[regime_key]['TW']['main']
        t_physical_tw = data_tw['time_physical']
        ax.plot(t_physical_tw, data_tw['n2_proxy_avg'], label='TW')
        
        data_pp = all_results[regime_key]['PP']['main']
        t_physical_pp = data_pp['time_physical']
        ax.plot(t_physical_pp, data_pp['n2_proxy_avg'], label='+P', linestyle='--')
        
        # DIAGNOSTIC: Check data ranges if ylim seems off
        if data_tw is not None and 'n2_proxy_avg' in data_tw and len(data_tw['n2_proxy_avg']) > 0 :
            print(f"     PLOT 2 ({regime_key}, TW): n2_proxy_avg min: {np.nanmin(data_tw['n2_proxy_avg'])}, max: {np.nanmax(data_tw['n2_proxy_avg'])}")
        if data_pp is not None and 'n2_proxy_avg' in data_pp and len(data_pp['n2_proxy_avg']) > 0:
            print(f"     PLOT 2 ({regime_key}, +P): n2_proxy_avg min: {np.nanmin(data_pp['n2_proxy_avg'])}, max: {np.nanmax(data_pp['n2_proxy_avg'])}")


        ax.set_title(f'{regime_key} Regime (n={n_val})')
        ax.set_xlabel(f'Time (physical units)')
        if i == 0:
            ax.set_ylabel('Proxy Pump Photon Number $|\\langle \\alpha_2 \\rangle|^2$')
        ax.legend()
        ax.grid(True)
        ax.set_xlim(current_xlim)
        
        if current_ylim[0] is not None or current_ylim[1] is not None:
            ax.set_ylim(current_ylim)
            print(f"     PLOT 2 ({regime_key}): Applied ylim: {ax.get_ylim()}") # DIAGNOSTIC
        else:
            print(f"     PLOT 2 ({regime_key}): Autoscaling Y-axis.") # DIAGNOSTIC
        
        # Ensure y-axis starts at 0 if data is non-negative and current lower limit is None or <0
        if (current_ylim[0] is None or current_ylim[0] < 0):
            min_val_tw = np.nanmin(data_tw.get('n2_proxy_avg', [0])) if data_tw else 0
            min_val_pp = np.nanmin(data_pp.get('n2_proxy_avg', [0])) if data_pp else 0
            if min_val_tw >= 0 and min_val_pp >= 0:
                 ax.set_ylim(bottom=0) # Adjust bottom to 0
                 print(f"     PLOT 2 ({regime_key}): Adjusted ylim bottom to 0. New ylim: {ax.get_ylim()}") # DIAGNOSTIC


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "plot2_pump_depletion_proxy.png"), dpi=300)
    print("   Plot 2 (Pump Depletion Proxy - Physical Time) generated.")
    plt.close(fig)
    print("--- Finished create_pump_depletion_plot (Plot 2) ---") # DIAGNOSTIC