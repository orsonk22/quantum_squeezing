# File: simulation_project/plotting/plot_photon_tracker.py
import numpy as np
import matplotlib.pyplot as plt
import os

def create_photon_tracker_plot(all_results, config, output_dir):
    """
    Plot (Photon Tracker): Photon Number Tracker. Uses PHYSICAL TIME.
    X-axis limit is taken from plot1_squeezing_regime_limits for the specified regime.
    Y-axis limit is taken from plot3_photon_tracker_ylim.
    Includes diagnostic prints.
    """
    print("--- Running create_photon_tracker_plot (Plot 3) ---")
    photon_regimes_map = config['photon_regimes']
    medium_regime_key = config['plotting'].get('medium_regime_key_for_plot3', 'Medium')
    default_xlim_physical = config['plotting']['plot_xlim_physical']
    
    default_ylim_photon_tracker = config['plotting'].get('plot_ylim_photon_tracker_default', [None, None])
    print(f"   PLOT 3: Default y-limit (plot_ylim_photon_tracker_default): {default_ylim_photon_tracker}") # DIAGNOSTIC

    plot1_regime_master_limits = config['plotting'].get('plot1_squeezing_regime_limits', {})

    if medium_regime_key not in photon_regimes_map:
        print(f"   PLOT 3: Medium regime key '{medium_regime_key}' not in photon_regimes. Skipping plot.")
        return
    if medium_regime_key not in all_results:
        print(f"   PLOT 3: No results for '{medium_regime_key}' in all_results. Skipping plot.")
        return
        
    n_val_medium = photon_regimes_map[medium_regime_key]
    
    regime_specific_master_limits = plot1_regime_master_limits.get(medium_regime_key, {})
    current_xlim = regime_specific_master_limits.get('xlim', default_xlim_physical)
    print(f"   PLOT 3: Using xlim {current_xlim} for regime '{medium_regime_key}'.")

    current_ylim = config['plotting'].get('plot3_photon_tracker_ylim', default_ylim_photon_tracker)
    print(f"   PLOT 3: Found plot3_photon_tracker_ylim: {config['plotting'].get('plot3_photon_tracker_ylim')}") # DIAGNOSTIC
    print(f"   PLOT 3: Using ylim {current_ylim} for regime '{medium_regime_key}'.") # DIAGNOSTIC

    if not (isinstance(current_ylim, list) and len(current_ylim) == 2):
        print(f"   PLOT 3 ({medium_regime_key}): Warning - current_ylim is not a list of two elements. Will autoscale Y. current_ylim: {current_ylim}")
        current_ylim = [None, None]

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(f'Photon Number Tracker - {medium_regime_key} Regime (n={n_val_medium}) - Physical Time', fontsize=16)

    # TW Data
    # ... (Previous TW data fetching and plotting logic, including your existing diagnostics) ...
    print(f"   PLOT 3: Checking TW data for regime '{medium_regime_key}'.")
    if 'TW' in all_results[medium_regime_key] and 'main' in all_results[medium_regime_key]['TW']:
        data_tw = all_results[medium_regime_key]['TW']['main']
        if 'time_physical' in data_tw and 'n1_actual_avg' in data_tw and 'n2_actual_avg' in data_tw:
            t_physical_tw = data_tw['time_physical']
            n1_tw = data_tw['n1_actual_avg']
            n2_tw = data_tw['n2_actual_avg']
            total_conserved_approx_tw = 2 * n2_tw + n1_tw
            
            print(f"   PLOT 3 (TW): Plotting. First 5 n1: {n1_tw[:5]}, n2: {n2_tw[:5]}")
            if np.any(np.isinf(n1_tw)) or np.any(np.isinf(n2_tw)): print("   PLOT 3 (TW): Inf values found in data!")
            if np.any(np.isnan(n1_tw)) or np.any(np.isnan(n2_tw)): print("   PLOT 3 (TW): NaN values found in data!")

            axes[0].plot(t_physical_tw, n1_tw, label=r'$\langle \hat{n}_1 \rangle_{TW}$')
            axes[0].plot(t_physical_tw, n2_tw, label=r'$\langle \hat{n}_2 \rangle_{TW}$')
            axes[0].plot(t_physical_tw, total_conserved_approx_tw, label=r'$2\langle \hat{n}_2 \rangle + \langle \hat{n}_1 \rangle_{TW}$', linestyle='--')
            axes[0].set_ylabel('Avg. Photon Number')
            axes[0].set_title('TW Method')
            axes[0].legend()
            axes[0].grid(True)
        # ... (else clauses for missing keys/data)
    # ...

    # +P Data
    # ... (Previous +P data fetching and plotting logic, including your existing diagnostics) ...
    print(f"   PLOT 3: Checking +P data for regime '{medium_regime_key}'.")
    if 'PP' in all_results[medium_regime_key] and 'main' in all_results[medium_regime_key]['PP']:
        data_pp = all_results[medium_regime_key]['PP']['main']
        if 'time_physical' in data_pp and 'n1_actual_avg' in data_pp and 'n2_actual_avg' in data_pp:
            t_physical_pp = data_pp['time_physical']
            n1_pp = data_pp['n1_actual_avg']
            n2_pp = data_pp['n2_actual_avg']
            total_conserved_approx_pp = 2 * n2_pp + n1_pp

            print(f"   PLOT 3 (+P): Plotting. First 5 n1: {n1_pp[:5]}, n2: {n2_pp[:5]}")
            if np.any(np.isinf(n1_pp)) or np.any(np.isinf(n2_pp)): print("   PLOT 3 (+P): Inf values found in data!")
            if np.any(np.isnan(n1_pp)) or np.any(np.isnan(n2_pp)): print("   PLOT 3 (+P): NaN values found in data!")

            axes[1].plot(t_physical_pp, n1_pp, label=r'$\langle \hat{n}_1 \rangle_{+P}$')
            axes[1].plot(t_physical_pp, n2_pp, label=r'$\langle \hat{n}_2 \rangle_{+P}$')
            axes[1].plot(t_physical_pp, total_conserved_approx_pp, label=r'$2\langle \hat{n}_2 \rangle + \langle \hat{n}_1 \rangle_{+P}$', linestyle='--')
            axes[1].set_ylabel('Avg. Photon Number')
            axes[1].set_title('+P Method')
            axes[1].legend()
            axes[1].grid(True)
        # ... (else clauses for missing keys/data)
    # ...


    axes[1].set_xlabel(f'Time (physical units)')
    axes[0].set_xlim(current_xlim) 
    axes[1].set_xlim(current_xlim)
    
    if current_ylim[0] is not None or current_ylim[1] is not None:
        axes[0].set_ylim(current_ylim)
        axes[1].set_ylim(current_ylim)
        print(f"   PLOT 3 ({medium_regime_key}): Applied ylim {current_ylim} to both subplots. Actual ylim0: {axes[0].get_ylim()}, ylim1: {axes[1].get_ylim()}")
    else:
        print(f"   PLOT 3 ({medium_regime_key}): Autoscaling Y-axes for both subplots.")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_path = os.path.join(output_dir, "plot_photon_tracker.png")
    
    try:
        plt.savefig(fig_path, dpi=300)
        print(f"   PLOT 3: Successfully saved plot to {fig_path}")
    except Exception as e:
        print(f"   PLOT 3: ERROR saving plot to {fig_path} - {e}")
        
    plt.close(fig)
    print("--- Finished create_photon_tracker_plot (Plot 3) ---")