# File: simulation_project/data_handling/storage.py

import numpy as np
import os
import hashlib
import json # For config hashing if needed, but string of sorted items is fine

def get_params_hash(params_dict):
    """Creates a hash from a dictionary of parameters for filename uniqueness."""
    # Convert all values to string and sort items by key for consistent hash
    # Using json.dumps for a more robust serialization than str(sorted(...))
    try:
        params_str = json.dumps(params_dict, sort_keys=True)
    except TypeError: # Fallback for non-serializable items like None, though config should avoid complex objects
        params_str = str(sorted(params_dict.items()))
    return hashlib.md5(params_str.encode('utf-8')).hexdigest()[:10]


def generate_filename(params_dict, base_dir="simulation_data_combined"):
    """Generates a consistent filename based on parameters."""
    hash_str = get_params_hash(params_dict)
    method = params_dict.get('method', 'unknown')
    regime = params_dict.get('regime', 'unknown')
    n_val = params_dict.get('n', 0)
    seed_type = params_dict.get('seed_run_type', 'main')
    global_seed = params_dict.get('global_seed', 0)
    
    descriptive_part = f"{method}_{regime}_n{n_val}_{seed_type}_seed{global_seed}"
    filename = f"data_{descriptive_part}_{hash_str}.npz"
    return os.path.join(base_dir, filename)


def save_simulation_data(filepath, data_dict):
    """Saves simulation data dictionary to an .npz file."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez_compressed(filepath, **data_dict)
        print(f"   Data saved to {filepath}")
    except Exception as e:
        print(f"   Error saving data to {filepath}: {e}")


def load_simulation_data(filepath):
    """Loads simulation data from an .npz file."""
    if os.path.exists(filepath):
        try:
            data = np.load(filepath, allow_pickle=True)
            print(f"   Data loaded from {filepath}")
            return {key: data[key] for key in data} # Convert NpzFile to a standard dictionary
        except Exception as e:
            print(f"   Error loading data from {filepath}: {e}. Will recompute.")
            return None
    return None