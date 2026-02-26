"""
Cooperation Factor Calculation Module

Fast lookup module that loads pre-computed cooperation factors from JSON cache.
For computation logic, see spoiled_broth/maps/compute_cooperation_factor.py
"""

import os
import json


def get_cooperation_factor(map_nr, maps_directory=None):
    """
    Get cooperation necessity factor for a given map from pre-computed cache.
    
    Args:
        map_nr: Map identifier
        maps_directory: Optional path to maps directory
        
    Returns:
        float: Cooperation factor (0.2-2.0) where higher values indicate more cooperation necessity
    """
    # Default maps directory
    if maps_directory is None:
        maps_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'spoiled_broth', 'maps')
    
    # Load from pre-computed cache
    lookup_file = os.path.join(maps_directory, 'cooperation_factors.json')
    
    if not os.path.exists(lookup_file):
        print(f"Warning: Cooperation factors cache not found at {lookup_file}")
        print("Please run: python spoiled_broth/maps/compute_cooperation_factor.py")
        return 1.0  # Default fallback
    
    try:
        with open(lookup_file, 'r') as f:
            cooperation_factors = json.load(f)
        
        if map_nr in cooperation_factors:
            factor = cooperation_factors[map_nr]
            print(f"Loaded cooperation factor for {map_nr}: {factor:.3f}")
            return factor
        else:
            print(f"Warning: Map {map_nr} not found in cooperation factors cache")
            print(f"Available maps: {sorted(cooperation_factors.keys())}")
            return 1.0  # Default fallback
            
    except Exception as e:
        print(f"Error loading cooperation factors cache: {e}")
        return 1.0  # Default fallback


def calculate_cooperation_factor(map_nr, maps_directory=None):
    """
    Legacy function for compatibility. Redirects to get_cooperation_factor.
    
    Args:
        map_nr: Map identifier  
        maps_directory: Optional path to maps directory
        
    Returns:
        float: Cooperation factor (0.2-2.0)
    """
    return get_cooperation_factor(map_nr, maps_directory)


def update_cooperation_cache(maps_directory=None):
    """
    Utility function to regenerate the cooperation factors cache.
    Call this when maps are added/modified or calculation method changes.
    
    Args:
        maps_directory: Optional path to maps directory
    """
    if maps_directory is None:
        maps_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'spoiled_broth', 'maps')
    
    # Run the computation script
    import subprocess
    import sys
    
    compute_script = os.path.join(maps_directory, 'compute_cooperation_factor.py')
    if os.path.exists(compute_script):
        print(f"Running cooperation factor computation...")
        result = subprocess.run([sys.executable, compute_script], 
                              cwd=maps_directory, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Cooperation factors cache updated successfully")
            print(result.stdout)
        else:
            print("✗ Error updating cooperation factors cache:")
            print(result.stderr)
    else:
        print(f"Computation script not found at {compute_script}")
        print(f"Please ensure compute_cooperation_factor.py exists in {maps_directory}")
