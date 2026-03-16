"""
Path and Directory Management Module

Handles path generation and directory creation for training runs.
"""

import os

def generate_save_directory(local_path, game_version, num_agents, map_nr, init_folder, 
                           synergy_scaling_factor, specialization_penalty_scale, agent_to_train=None):
    """
    Generate save directory path based on training configuration.
    
    Args:
        local_path: Base local path
        game_version: Game version string
        num_agents: Number of agents
        map_nr: Map identifier
        init_folder: Initialization folder name
        synergy_scaling_factor: Synergy scaling factor
        specialization_penalty_scale: Specialization penalty scale
        agent_to_train: Agent to train (for single agent)
        
    Returns:
        str: Complete save directory path
    """
    # Add synergy subfolder if reference reward is enabled
    synergy_folder = f"synergy_{synergy_scaling_factor:.2f}" if synergy_scaling_factor > 0 else "synergy_0"
    
    # Add specialization penalty subfolder based on lambda value
    if specialization_penalty_scale == 0:
        spec_folder = "specialized_0"
    else:
        lambda_str = f"{specialization_penalty_scale:.2f}"
        spec_folder = f"specialized_{lambda_str}"
    
    # Determine base directory structure
    if num_agents == 1:
        base_dir = f'{local_path}/data/samuel_lozano/cooked/pretraining/{game_version}/{init_folder}/map_{map_nr}'
    else:
        base_dir = f'{local_path}/data/samuel_lozano/cooked/{game_version}/{init_folder}/map_{map_nr}'
    
    # Add synergy and specialization folders
    save_dir = f'{base_dir}/{synergy_folder}/{spec_folder}'
    
    return save_dir


def get_map_grid_size(map_nr, maps_directory=None):
    """
    Determine grid size from map file.
    
    Args:
        map_nr: Map identifier
        maps_directory: Optional path to maps directory
        
    Returns:
        tuple: (cols, rows) grid size
    """
    if maps_directory is None:
        maps_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'spoiled_broth', 'maps', 'maps_txt')
    
    map_txt_path = os.path.join(maps_directory, f'{map_nr}.txt')
    if not os.path.exists(map_txt_path):
        raise FileNotFoundError(f"Map file {map_txt_path} not found.")
    
    with open(map_txt_path, 'r') as f:
        map_lines = [line.rstrip('\n') for line in f.readlines()]
    
    rows = len(map_lines)
    cols = len(map_lines[0]) if rows > 0 else 0
    
    if rows != cols:
        print(f"WARNING: Map is not square, this could cause errors in the future (got {rows} rows and {cols} columns).")
    
    return (cols, rows)
