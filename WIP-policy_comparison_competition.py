import numpy as np
import copy
import os
import matplotlib.pyplot as plt
from ray.rllib.algorithms.ppo import PPO
from spoiled_broth.game import SpoiledBroth
from spoiled_broth.maps.map_paths import get_map_txt_path
from pathlib import Path
from spoiled_broth.rl.game_env_competition import get_action_type, ACTION_TYPE_OWN_USEFUL_CUTTING_BOARD, ACTION_TYPE_OTHER_USEFUL_CUTTING_BOARD, ACTION_TYPE_OWN_USEFUL_DELIVERY, ACTION_TYPE_OTHER_USEFUL_DELIVERY
from spoiled_broth.game import game_to_obs_vector_competition
from spoiled_broth.game import SpoiledBroth
from spoiled_broth.rl.game_env_competition import init_game

EPOCH = 10000  # Epoch to load
DATETIME = "2025-08-06_18-59-00"  # Date of training run
MAP_NR = "simple_kitchen_competition"  # Map used in training
ATTITUDE_1 = "competitive"  # or "cooperative"
ATTITUDE_2 = "individualistic"  # or "competitive"
INTENT_VERSION = "v3.1"

def generate_key_observations(agent_id="ai_rl_1"):
    """
    Generates only observations where the agent holds a raw, cut, or salad item and can perform an own/other action (as defined in game_env_competition.py).
    For each, classifies the possible actions as own_action or other_action using get_action_type.
    """
    game = SpoiledBroth(map_nr=MAP_NR, grid_size=GRID_SIZE, intent_version=INTENT_VERSION)
    game.add_agent("ai_rl_1")
    game.add_agent("ai_rl_2")

    agent_food_types = {
        "ai_rl_1": "tomato",
        "ai_rl_2": "pumpkin"
    }

    if agent_id == "ai_rl_1":
        other_agent_id = "ai_rl_2"
    else:
        other_agent_id = "ai_rl_1"

    agent = game.gameObjects[agent_id]
    walkable_positions = []
    for x in range(game.grid.width):
        for y in range(game.grid.height):
            if (game.grid.tiles[x][y] and game.grid.tiles[x][y].is_walkable):
                walkable_positions.append((x, y))

    scenarios = [
        {"item": agent_food_types[agent_id], "label": "own_food"},
        {"item": agent_food_types[other_agent_id], "label": "other_food"},
        {"item": f"{agent_food_types[agent_id]}_cut", "label": "own_food_cut"},
        {"item": f"{agent_food_types[other_agent_id]}_cut", "label": "other_food_cut"},
        {"item": f"{agent_food_types[agent_id]}_salad", "label": "own_salad"},
        {"item": f"{agent_food_types[other_agent_id]}_salad", "label": "other_salad"},
    ]

    observations = []
    for scenario in scenarios:
        for pos in walkable_positions:
            # For each scenario and position, just create the agent state
            game_copy = copy.deepcopy(game)
            agent_copy = game_copy.gameObjects[agent_id]
            agent_copy.x = pos[0] * game_copy.grid.tile_size + game_copy.grid.tile_size // 2
            agent_copy.y = pos[1] * game_copy.grid.tile_size + game_copy.grid.tile_size // 2
            agent_copy.item = scenario["item"]
            obs = game_to_obs_vector_competition(game_copy, agent_id, agent_food_types)
            observations.append({
                "obs": obs,
                "item": scenario["label"],
                "holding": scenario["item"],
                "position": (pos[0], pos[1])
            })
    return observations

def load_policy(map_nr, attitude, date, epoch, game_version="COMPETITION", intent_version=None):
    """
    Load a policy from a specific training run and epoch following the training script structure
    Args:
        map_nr: The map number used in training
        attitude: 'cooperative' or 'competitive'
        date: Training date in format YYYY-MM-DD_HH-MM-SS
        epoch: Epoch number to load
        game_version: 'CLASSIC' or 'COMPETITION'
        intent_version: Optional intent version if used in training
    """
    # Construct path following the training script structure
    base_path = Path("/data/samuel_lozano/cooked")
    
    if game_version == "CLASSIC":
        base_path = base_path / "classic"
    elif game_version == "COMPETITION":
        base_path = base_path / "competition"
    
    if intent_version:
        base_path = base_path / intent_version
    
    model_path = (base_path / f"map_{map_nr}" / attitude.lower() / 
                 f"Training_{date}" / f"checkpoint_{epoch}")
    print(model_path)
    model = PPO.from_checkpoint(str(model_path))
    return lambda obs: model.predict(obs, deterministic=True)[0]

def visualize_policy_comparison(policy_results, title):
    """
    Create a bar plot comparing own vs other actions for different item states
    """
    categories = ['Raw Food', 'Cut Food', 'Salad']
    own_actions = []
    other_actions = []
    
    # Map of states to what we consider a "correct" action
    correct_action_map = {
        'own_food': lambda a: a in [3, 4],  # Interact with cutting board
        'other_food': lambda a: a in [3, 4],  # Interact with cutting board
        'own_food_cut': lambda a: a in [3, 4],  # Interact with counter/plate
        'other_food_cut': lambda a: a in [3, 4],  # Interact with counter/plate
        'own_salad': lambda a: a in [3, 4],  # Interact with delivery
        'other_salad': lambda a: a in [3, 4],  # Interact with delivery
    }
    
    # Count correct actions for each category
    category_map = {
        'Raw Food': ['own_food', 'other_food'],
        'Cut Food': ['own_food_cut', 'other_food_cut'],
        'Salad': ['own_salad', 'other_salad']
    }
    
    for cat in categories:
        own_key = category_map[cat][0]
        other_key = category_map[cat][1]
        
        # Count correct actions according to our mapping
        own_correct = sum(1 for action in policy_results[own_key] 
                         if correct_action_map[own_key](action['action']))
        other_correct = sum(1 for action in policy_results[other_key] 
                          if correct_action_map[other_key](action['action']))
        
        # Get total actions for this category
        own_total = len(policy_results[own_key]) or 1  # avoid division by zero
        other_total = len(policy_results[other_key]) or 1
        
        # Convert to percentages
        own_actions.append((own_correct / own_total) * 100)
        other_actions.append((other_correct / other_total) * 100)
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, own_actions, width, label='Own Items')
    ax.bar(x + width/2, other_actions, width, label='Other\'s Items')
    
    ax.set_ylabel('Percentage of Correct Actions')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    
    # Add percentage labels on top of each bar
    for i, v in enumerate(own_actions):
        ax.text(i - width/2, v + 1, f'{v:.1f}%', ha='center')
    for i, v in enumerate(other_actions):
        ax.text(i + width/2, v + 1, f'{v:.1f}%', ha='center')
    
    plt.savefig(f'policy_comparison_{title.replace(" ", "_")}.png')
    plt.close()

# Configuration
training_configs = [
    {
        "agent_id": "ai_rl_1",
        "map_nr": MAP_NR,
        "attitude": ATTITUDE_1.lower(),
        "date": DATETIME,  # Replace with actual training date
        "epoch": EPOCH,  # Replace with desired epoch
        "game_version": "COMPETITION",
        "intent_version": INTENT_VERSION,  # Set if you used an intent version
        "label": ATTITUDE_1  # Label for the plot
    },
    {
        "agent_id": "ai_rl_2",
        "map_nr": MAP_NR,
        "attitude": ATTITUDE_2.lower(),
        "date": DATETIME,  # Replace with actual training date
        "epoch": EPOCH,  # Replace with desired epoch
        "game_version": "COMPETITION",
        "intent_version": INTENT_VERSION,  # Set if you used an intent version
        "label": ATTITUDE_2  # Label for the plot
    }
]


# --- New Policy Evaluation and Action Classification ---
from spoiled_broth.rl.game_env_competition import get_action_type, ACTION_TYPE_OWN_USEFUL_CUTTING_BOARD, ACTION_TYPE_OTHER_USEFUL_CUTTING_BOARD, ACTION_TYPE_OWN_USEFUL_DELIVERY, ACTION_TYPE_OTHER_USEFUL_DELIVERY
from spoiled_broth.rl.game_env_competition import ACTION_TYPE_OWN_USEFUL_FOOD_DISPENSER, ACTION_TYPE_OTHER_USEFUL_FOOD_DISPENSER

def classify_action(action_type):
    if action_type == ACTION_TYPE_OWN_USEFUL_DELIVERY:
        return "deliver_own"
    elif action_type == ACTION_TYPE_OTHER_USEFUL_DELIVERY:
        return "deliver_other"
    elif action_type == ACTION_TYPE_OWN_USEFUL_CUTTING_BOARD:
        return "cut_own"
    elif action_type == ACTION_TYPE_OTHER_USEFUL_CUTTING_BOARD:
        return "cut_other"
    elif action_type == ACTION_TYPE_OWN_USEFUL_FOOD_DISPENSER:
        return "get_own_food"
    elif action_type == ACTION_TYPE_OTHER_USEFUL_FOOD_DISPENSER:
        return "get_other_food"
    else:
        return "other"

def decode_action_index_to_tile(game, action_idx, clickable_indices):
    # clickable_indices: list of indices (y * width + x)
    if action_idx >= len(clickable_indices):
        return None  # do nothing
    idx = clickable_indices[action_idx]
    width = game.grid.width
    x = idx % width
    y = idx // width
    return (x, y)

def evaluate_policy_on_observations(policy, observations, agent_id="ai_rl_1"):
    # Rebuild a game for decoding action indices
    game = SpoiledBroth(map_nr=MAP_NR)
    game.add_agent("ai_rl_1")
    game.add_agent("ai_rl_2")
    agent_food_types = {"ai_rl_1": "tomato", "ai_rl_2": "pumpkin"}
    # Get clickable indices for this map
    _, _, _, clickable_indices = init_game(["ai_rl_1", "ai_rl_2"], map_nr=MAP_NR, grid_size=GRID_SIZE, intent_version=INTENT_VERSION, seed=None)

    results = []
    for obs_dict in observations:
        obs = obs_dict["obs"]
        action_idx = policy(obs)
        tile = decode_action_index_to_tile(game, action_idx, clickable_indices)
        # Rebuild agent state for get_action_type
        game_copy = copy.deepcopy(game)
        agent_copy = game_copy.gameObjects[agent_id]
        agent_copy.x = obs_dict["position"][0] * game_copy.grid.tile_size + game_copy.grid.tile_size // 2
        agent_copy.y = obs_dict["position"][1] * game_copy.grid.tile_size + game_copy.grid.tile_size // 2
        agent_copy.item = obs_dict["holding"]
        # The action_type is now determined from the action performed (tile chosen by the policy)
        if tile is not None:
            tile_obj = game_copy.grid.tiles[tile[0]][tile[1]]
            action_type = get_action_type(tile_obj, agent_copy, agent_id, agent_food_types)
        else:
            action_type = "do_nothing"
        action_label = classify_action(action_type)
        results.append({
            "item": obs_dict["item"],
            "position": obs_dict["position"],
            "chosen_action": action_label,
            "raw_action_type": action_type,
            "tile": tile
        })
    return results

def analyze_action_results(results):
    from collections import Counter, defaultdict
    summary = defaultdict(Counter)
    for r in results:
        summary[r["item"]][r["chosen_action"]] += 1
    return summary

def print_action_summary(summary):
    for item, counter in summary.items():
        print(f"\nItem: {item}")
        total = sum(counter.values())
        for action, count in counter.items():
            print(f"  {action}: {count} ({count/total*100:.1f}%)")

# Determine grid size from map file (text format)
map_txt_path = get_map_txt_path(MAP_NR, game_version=GAME_VERSION)
if not os.path.exists(map_txt_path):
    raise FileNotFoundError(f"Map file {map_txt_path} not found.")
with open(map_txt_path, 'r') as f:
    map_lines = [line.rstrip('\n') for line in f.readlines()]
rows = len(map_lines)
cols = len(map_lines[0]) if rows > 0 else 0
if rows != cols:
    raise ValueError(f"Map must be square, but got {rows} rows and {cols} columns.")
GRID_SIZE = (rows, cols)

# --- Main evaluation loop ---
for config in training_configs:
    observations = generate_key_observations(agent_id=config["agent_id"])
    print(f"\nEvaluating policy: {config['label']} (epoch {config['epoch']})")
    policy = load_policy(
        map_nr=config["map_nr"],
        attitude=config["attitude"],
        date=config["date"],
        epoch=config["epoch"],
        game_version=config["game_version"],
        intent_version=config["intent_version"]
    )
    results = evaluate_policy_on_observations(policy, observations)
    summary = analyze_action_results(results)
    print_action_summary(summary)