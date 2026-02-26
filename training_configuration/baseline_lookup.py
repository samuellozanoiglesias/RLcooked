"""
Baseline Lookup Module

Contains baseline performance data and lookup functions.
"""

# Solo Baseline Lookup Dictionary
# To add new baselines:
# 1. Add map name as key if not already present
# 2. Add speed configuration tuple (walk1, cut1, walk2, cut2) with number of deliveries
BASELINE_LOOKUP = {
    "baseline_division_of_labor_large": {
        (1.0, 1.0, 1.0, 1.0): 14.0,
        (0.8, 1.0, 1.0, 0.8): 12.0,
        (0.6, 1.0, 1.0, 0.6): 8.0,
        (0.4, 1.0, 1.0, 0.4): 5.0,
        (0.4, 1.0, 1.0, 0.2): 4.0,
        (0.2, 1.0, 1.0, 0.2): 2.0,
    },
    "semiencouraged_division_of_labor_large": {
        (1.0, 1.0, 1.0, 1.0): 10.0,
        (0.8, 1.0, 1.0, 0.8): 8.0,
        (0.6, 1.0, 1.0, 0.6): 6.0,
        (0.4, 1.0, 1.0, 0.4): 4.0,
        (0.4, 1.0, 1.0, 0.2): 3.0,
        (0.2, 1.0, 1.0, 0.2): 2.0,
    },
    "1-semiencouraged_division_of_labor_large": {
        (1.0, 1.0, 1.0, 1.0): 10.0,
        (0.8, 1.0, 1.0, 0.8): 8.0,
        (0.6, 1.0, 1.0, 0.6): 6.0,
        (0.4, 1.0, 1.0, 0.4): 4.0,
        (0.4, 1.0, 1.0, 0.2): 3.0,
        (0.2, 1.0, 1.0, 0.2): 2.0,
    },
    "2-semiencouraged_division_of_labor_large": {
        (1.0, 1.0, 1.0, 1.0): 10.0,
        (0.8, 1.0, 1.0, 0.8): 8.0,
        (0.6, 1.0, 1.0, 0.6): 6.0,
        (0.4, 1.0, 1.0, 0.4): 4.0,
        (0.4, 1.0, 1.0, 0.2): 3.0,
        (0.2, 1.0, 1.0, 0.2): 2.0,
    },
    "encouraged_division_of_labor_large": {
        (1.0, 1.0, 1.0, 1.0): 10.0,
        (0.8, 1.0, 1.0, 0.8): 8.0,
        (0.6, 1.0, 1.0, 0.6): 6.0,
        (0.4, 1.0, 1.0, 0.4): 4.0,
        (0.4, 1.0, 1.0, 0.2): 3.0,
        (0.2, 1.0, 1.0, 0.2): 2.0,
    },
    "1-encouraged_division_of_labor_large": {
        (1.0, 1.0, 1.0, 1.0): 8.0,
        (0.8, 1.0, 1.0, 0.8): 6.0,
        (0.6, 1.0, 1.0, 0.6): 5.0,
        (0.4, 1.0, 1.0, 0.4): 3.0,
        (0.4, 1.0, 1.0, 0.2): 2.0,
        (0.2, 1.0, 1.0, 0.2): 1.0,
    },
    "2-encouraged_division_of_labor_large": {
        (1.0, 1.0, 1.0, 1.0): 6.0,
        (0.8, 1.0, 1.0, 0.8): 4.0,
        (0.6, 1.0, 1.0, 0.6): 3.0,
        (0.4, 1.0, 1.0, 0.4): 2.0,
        (0.4, 1.0, 1.0, 0.2): 1.0,
        (0.2, 1.0, 1.0, 0.2): 0.0,
    },
    "3-encouraged_division_of_labor_large": {
        (1.0, 1.0, 1.0, 1.0): 4.0,
        (0.8, 1.0, 1.0, 0.8): 3.0,
        (0.6, 1.0, 1.0, 0.6): 2.0,
        (0.4, 1.0, 1.0, 0.4): 1.0,
        (0.4, 1.0, 1.0, 0.2): 0.0,
        (0.2, 1.0, 1.0, 0.2): 0.0,
    },
}


def lookup_solo_baseline(map_nr, walking_speeds, cutting_speeds, delivery_reward, num_agents=2):
    """
    Look up team baseline performance from BASELINE_LOOKUP dictionary.
    
    Args:
        map_nr: Map identifier (e.g., 'baseline_division_of_labor_v2')
        walking_speeds: Dict of {agent_id: walk_speed}
        cutting_speeds: Dict of {agent_id: cut_speed}
        delivery_reward: Reward per delivery (from REWARDS_CFG["deliver"])
        num_agents: Number of agents
    
    Returns:
        tuple: (solo_baselines_dict, team_baseline)
            solo_baselines_dict: {agent_id: individual_baseline_reward}
            team_baseline: sum of individual baseline rewards
    """
    if num_agents != 2:
        raise ValueError("Baseline lookup currently only supports 2 agents")
    
    # Extract agent speeds in sorted order
    agent_ids = sorted(walking_speeds.keys())
    speeds_tuple = tuple([
        walking_speeds[agent_ids[0]],
        cutting_speeds[agent_ids[0]],
        walking_speeds[agent_ids[1]],
        cutting_speeds[agent_ids[1]]
    ])
    
    # Round speeds to avoid floating point precision issues
    speeds_tuple = tuple(round(s, 2) for s in speeds_tuple)
    
    # Lookup baseline
    if map_nr not in BASELINE_LOOKUP:
        raise KeyError(f"Map '{map_nr}' not found in BASELINE_LOOKUP. Available maps: {list(BASELINE_LOOKUP.keys())}")
    
    map_baselines = BASELINE_LOOKUP[map_nr]
    if speeds_tuple not in map_baselines:
        raise KeyError(
            f"Speed configuration {speeds_tuple} not found for map '{map_nr}'.\n"
            f"Available configurations: {list(map_baselines.keys())}"
        )
    
    # Get number of deliveries and convert to reward
    team_deliveries = map_baselines[speeds_tuple]
    team_baseline = team_deliveries * delivery_reward
    
    # Split baseline among agents proportionally to their competence
    # Competence = average of walking and cutting speed
    competences = {}
    total_competence = 0.0
    for agent_id in agent_ids:
        comp = (walking_speeds[agent_id] + cutting_speeds[agent_id]) / 2.0
        competences[agent_id] = comp
        total_competence += comp
    
    # Distribute team baseline proportionally
    solo_baselines = {}
    if total_competence > 0:
        for agent_id in agent_ids:
            solo_baselines[agent_id] = team_baseline * (competences[agent_id] / total_competence)
    else:
        # Equal split if all competences are zero (shouldn't happen)
        for agent_id in agent_ids:
            solo_baselines[agent_id] = team_baseline / num_agents
    
    return solo_baselines, team_baseline
