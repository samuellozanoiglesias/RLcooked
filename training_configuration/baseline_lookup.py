"""
Baseline Lookup Module

Contains baseline performance data and lookup functions.
"""

# Solo baseline HA anchors (all abilities at 1.0)
# To add new baselines:
# 1. Add map name as key if not already present
# 2. Set the HA (1.0, 1.0, 1.0, 1.0) delivery count for that map
BASELINE_LOOKUP = {
    "baseline_division_of_labor_large": 18.0,
    "semiencouraged_division_of_labor_large": 10.0,
    "1-semiencouraged_division_of_labor_large": 10.0,
    "2-semiencouraged_division_of_labor_large": 10.0,
    "d5_encouraged_division_of_labor_large": 14.0,
    "d4_encouraged_division_of_labor_large": 16.0,
    "d3_encouraged_division_of_labor_large": 14.0,
    "d2_encouraged_division_of_labor_large": 12.0,
    "d1_encouraged_division_of_labor_large": 11.0,
    "d1_encouraged_division_of_labor_large_random_positions": 11.0,
    "m7_encouraged_division_of_labor_large": 15.0,
    "m6_encouraged_division_of_labor_large": 16.0,
    "m5_encouraged_division_of_labor_large": 14.0,
    "m4_encouraged_division_of_labor_large": 13.0,
    "m3_encouraged_division_of_labor_large": 12.0,
    "m2_encouraged_division_of_labor_large": 11.0,
    "m1_encouraged_division_of_labor_large": 10.0,
    "m1_encouraged_division_of_labor_large_random_positions": 10.0,
    "encouraged_division_of_labor_large": 10.0,
    "encouraged_division_of_labor_large_random_positions": 10.0,
    "a1_encouraged_division_of_labor_large": 9.0,
    "a2_encouraged_division_of_labor_large": 8.0,
    "a2_encouraged_division_of_labor_large_random_positions": 8.0,
    "a3_encouraged_division_of_labor_large": 6.0,
    "a4_encouraged_division_of_labor_large": 5.0,
    "a5_encouraged_division_of_labor_large": 4.0,
    "1-encouraged_division_of_labor_large": 3.0,
    "2-encouraged_division_of_labor_large": 2.0,
    "3-encouraged_division_of_labor_large": 1.0,
}

# Ability tuples are handled as (cut1, walk1, cut2, walk2, ...)

def _team_ability_feature(*abilities):
    """Feature used by linear baseline regression.

    We use walk*cut per agent to capture that both abilities are needed,
    then sum all agents' terms.
    """
    if len(abilities) % 2 != 0:
        raise ValueError("Abilities must contain cut/walk pairs")

    return sum(
        abilities[i] * abilities[i + 1]
        for i in range(0, len(abilities), 2)
    )

def _predict_team_deliveries(ha_deliveries, abilities):
    """Predict team deliveries from two anchor points via a linear model."""
    num_agents = len(abilities) // 2
    ha_abilities = tuple([1.0, 1.0] * num_agents)
    zero_abilities = tuple([1.0, 0.1] * num_agents)

    x_ha = _team_ability_feature(*ha_abilities)
    x_zero = _team_ability_feature(*zero_abilities)
    x = _team_ability_feature(*abilities)

    if x_ha == x_zero:
        raise ValueError("Invalid regression anchors: identical feature values")

    # y = m*x + b through (x_zero, 0.0) and (x_ha, ha_deliveries)
    slope = ha_deliveries / (x_ha - x_zero)
    intercept = -slope * x_zero
    predicted = (slope * x) + intercept

    # Keep predicted deliveries within feasible range.
    return max(0.0, min(ha_deliveries, predicted))


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
    if num_agents < 1:
        raise ValueError("Baseline lookup requires at least 1 agent")
    
    # Extract abilities in sorted agent order as (cut1, walk1, cut2, walk2)
    agent_ids = sorted(walking_speeds.keys())
    abilities_tuple = tuple(
        value
        for agent_id in agent_ids
        for value in (cutting_speeds[agent_id], walking_speeds[agent_id])
    )
    
    # Round speeds to avoid floating point precision issues
    abilities_tuple = tuple(round(s, 2) for s in abilities_tuple)
    
    # Lookup baseline. If the map is missing, use a conservative fallback anchor so
    # synergy shaping remains usable for newly added maps.
    if map_nr not in BASELINE_LOOKUP:
        fallback_map = "baseline_division_of_labor_large"
        if fallback_map not in BASELINE_LOOKUP:
            raise KeyError(f"Map '{map_nr}' not found in BASELINE_LOOKUP. Available maps: {list(BASELINE_LOOKUP.keys())}")
        ha_deliveries = BASELINE_LOOKUP[fallback_map]
        print(
            f"Warning: Map '{map_nr}' not found in BASELINE_LOOKUP. "
            f"Using fallback anchor '{fallback_map}' with HA deliveries={ha_deliveries}."
        )
    else:
        ha_deliveries = BASELINE_LOOKUP[map_nr]

    # Predict deliveries via linear regression from HA and fixed-zero anchors.
    team_deliveries = _predict_team_deliveries(ha_deliveries, abilities_tuple)
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
