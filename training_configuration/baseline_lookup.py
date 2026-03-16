"""
Baseline Lookup Module

Contains baseline performance data and lookup functions.
"""

# Solo baseline HA anchors (all abilities at 1.0)
# To add new baselines:
# 1. Add map name as key if not already present
# 2. Set the HA (1.0, 1.0, 1.0, 1.0) delivery count for that map
BASELINE_LOOKUP = {
    "baseline_division_of_labor_large": 14.0,
    "semiencouraged_division_of_labor_large": 10.0,
    "1-semiencouraged_division_of_labor_large": 10.0,
    "2-semiencouraged_division_of_labor_large": 10.0,
    "d5_encouraged_division_of_labor_large": 14.0,
    "d4_encouraged_division_of_labor_large": 12.0,
    "d3_encouraged_division_of_labor_large": 10.0,
    "d2_encouraged_division_of_labor_large": 9.0,
    "d1_encouraged_division_of_labor_large": 8.5,
    "m7_encouraged_division_of_labor_large": 14.0,
    "m6_encouraged_division_of_labor_large": 14.0,
    "m5_encouraged_division_of_labor_large": 14.0,
    "m4_encouraged_division_of_labor_large": 12.0,
    "m3_encouraged_division_of_labor_large": 10.0,
    "m2_encouraged_division_of_labor_large": 9.0,
    "m1_encouraged_division_of_labor_large": 8.5,
    "encouraged_division_of_labor_large": 8.0,
    "a1_encouraged_division_of_labor_large": 7.0,
    "a2_encouraged_division_of_labor_large": 6.0,
    "a3_encouraged_division_of_labor_large": 5.5,
    "a4_encouraged_division_of_labor_large": 5.0,
    "a5_encouraged_division_of_labor_large": 4.0,
    "1-encouraged_division_of_labor_large": 4.0,
    "2-encouraged_division_of_labor_large": 2.0,
    "3-encouraged_division_of_labor_large": 1.0,
}

# Ability tuples are handled as (cut1, walk1, cut2, walk2)
HA_ANCHOR_ABILITIES = (1.0, 1.0, 1.0, 1.0)
ZERO_ANCHOR_ABILITIES = (1.0, 0.1, 0.1, 1.0)

def _team_ability_feature(cut1, walk1, cut2, walk2):
    """Feature used by linear baseline regression.

    We use walk*cut per agent to capture that both abilities are needed,
    then sum both agents' terms.
    """
    return (cut1 * walk1) + (cut2 * walk2)

def _predict_team_deliveries(ha_deliveries, cut1, walk1, cut2, walk2):
    """Predict team deliveries from two anchor points via a linear model."""
    x_ha = _team_ability_feature(*HA_ANCHOR_ABILITIES)
    x_zero = _team_ability_feature(*ZERO_ANCHOR_ABILITIES)
    x = _team_ability_feature(cut1, walk1, cut2, walk2)

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
    if num_agents != 2:
        raise ValueError("Baseline lookup currently only supports 2 agents")
    
    # Extract abilities in sorted agent order as (cut1, walk1, cut2, walk2)
    agent_ids = sorted(walking_speeds.keys())
    abilities_tuple = tuple([
        cutting_speeds[agent_ids[0]],
        walking_speeds[agent_ids[0]],
        cutting_speeds[agent_ids[1]],
        walking_speeds[agent_ids[1]],
    ])
    
    # Round speeds to avoid floating point precision issues
    abilities_tuple = tuple(round(s, 2) for s in abilities_tuple)
    
    # Lookup baseline
    if map_nr not in BASELINE_LOOKUP:
        raise KeyError(f"Map '{map_nr}' not found in BASELINE_LOOKUP. Available maps: {list(BASELINE_LOOKUP.keys())}")
    
    ha_deliveries = BASELINE_LOOKUP[map_nr]

    # Predict deliveries via linear regression from HA and fixed-zero anchors.
    team_deliveries = _predict_team_deliveries(ha_deliveries, *abilities_tuple)
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
