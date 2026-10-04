"""
Tick-based simulation helper functions for GameEnv.

This module contains all the helper methods for tick-based simulation:
- Agent state management
- Movement along precomputed paths
- Interaction timing

TIME MANAGEMENT (single source of truth for the constants used everywhere):
- TICK_DURATION = 0.5 s. Every simulation tick advances time by this amount.
- Movement: agent.speed is 30 * walk_speed px/s (set in SpoiledBroth.add_agent);
  with 16 px tiles this is 1.875 * walk_speed tiles/s. Progress is accumulated
  continuously and the remainder is CARRIED OVER between tiles, so the realised
  speed matches the nominal speed on average (previously the remainder was
  discarded, which made every walk_speed in [0.55, 1.0] behave identically).
- A step between two path nodes costs its Euclidean length (1 for straight
  moves, sqrt(2) for diagonal moves), consistent with the A* path lengths that
  the observation uses to compute travel times and with the continuous
  movement of the human-facing game engine.
- At most one tile is entered per tick (valid for walk_speed <= ~1.06, which
  covers every configuration used in the study).
- Interactions: cutting takes cutting_time / cut_speed seconds; every other
  interaction takes INTENT_TIME (one tick).
"""

import logging
import math

logger = logging.getLogger(__name__)

TICK_DURATION = 0.5          # seconds per simulation tick
INTENT_TIME = TICK_DURATION  # pick-up / put-down / delivery take one tick
TILE_SIZE_PX = 16


def init_agent_state():
    """Fresh per-agent execution state (used at construction and at every reset)."""
    return {
        'current_action': None,          # Action name being executed
        'current_path': [],              # Remaining path nodes (excluding current tile)
        'path_index': 0,                 # Index of the next node to enter
        'movement_progress': 0.0,        # Tiles of progress toward the next node
        'interaction_timer': 0.0,        # Seconds left for the interaction
        'target_tile_index': None,       # Tile that will be interacted with
        'action_type': None,             # Classification of current action (logging only)
        'interaction_target_tile': None, # (x, y) of tile being interacted with
    }


def agent_is_idle(env, agent_id):
    """
    An agent is idle (ready for a new decision) only when it has no action,
    no running interaction timer and no remaining movement path.
    """
    state = env.agent_state[agent_id]
    return (state['current_action'] is None and
            state['interaction_timer'] <= 0.0 and
            len(state['current_path']) == 0)


def get_interaction_time(env, agent_id, action_name):
    """Interaction duration once the agent stands at its destination."""
    if action_name == "use_cutting_board":
        from spoiled_broth.rl.reward_analysis import get_cutting_time
        return get_cutting_time(env.agent_map[agent_id], env.game)
    return INTENT_TIME


def movement_step_cost(agent, next_node):
    """Distance (in tiles) from the agent's current tile to the next path node."""
    d = math.hypot(next_node.x - agent.slot_x, next_node.y - agent.slot_y)
    return d if d > 0 else 1.0


def _xy_to_index(env, xy):
    return xy[1] * env.game.grid.width + xy[0]


def assign_action(env, agent_id, action_idx, action_name, tile_index, cached_path, action_type,
                  interaction_target_tile=None):
    """Assign a new action to an agent for execution.

    Args:
        tile_index: index of the tile to interact with, or -1 for an action that
            is blocked by another agent (only executed when allow_blocked=True).
        cached_path: path from the agent's tile to the standing tile
            ([start] if the agent is already adjacent to the target).
        interaction_target_tile: (x, y) of the tile to interact with.

    Returns:
        int: number of path steps to walk (0 if none or if the action was rejected).
    """
    state = env.agent_state[agent_id]
    agent = env.agent_map[agent_id]

    # Resolve which tile will be interacted with on completion.
    if tile_index == -1:
        # BLOCKED action: the interaction tile is the target, NOT the last floor
        # tile of the path (previously the floor tile was used, so blocked
        # actions completed as no-ops).
        if interaction_target_tile is None:
            return 0
        target_index = _xy_to_index(env, interaction_target_tile)
    else:
        target_index = tile_index

    if not cached_path:
        # No path at all: only valid if the agent already stands on the target.
        grid_w = env.game.grid.width
        if (agent.slot_x, agent.slot_y) != (target_index % grid_w, target_index // grid_w):
            return 0
        remaining = []
    else:
        start = cached_path[0]
        if (start.x, start.y) != (agent.slot_x, agent.slot_y):
            logger.debug(f"Path mismatch for {agent_id} action {action_name}: agent at "
                         f"{(agent.slot_x, agent.slot_y)}, path starts at {(start.x, start.y)}")
            return 0
        remaining = list(cached_path[1:])

    state['current_action'] = action_name
    state['target_tile_index'] = target_index
    state['action_type'] = action_type
    state['interaction_target_tile'] = interaction_target_tile
    state['current_path'] = remaining
    state['path_index'] = 0
    state['movement_progress'] = 0.0
    if not remaining:
        # Already in place: start the interaction immediately.
        state['interaction_timer'] = get_interaction_time(env, agent_id, action_name)
    return len(remaining)


def cancel_agent_action(env, agent_id):
    """Cancel agent's current action due to collision or invalidation."""
    env.agent_state[agent_id].update(init_agent_state())
    if env.path_processor.is_enabled():
        env.path_processor.clear_agent_path(agent_id)


def advance_agent_movement(env, agent_id, tick_duration):
    """Move an agent along its path for one tick.

    Progress accumulates at agent.speed / 16 tiles per second; entering the next
    node consumes its step cost and the remainder is carried over.

    Returns:
        tuple: (reached_next_tile, reached_final_tile, blocked) -- blocked is always
        False here (collisions are resolved before movement).
    """
    state = env.agent_state[agent_id]
    agent = env.agent_map[agent_id]
    path = state['current_path']

    if not path or state['path_index'] >= len(path):
        return False, False, False

    state['movement_progress'] += (agent.speed / TILE_SIZE_PX) * tick_duration

    next_node = path[state['path_index']]
    cost = movement_step_cost(agent, next_node)
    if state['movement_progress'] < cost:
        return False, False, False

    # Enter the next node, keep the leftover progress.
    state['movement_progress'] -= cost
    agent.x = next_node.x * TILE_SIZE_PX + TILE_SIZE_PX // 2
    agent.y = next_node.y * TILE_SIZE_PX + TILE_SIZE_PX // 2
    state['path_index'] += 1

    if state['path_index'] >= len(path):
        # Destination reached: start the interaction and clear movement.
        state['interaction_timer'] = get_interaction_time(env, agent_id, state['current_action'])
        state['current_path'] = []
        state['path_index'] = 0
        state['movement_progress'] = 0.0
        return True, True, False

    return True, False, False


def update_agent_interactions(env, agent_events, agent_penalties, tick_duration):
    """Count down interaction timers and complete interactions that finish this tick.

    Cutting: cutting_time / cut_speed seconds (e.g. 3 s at cut_speed=1, i.e. 6 ticks).
    Other interactions: INTENT_TIME (1 tick). An interaction that starts on the
    tick the agent arrives finishes on that same tick if it lasts one tick.
    """
    from spoiled_broth.rl.game_step import complete_agent_action

    for agent_id in env.agents:
        state = env.agent_state[agent_id]
        if state['interaction_timer'] <= 0:
            continue
        state['interaction_timer'] -= tick_duration
        if state['interaction_timer'] > 1e-9:
            continue

        state['interaction_timer'] = 0.0
        tile_index = state['target_tile_index']
        if tile_index is not None:
            agent = env.agent_map[agent_id]
            agent_food_type = (env.agent_food_type.get(agent_id)
                               if getattr(env, 'agent_food_type', None) is not None else None)
            agent_events = complete_agent_action(env, agent_id, agent, {'tile_index': tile_index},
                                                 agent_events, agent_food_type)
        cancel_agent_action(env, agent_id)

    return agent_events