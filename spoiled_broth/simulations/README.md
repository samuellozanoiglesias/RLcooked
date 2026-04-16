# Simulation Module Refactoring

## Overview

The large `utils.py` file has been refactored into individual modules for better maintainability and organization. Each class now has its own file while maintaining backward compatibility.

## New File Structure

```
spoiled_broth/simulations/
├── __init__.py                 # Package initialization with imports
├── utils.py                    # Original file (kept for backward compatibility)
├── simulation_config.py        # SimulationConfig class
├── path_manager.py            # PathManager class
├── controller_manager.py      # ControllerManager class
├── ray_manager.py             # RayManager class
├── data_logger.py             # DataLogger class
├── action_tracker.py          # ActionTracker class
├── video_recorder.py          # VideoRecorder class
├── game_manager.py            # GameManager class
├── simulation_runner.py       # SimulationRunner class
└── simulation_utils.py        # Utility functions
```

## Classes and Their Files

1. **SimulationConfig** → `simulation_config.py`
   - Configuration class for simulation parameters
   - Handles cluster settings, timing, video settings, grid settings

2. **PathManager** → `path_manager.py`
   - Manages paths and directories for simulation runs
   - Handles path setup and grid size detection from maps

3. **ControllerManager** → `controller_manager.py`
   - Manages RL controller initialization and configuration
   - Determines controller types and initializes agent controllers

4. **RayManager** → `ray_manager.py`
   - Manages Ray cluster initialization and shutdown
   - Simple static methods for Ray operations

5. **DataLogger** → `data_logger.py`
   - Handles logging of simulation state and action data
   - Creates CSV files and configuration files

6. **ActionTracker** → `action_tracker.py`
   - Tracks agent actions with detailed logging and timing
   - Handles action start/end tracking and cleanup

7. **VideoRecorder** → `video_recorder.py`
   - Handles video recording with HUD overlay
   - Manages video encoding and frame processing

8. **GameManager** → `game_manager.py`
   - Manages game instance creation and configuration
   - Handles game state reset and factory creation

9. **SimulationRunner** → `simulation_runner.py`
   - Main class for running complete simulations
   - Orchestrates all other components for full simulation execution

10. **Utility Functions** → `simulation_utils.py`
    - `setup_simulation_argument_parser()`
    - `main_simulation_pipeline()`

## Import Usage

### New Modular Imports (Recommended)
```python
from spoiled_broth.simulations import SimulationConfig, SimulationRunner
from spoiled_broth.simulations import setup_simulation_argument_parser, main_simulation_pipeline

# Or import specific modules
from spoiled_broth.simulations.simulation_config import SimulationConfig
from spoiled_broth.simulations.simulation_runner import SimulationRunner
```

### Legacy Imports (Still Supported)
```python
from spoiled_broth.simulations.utils import (
    SimulationConfig,
    SimulationRunner,
    setup_simulation_argument_parser,
    main_simulation_pipeline
)
```

## Key Dependencies Between Modules

- `simulation_runner.py` imports most other modules as it orchestrates the simulation
- `data_logger.py` imports `action_tracker.py` to avoid circular imports
- `path_manager.py`, `controller_manager.py`, and `game_manager.py` all depend on `simulation_config.py`
- All imports use relative imports (`.module_name`) within the package

## Backward Compatibility

- The original `utils.py` file is preserved
- The `__init__.py` file imports all classes and functions, maintaining the same public API
- Existing code using imports from `spoiled_broth.simulations.utils` will continue to work
- New code can use the more specific module imports for better dependency management

## Benefits of Refactoring

1. **Better Organization**: Each class has its own focused file
2. **Easier Maintenance**: Changes to one class don't require editing a large file
3. **Clearer Dependencies**: Import relationships are more explicit
4. **Improved Testing**: Individual classes can be tested in isolation
5. **Reduced Merge Conflicts**: Multiple developers can work on different classes simultaneously
6. **Better IDE Support**: Faster loading and better autocomplete for smaller files

## CSV Output Documentation

The simulation runner writes one folder per run (for example `simulation_2026_04_16-16_18_45`) containing:

- `actions.csv`
- `collisions.csv`
- `config.txt`
- `counters.csv`
- `items.csv`
- `positions_{agent_id}.csv` (for example `positions_ai_rl_1.csv`)
- `human_like_actions_{agent_id}.csv` (for example `human_like_actions_ai_rl_1.csv`)
- `human_like_positions_{agent_id}.csv` (for example `human_like_positions_ai_rl_1.csv`)
- optional run log file (`experimental_simulation_*.log`)

### actions.csv

Written when actions are assigned, plus synthetic rows for collision events.

Columns:
- `tick`
- `second`
- `agent_id`
- `action_idx`
- `action_name`
- `action_type`
- `agent_tile_x`
- `agent_tile_y`
- `tile_x`
- `tile_y`
- `action_performed`
- `action_execution_status`
- `cancelled_by_collision`
- `collision_detected`
- `collision_rerouted`

### collisions.csv

Written once per tick with predictive collision summary.

Columns:
- `frame`
- `second`
- `collision_occurred`
- `collisions_detected`
- `collisions_rerouted`
- `collisions_failed`
- `detected_agents`
- `rerouted_agents`
- `failed_agents`

### positions_{agent_id}.csv

Written every tick for each RL agent.

Columns:
- `frame`
- `second`
- `tile_x`
- `tile_y`
- `pixel_x`
- `pixel_y`
- `item_id`
- `item`
- `score`

### counters.csv

Written every tick.

Fixed columns:
- `frame`
- `second`
- `adjusted_second`

Dynamic columns:
- `counter_X_Y_id` for each counter tile (1-indexed map coordinates)

### items.csv

Written at shutdown by `ItemTracker` with live and delivered item lineage.

Columns:
- `item_id`
- `item_type`
- `creation_index`
- `created_by`
- `created_tick`
- `created_second`
- `created_source`
- `last_touched`
- `touched_list`
- `touched_list_history`
- `tomato_id`
- `plate_id`
- `tomato_cut_id`
- `tomato_salad_id`
- `tomato_delivered_id`
- `who_picked_tomato`
- `who_picked_plate`
- `who_cutted`
- `who_assembled`
- `who_delivered`
- `number_of_counters_used`

### human_like_actions_{agent_id}.csv

Human-readable per-agent action table with lineage and collaboration metadata.

Important timing semantics:
- Row is emitted when action finishes, or when it is cancelled.
- `init_second` is action start time (adjusted for initialization period).
- `finish_second` is completion/cancellation time.
- `item` and lineage fields reflect the post-action state at finish/cancel time.

Columns and meaning:
- `init_second`: Action start time in seconds, with initialization period removed.
- `finish_second`: Action completion or cancellation time in seconds, with initialization period removed.
- `item`: Item held by the agent at finish/cancel time (post-action state).
- `item_id`: Internal tracked ID of `item` from the item tracker.
- `action`: Raw action name assigned to the agent.
- `target_type`: Coarse target category (`dispenser`, `counter`, `cuttingboard`, `delivery`) inferred from action metadata.
- `target_position`: Logged target coordinates formatted as `(x, y)`.
- `action_long`: Normalized human-readable action label (for example `pick up plate from dispenser`).
- `player_id`: Agent ID (`ai_rl_*`) for this row.
- `map_name`: Map identifier from simulation config.
- `game_id`: Simulation identifier (`simulation_<timestamp>`).
- `distance_walked`: Agent cumulative walked distance (pixels) at finish/cancel time.
- `distance_walked_since_last_action`: Incremental walked distance since this agent's previous emitted human-like action row.
- `overall_score`: Sum of all RL agents' scores at finish/cancel time.
- `player_score_change`: This agent's score delta since its previous emitted human-like action row.
- `player_score`: This agent's current score.
- `walking_speed`: Agent walking speed configured for the run.
- `cutting_speed`: Agent cutting speed configured for the run.
- `start_pos`: Agent start tile position string, such as `(2, 4)`.
- `last_touched`: Last agent that touched `item` according to item tracking.
- `touched_list`: Agents that touched this concrete `item` (semicolon-separated).
- `touched_list_history`: Agents that touched any item in this item's lineage (semicolon-separated).
- `tomato_id`: Lineage tomato ID associated with this row item (if any).
- `plate_id`: Lineage plate ID associated with this row item (if any).
- `tomato_cut_id`: Lineage cut-tomato ID associated with this row item (if any).
- `tomato_salad_id`: Lineage salad ID associated with this row item (if any).
- `is_item_collaboration`: `True` when more than one agent touched this concrete item.
- `is_history_collaboration`: `True` when more than one agent appears across lineage touch history.
- `who_picked_tomato`: Agent attributed as tomato picker in lineage roles.
- `who_picked_plate`: Agent attributed as plate picker in lineage roles.
- `who_cutted`: Agent attributed as cutter in lineage roles.
- `who_assembled`: Agent attributed as assembler in lineage roles.
- `who_delivered`: Agent attributed as deliverer in lineage roles.
- `number_of_counters_used`: Count of counter interactions accumulated for this row item.
- `proportion_of_collaboration`: Per-agent role proportion vector serialized as a Python-style list string.
- `cancelled_by_collision`: `True` when the action was cancelled due to blocked/collision failure.

### human_like_positions_{agent_id}.csv

Human-readable per-agent positions table.

Columns and meaning:
- `second`: Position timestamp in seconds with initialization period removed.
- `x`: Agent x-coordinate in pixels.
- `y`: Agent y-coordinate in pixels.
- `tile_x`: Agent x-coordinate in tile space.
- `tile_y`: Agent y-coordinate in tile space.
- `distance_walked`: Agent cumulative walked distance in pixels.
- `walking_speed`: Agent walking speed configured for the run.
- `cutting_speed`: Agent cutting speed configured for the run.
- `start_pos`: Agent start tile position string.
- `item`: Item currently held by the agent at this frame (empty if none).
- `score`: Agent score at this frame.
- `frame`: Original simulation tick/frame index.

### Timing Notes

- Raw wall-clock simulation time is represented by `second = tick / engine_tick_rate`.
- `adjusted_second`, `init_second`, and `finish_second` subtract the initialization period.
- `config.txt` stores timing parameters (`AGENT_INITIALIZATION_PERIOD`, `ENGINE_TICK_RATE`, total frames/time) used for conversion.