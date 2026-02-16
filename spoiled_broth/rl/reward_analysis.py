# ---- Reward analysis module ---- #
import numpy as np

def get_cutting_time(agent, game):
    """Calculate actual cutting time based on agent's cutting speed."""
    cutting_speed = getattr(agent, 'cut_speed', 1.0)
    base_cutting_time = getattr(game, 'cutting_time', 3.0)
    # Lower speed = more time (inverse relationship)
    return base_cutting_time / cutting_speed if cutting_speed > 0 else base_cutting_time

def get_rewards(self, agent_events, agent_penalties, rewards_cfg):
    """
    Calculate pure and modified rewards based on the game mode.
    
    For team synergy-based shaping, this uses:
    - self._elapsed_time: current time in episode (seconds)
    - self._max_seconds_per_episode: maximum episode duration (seconds)
    - self.solo_baseline_team: sum of individual solo baselines
    - self.cumulated_pure_rewards: cumulative rewards so far
    - self.agent_abilities: agent cutting and walking abilities (kappa values) for dynamic competence calculation
    """
    if self.game_mode == "classic":
        return get_rewards_classic(self, agent_events, agent_penalties, rewards_cfg)
    elif self.game_mode == "competition":
        return get_rewards_competition(self, agent_events, agent_penalties, rewards_cfg)
    else:
        raise ValueError(f"Unknown game mode: {self.game_mode}")


# ---- Classic mode without ownership awareness ---- #
def get_rewards_classic(self, agent_events, agent_penalties, rewards_cfg):
    """
    Calculate pure and modified rewards for classic mode.
    """    
    event_rewards = {agent_id: 0.0 for agent_id in self.agents}
    deliver_rewards = {agent_id: 0.0 for agent_id in self.agents}
    for agent_id in self.agents:
        # Event rewards: apply specialization-based rewards/penalties
        event_rewards[agent_id] = _apply_specialization_rewards(
            self, agent_id, agent_events, rewards_cfg
        )
        deliver_rewards[agent_id] = _get_specialized_reward_for_event(
            self, agent_id, "deliver", agent_events[agent_id]["deliver"], rewards_cfg["deliver"]
        )

    shared_deliver_reward = sum(deliver_rewards.values())

    for agent_id in self.agents:
        reward = shared_deliver_reward + event_rewards[agent_id]
        self.cumulated_pure_rewards[agent_id] += reward

        alpha, beta = self.reward_weights.get(agent_id, (1.0, 0.0))
        other_agents = [a for a in self.agents if a != agent_id]
        avg_other_reward = (shared_deliver_reward +
            sum(event_rewards[a] for a in other_agents) / len(other_agents)
            if other_agents else 0.0
        )  # in case there is only one agent

        # Modified rewards: include penalties and team synergy-based shaping
        modified_reward = alpha * (reward - agent_penalties[agent_id]) + beta * avg_other_reward
        
        # Apply team synergy-based shaping if enabled
        if self.reference_reward_enabled:
            # Calculate episode progress τ ∈ [0,1]
            tau = self._elapsed_time / self._max_seconds_per_episode if self._max_seconds_per_episode > 0 else 0
            
            # Calculate cumulative team performance
            cumulative_team_reward = sum(self.cumulated_pure_rewards[a] for a in self.agents)
            
            # Team Synergy: S(τ) = tanh(R^cum_team(τ) - τ * R̄^solo_team)
            time_scaled_baseline = tau * self.solo_baseline_team
            team_synergy = np.tanh(cumulative_team_reward - time_scaled_baseline)
            
            # Asymmetric distribution based on agent competence using abilities (kappa values)
            if hasattr(self, 'agent_abilities') and agent_id in self.agent_abilities:
                competence = (np.exp(self.kappa * self.agent_abilities[agent_id]['average']) - 1)/(np.exp(self.kappa) - 1)
                
                if team_synergy < 0:  # Risk Aversion: S(τ) * competence
                    synergy_signal = team_synergy * competence
                elif team_synergy >= 0 and self.activate_synergy_positive:  # Cooperation Incentive: S(τ) * (1 - competence)
                    synergy_signal = team_synergy * (1 - competence)
                else:
                    synergy_signal = 0.0  # No positive synergy if not activated
                
                # Scale by synergy_scaling_factor: R^{ref}_i = R^{env}_i - P^{spec}_i + synergy_scaling_factor · Ψ_i(τ)
                synergy_contribution = self.synergy_scaling_factor * synergy_signal
                modified_reward += synergy_contribution
        
        self.modified_rewards[agent_id] = modified_reward
        self.cumulated_modified_rewards[agent_id] += self.modified_rewards[agent_id]

    return self.cumulated_pure_rewards, self.cumulated_modified_rewards

# ---- Competition mode with ownership awareness ---- #
def get_rewards_competition(self, agent_events, agent_penalties, rewards_cfg):
    pure_rewards = {agent_id: 0.0 for agent_id in self.agents}
    for agent_id in self.agents:
        # Pure rewards: apply specialization-based rewards/penalties
        reward_from_own = (
            _get_specialized_reward_for_event(self, agent_id, "deliver", agent_events[agent_id]["deliver_own"], rewards_cfg["deliver"])
            + _get_specialized_reward_for_event(self, agent_id, "salad", agent_events[agent_id]["salad_own"], rewards_cfg["salad"])
            + _get_specialized_reward_for_event(self, agent_id, "cut", agent_events[agent_id]["cut_own"], rewards_cfg["cut"])
            + agent_events[agent_id]["counter"] * rewards_cfg["counter"]  # No specialization for counter
            + _get_specialized_reward_for_event(self, agent_id, "raw_food", agent_events[agent_id]["raw_food_own"], rewards_cfg["raw_food"])
            + _get_specialized_reward_for_event(self, agent_id, "plate", agent_events[agent_id]["plate"], rewards_cfg["plate"])
        )

        reward_from_other = (
            _get_specialized_reward_for_event(self, agent_id, "deliver", agent_events[agent_id]["deliver_other"], rewards_cfg["deliver"])
            + _get_specialized_reward_for_event(self, agent_id, "salad", agent_events[agent_id]["salad_other"], rewards_cfg["salad"])
            + _get_specialized_reward_for_event(self, agent_id, "cut", agent_events[agent_id]["cut_other"], rewards_cfg["cut"])
            + _get_specialized_reward_for_event(self, agent_id, "raw_food", agent_events[agent_id]["raw_food_other"], rewards_cfg["raw_food"])
        )

        penalty_from_other = 0
        for other_agent_id in self.agents:
            if other_agent_id == agent_id:
                continue
            penalty_from_other += (
                agent_events[other_agent_id]["deliver_other"] * rewards_cfg["deliver"]
            )

        pure_rewards[agent_id] = (
            self.payoff_matrix[0] * reward_from_own +
            self.payoff_matrix[1] * reward_from_other +
            self.payoff_matrix[2] * penalty_from_other
        )
        self.cumulated_pure_rewards[agent_id] += pure_rewards[agent_id]

    for agent_id in self.agents:
        alpha, beta = self.reward_weights.get(agent_id, (1.0, 0.0))
        other_agents = [a for a in self.agents if a != agent_id]
        if other_agents:
            avg_other_reward = sum(pure_rewards[a] for a in other_agents) / len(other_agents)
        else:
            avg_other_reward = 0.0  # in case there is only one agent
        
        # Modified rewards: include penalties and team synergy-based shaping
        modified_reward = alpha * (pure_rewards[agent_id] - agent_penalties[agent_id]) + beta * avg_other_reward
        
        # Apply team synergy-based shaping if enabled
        if self.reference_reward_enabled:
            # Calculate episode progress τ ∈ [0,1]
            tau = self._elapsed_time / self._max_seconds_per_episode if self._max_seconds_per_episode > 0 else 0
            
            # Calculate cumulative team performance
            cumulative_team_reward = sum(self.cumulated_pure_rewards[a] for a in self.agents)
            
            # Team Synergy: S(τ) = tanh(R^cum_team(τ) - τ * R̄^solo_team)
            time_scaled_baseline = tau * self.solo_baseline_team
            team_synergy = np.tanh(cumulative_team_reward - time_scaled_baseline)
            
            # Asymmetric distribution based on agent competence using abilities (kappa values)
            if hasattr(self, 'agent_abilities') and agent_id in self.agent_abilities:
                competence = (np.exp(self.kappa * self.agent_abilities[agent_id]['average']) - 1)/(np.exp(self.kappa) - 1)
                
                if team_synergy < 0:  # Risk Aversion: S(τ) * competence
                    synergy_signal = team_synergy * competence
                elif team_synergy >= 0 and self.activate_synergy_positive:  # Cooperation Incentive: S(τ) * (1 - competence)
                    synergy_signal = team_synergy * (1 - competence)
                else:
                    synergy_signal = 0.0  # No positive synergy if not activated
                
                # Scale by synergy_scaling_factor: R^{ref}_i = R^{env}_i - P^{spec}_i + synergy_scaling_factor · Ψ_i(τ)
                synergy_contribution = self.synergy_scaling_factor * synergy_signal
                modified_reward += synergy_contribution
        
        self.modified_rewards[agent_id] = modified_reward
        self.cumulated_modified_rewards[agent_id] += self.modified_rewards[agent_id]

    return self.cumulated_pure_rewards, self.cumulated_modified_rewards

def _get_specialized_reward_for_event(self, agent_id, event_type, event_count, base_reward):
    """
    Calculate specialized reward for a specific event based on agent abilities.
    
    Args:
        self: GameEnv instance
        agent_id: ID of the agent
        event_type: Type of event ("cut", "deliver", "raw_food", "plate", "salad")
        event_count: Number of times the event occurred
        base_reward: Base reward value from REWARDS_CFG
    
    Returns:
        float: Reward (positive for specialists, negative penalty for non-specialists)
    """
    if event_count == 0:
        return 0.0
    
    # Get agent abilities
    if hasattr(self, 'agent_abilities') and agent_id in self.agent_abilities:
        walk_speed = self.agent_abilities[agent_id].get('walking', 1.0)
        cut_speed = self.agent_abilities[agent_id].get('cutting', 1.0)
    else:
        # Fallback if abilities not available
        walk_speed = cut_speed = 1.0
    
    # Get specialization scale from penalties config
    specialization_scale = self.penalties_cfg.get("specialization_penalty_scale", 0.0)
    
    # Increase specialization significance when collisions are enabled
    if hasattr(self, 'collision_enabled') and self.collision_enabled:
        collision_harshness = self.penalties_cfg.get("collision_harshness", 2.0)
        specialization_scale *= collision_harshness
    
    if specialization_scale == 0.0:
        # Specialization system disabled
        return base_reward * event_count
    
    # Determine reward/penalty based on event type and agent specialization
    if event_type == "cut":
        # Cutting action - requires cut_speed = 1
        if cut_speed >= 1.0:
            if walk_speed < 1.0:
                reward = base_reward * event_count  # Full reward for specialist
                return reward
            else:
                # Both speeds are >= 1.0
                reward = 0
                return reward
        else:
            penalty = base_reward * (1.0 - cut_speed) * specialization_scale * event_count
            return -penalty  # Penalty for non-specialist
    
    elif event_type in ["deliver", "raw_food", "plate"]:
        # Delivery/dispenser actions - require walk_speed = 1
        if walk_speed >= 1.0:
            if cut_speed < 1.0:
                reward = base_reward * event_count * specialization_scale  # Full reward for specialist
                return reward
            else:
                # Both speeds are >= 1.0
                reward = 0
                return reward
        else:
            penalty = base_reward * (1.0 - walk_speed) * specialization_scale * event_count
            return -penalty  # Penalty for non-specialist
    
    elif event_type == "salad":
        # Salad assembly - rewarded for both specializations
        if cut_speed >= 1.0 or walk_speed >= 1.0:
            reward = base_reward * event_count * specialization_scale  # Full reward if specialist in either
            return reward
        else:
            # Penalty if not specialist in either (use max speed to be lenient)
            max_speed = max(cut_speed, walk_speed)
            penalty = base_reward * (1.0 - max_speed) * specialization_scale * event_count
            return -penalty
    
    else:
        # No specialization requirement (e.g., "counter")
        reward = base_reward * event_count * specialization_scale
        return reward

def _apply_specialization_rewards(self, agent_id, agent_events, rewards_cfg):
    """
    Apply specialization-based rewards/penalties for all non-delivery events.
    
    Args:
        self: GameEnv instance
        agent_id: ID of the agent
        agent_events: Dictionary of event counts for the agent
        rewards_cfg: Reward configuration dictionary
    
    Returns:
        float: Total reward after specialization adjustments
    """
    total_reward = 0.0
    
    # Process each event type (excluding deliver which is handled separately)
    event_types = ["raw_food", "plate", "counter", "cut", "salad"]
    for event_type in event_types:
        if event_type in agent_events[agent_id] and event_type in rewards_cfg:
            event_reward = _get_specialized_reward_for_event(
                self, agent_id, event_type, agent_events[agent_id][event_type], rewards_cfg[event_type]
            )
            total_reward += event_reward
    
    return total_reward

def apply_adaptive_cooperation_penalty(self, agent_id, path_length, agent_penalties):
    """
    Apply adaptive cooperation penalty to encourage path-length-based specialization.
    
    This encourages agents to specialize based on path length:
    - Slow walkers are penalized more for long paths
    - Fast walkers are encouraged to handle distant tasks
    - Penalty scales with: path_length * (1 - walk_speed) * adaptive_scale
    - When collisions are enabled, the scale is multiplied by collision_harshness for stronger cooperation
    
    Args:
        self: GameEnv instance
        agent_id: ID of the agent
        path_length: Number of tiles in the path
        agent_penalties: Dictionary to store penalties
    """
    # Get base adaptive cooperation scale
    adaptive_scale = self.penalties_cfg.get("adaptive_cooperation_scale", 0.0)
    
    if adaptive_scale == 0.0 or path_length == 0:
        return
    
    # Increase adaptive cooperation significance when collisions are enabled
    # This makes path-length-based specialization more important in collision environments
    if hasattr(self, 'collision_enabled') and self.collision_enabled:
        adaptive_scale *= self.collision_harshness
    
    # Get agent's walking ability
    if hasattr(self, 'agent_abilities') and agent_id in self.agent_abilities:
        walk_speed = self.agent_abilities[agent_id].get('walking', 1.0)
    else:
        walk_speed = 1.0
    
    # Calculate penalty: longer paths and slower walkers = higher penalty
    # Formula: path_length * (1 - walk_speed) * adaptive_scale
    # - walk_speed=1.0 (fast): no penalty regardless of path length
    # - walk_speed=0.5 (slow): 0.5 * path_length * scale penalty
    # - walk_speed=0.2 (very slow): 0.8 * path_length * scale penalty
    slowness_factor = 1.0 - walk_speed
    cooperation_penalty = path_length * slowness_factor * adaptive_scale
    
    # Add to agent's penalties
    agent_penalties[agent_id] += cooperation_penalty