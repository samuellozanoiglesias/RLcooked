# ---- Reward analysis module ---- #
import numpy as np

def get_cutting_time(agent, game):
    """Calculate actual cutting time based on agent's cutting speed."""
    cutting_speed = getattr(agent, 'cut_speed', 1.0)
    base_cutting_time = getattr(game, 'cutting_time', 3.0)
    # Lower speed = more time (inverse relationship)
    return base_cutting_time / cutting_speed if cutting_speed > 0 else base_cutting_time

def get_rewards(self, agent_events, agent_penalties, rewards_cfg, intermediate_reward_decay_cfg=None, episode=0):
    """
    Calculate pure and modified rewards based on the game mode.
    
    For team synergy-based shaping, this uses:
    - self._elapsed_time: current time in episode (seconds)
    - self._max_seconds_per_episode: maximum episode duration (seconds)
    - self.solo_baseline_team: sum of individual solo baselines
    - self.cumulated_pure_rewards: cumulative rewards so far
    - self.agent_abilities: agent cutting and walking abilities (kappa values) for dynamic competence calculation
    - intermediate_reward_decay_cfg: Configuration for intermediate reward decay
    - episode: Current episode number for reward decay calculation
    """
    if self.game_mode == "classic":
        return get_rewards_classic(self, agent_events, agent_penalties, rewards_cfg, intermediate_reward_decay_cfg, episode)
    elif self.game_mode == "competition":
        return get_rewards_competition(self, agent_events, agent_penalties, rewards_cfg, intermediate_reward_decay_cfg, episode)
    else:
        raise ValueError(f"Unknown game mode: {self.game_mode}")


# ---- Classic mode without ownership awareness ---- #
def get_rewards_classic(self, agent_events, agent_penalties, rewards_cfg, intermediate_reward_decay_cfg=None, episode=0):
    """
    Calculate pure and modified rewards for classic mode.
    """    
    # Apply intermediate reward decay if enabled
    effective_rewards_cfg = rewards_cfg.copy()
    if intermediate_reward_decay_cfg and intermediate_reward_decay_cfg.get("enabled", False):
        from training_configuration.reward_penalties import apply_reward_decay
        effective_rewards_cfg = apply_reward_decay(rewards_cfg, episode, intermediate_reward_decay_cfg)
    
    event_rewards = {agent_id: 0.0 for agent_id in self.agents}
    deliver_rewards = {agent_id: 0.0 for agent_id in self.agents}
    for agent_id in self.agents:
        # Event rewards: apply specialization-based rewards/penalties
        event_rewards[agent_id] = _apply_specialization_rewards(
            self, agent_id, agent_events, effective_rewards_cfg
        )
        deliver_rewards[agent_id] = _get_specialized_reward_for_event(
            self, agent_id, "deliver", agent_events[agent_id]["deliver"], effective_rewards_cfg["deliver"]
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
        self.cumulated_modified_rewards[agent_id] += modified_reward
        
    return self.cumulated_pure_rewards, self.cumulated_modified_rewards


# ---- Competition mode with ownership awareness ---- #
def get_rewards_competition(self, agent_events, agent_penalties, rewards_cfg, intermediate_reward_decay_cfg=None, episode=0):
    # Apply intermediate reward decay if enabled
    effective_rewards_cfg = rewards_cfg.copy()
    if intermediate_reward_decay_cfg and intermediate_reward_decay_cfg.get("enabled", False):
        from training_configuration.reward_penalties import apply_reward_decay
        effective_rewards_cfg = apply_reward_decay(rewards_cfg, episode, intermediate_reward_decay_cfg)
    
    pure_rewards = {agent_id: 0.0 for agent_id in self.agents}
    reward_from_own_food_by_agent = {agent_id: 0.0 for agent_id in self.agents}
    reward_from_other_food_by_agent = {agent_id: 0.0 for agent_id in self.agents}
    support_reward_by_agent = {agent_id: 0.0 for agent_id in self.agents}

    for agent_id in self.agents:
        # Reward from using the agent's own food type.
        reward_from_own_food = (
            _get_specialized_reward_for_event(self, agent_id, "deliver", agent_events[agent_id]["deliver_own"], effective_rewards_cfg["deliver"])
            + _get_specialized_reward_for_event(self, agent_id, "salad", agent_events[agent_id]["salad_own"], effective_rewards_cfg["salad"])
            + _get_specialized_reward_for_event(self, agent_id, "cut", agent_events[agent_id]["cut_own"], effective_rewards_cfg["cut"])
            + _get_specialized_reward_for_event(self, agent_id, "raw_food", agent_events[agent_id]["raw_food_own"], effective_rewards_cfg["raw_food"])
        )

        # Support rewards not tied to food ownership competition.
        support_reward = (
            agent_events[agent_id]["counter"] * effective_rewards_cfg["counter"]
            + _get_specialized_reward_for_event(self, agent_id, "plate", agent_events[agent_id]["plate"], effective_rewards_cfg["plate"])
        )

        # Reward from using the other agent's food type.
        reward_from_other_food = (
            _get_specialized_reward_for_event(self, agent_id, "deliver", agent_events[agent_id]["deliver_other"], effective_rewards_cfg["deliver"])
            + _get_specialized_reward_for_event(self, agent_id, "salad", agent_events[agent_id]["salad_other"], effective_rewards_cfg["salad"])
            + _get_specialized_reward_for_event(self, agent_id, "cut", agent_events[agent_id]["cut_other"], effective_rewards_cfg["cut"])
            + _get_specialized_reward_for_event(self, agent_id, "raw_food", agent_events[agent_id]["raw_food_other"], effective_rewards_cfg["raw_food"])
        )

        reward_from_own_food_by_agent[agent_id] = reward_from_own_food
        reward_from_other_food_by_agent[agent_id] = reward_from_other_food
        support_reward_by_agent[agent_id] = support_reward

    for agent_id in self.agents:
        other_agents = [other_id for other_id in self.agents if other_id != agent_id]

        # Third payoff term: points related to the OTHER agent using HIS own food.
        # This mirrors the first term but for the opponent(s), and uses payoff_matrix as-is.
        other_loss_from_own_food = sum(reward_from_own_food_by_agent[other_id] for other_id in other_agents)

        pure_rewards[agent_id] = (
            support_reward_by_agent[agent_id]
            + self.payoff_matrix[0] * reward_from_own_food_by_agent[agent_id]
            + self.payoff_matrix[1] * reward_from_other_food_by_agent[agent_id]
            + self.payoff_matrix[2] * other_loss_from_own_food
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
    
    # Get specialization parameters from penalties config
    specialization_scale = self.penalties_cfg.get("specialization_penalty_scale", 0.0)
    specialization_theta = self.penalties_cfg.get("specialization_theta", 1.0)
    
    # Increase specialization significance when collisions are enabled
    if hasattr(self, 'collision_enabled') and self.collision_enabled:
        collision_harshness = self.penalties_cfg.get("collision_harshness", 2.0)
        specialization_scale *= collision_harshness
    
    event_base_reward = base_reward * event_count

    if specialization_scale == 0.0:
        # Specialization system disabled
        return event_base_reward
    
    # Check for balanced agents (both abilities at 1.0) - they should get normal rewards
    if cut_speed >= 1.0 and walk_speed >= 1.0:
        # Balanced agent - same rewards as if specialization was disabled
        return event_base_reward
    
    # Determine reward/penalty based on event type and agent specialization
    if event_type == "cut":
        # Cutting action - requires cut_speed = 1
        if cut_speed >= 1.0:
            return event_base_reward
        else:
            # penalty = base_reward * (exp(theta * (1.0 - speed)) - 1) * specialization_scale * event_count
            penalty = base_reward * (np.exp(specialization_theta * (1.0 - cut_speed)) - 1.0) * specialization_scale * event_count
            return event_base_reward - penalty  # Penalty for non-specialist
    
    elif event_type in ["deliver", "raw_food", "plate"]:
        # Delivery/dispenser actions - require walk_speed = 1
        if walk_speed >= 1.0:
            return event_base_reward
        else:
            # penalty = base_reward * (exp(theta * (1.0 - speed)) - 1) * specialization_scale * event_count
            penalty = base_reward * (np.exp(specialization_theta * (1.0 - walk_speed)) - 1.0) * specialization_scale * event_count
            return event_base_reward - penalty  # Penalty for non-specialist
    
    else:
        # No specialization requirement (e.g., "counter" or "salad")
        return event_base_reward

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