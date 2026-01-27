# ---- Reward analysis module ---- #
def get_rewards(self, agent_events, agent_penalties, rewards_cfg):
    """
    Calculate pure and modified rewards based on the game mode.
    
    For reference-based opportunity cost shaping, this uses:
    - self._elapsed_time: current time in episode (seconds)
    - self._max_seconds_per_episode: maximum episode duration (seconds)
    - self.solo_baseline_team: sum of individual solo baselines
    - self.cumulated_pure_rewards: cumulative rewards so far
    - self.eta: opportunity cost sensitivity parameter
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
        # Event rewards: only positive events
        event_rewards[agent_id] = (
            agent_events[agent_id]["raw_food"] * rewards_cfg["raw_food"]
            + agent_events[agent_id]["plate"] * rewards_cfg["plate"]
            + agent_events[agent_id]["counter"] * rewards_cfg["counter"]
            + agent_events[agent_id]["cut"] * rewards_cfg["cut"]
            + agent_events[agent_id]["salad"] * rewards_cfg["salad"]
        )
        deliver_rewards[agent_id] = agent_events[agent_id]["deliver"] * rewards_cfg["deliver"]

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

        # Modified rewards: include penalties and reference-based opportunity cost
        modified_reward = alpha * (reward - agent_penalties[agent_id]) + beta * avg_other_reward
        
        # Apply reference-based opportunity cost shaping if enabled
        if self.reference_reward_enabled:
            # Calculate progress ratio based on elapsed time (how far through the episode we are)
            progress_ratio = self._elapsed_time / self._max_seconds_per_episode if self._max_seconds_per_episode > 0 else 0
            
            # Pro-rated baseline: what we expect to have earned by this point in time
            prorated_baseline_team = progress_ratio * self.solo_baseline_team
            
            # Cumulative team performance so far
            cumulative_team_reward = sum(self.cumulated_pure_rewards[a] for a in self.agents)
            
            # Opportunity cost: how much we're underperforming relative to expectation
            opportunity_cost = max(0, prorated_baseline_team - cumulative_team_reward)
            
            # Distribute penalty proportionally to each agent's solo baseline contribution
            if self.solo_baseline_team > 0:
                agent_solo_baseline = self.solo_baselines.get(agent_id, 0)
                agent_proportion = agent_solo_baseline / self.solo_baseline_team
                agent_penalty = self.eta * opportunity_cost * agent_proportion
                modified_reward -= agent_penalty
        
        self.modified_rewards[agent_id] = modified_reward
        self.cumulated_modified_rewards[agent_id] += self.modified_rewards[agent_id]

    return self.cumulated_pure_rewards, self.cumulated_modified_rewards

# ---- Competition mode with ownership awareness ---- #
def get_rewards_competition(self, agent_events, agent_penalties, rewards_cfg):
    pure_rewards = {agent_id: 0.0 for agent_id in self.agents}
    for agent_id in self.agents:
        # Pure rewards: only positive events, no IDLE or useless penalties
        reward_from_own = (
            agent_events[agent_id]["deliver_own"] * rewards_cfg["deliver"]
            + agent_events[agent_id]["salad_own"] * rewards_cfg["salad"]
            + agent_events[agent_id]["cut_own"] * rewards_cfg["cut"]
            + agent_events[agent_id]["counter"] * rewards_cfg["counter"]
            + agent_events[agent_id]["raw_food_own"] * rewards_cfg["raw_food"]
            + agent_events[agent_id]["plate"] * rewards_cfg["plate"]
        )

        reward_from_other = (
            agent_events[agent_id]["deliver_other"] * rewards_cfg["deliver"]
            + agent_events[agent_id]["salad_other"] * rewards_cfg["salad"]
            + agent_events[agent_id]["cut_other"] * rewards_cfg["cut"]
            + agent_events[agent_id]["raw_food_other"] * rewards_cfg["raw_food"]
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
        
        # Modified rewards: include penalties and reference-based opportunity cost
        modified_reward = alpha * (pure_rewards[agent_id] - agent_penalties[agent_id]) + beta * avg_other_reward
        
        # Apply reference-based opportunity cost shaping if enabled
        if self.reference_reward_enabled:
            # Calculate progress ratio based on elapsed time (how far through the episode we are)
            progress_ratio = self._elapsed_time / self._max_seconds_per_episode if self._max_seconds_per_episode > 0 else 0
            
            # Pro-rated baseline: what we expect to have earned by this point in time
            prorated_baseline_team = progress_ratio * self.solo_baseline_team
            
            # Cumulative team performance so far
            cumulative_team_reward = sum(self.cumulated_pure_rewards[a] for a in self.agents)
            
            # Opportunity cost: how much we're underperforming relative to expectation
            opportunity_cost = max(0, prorated_baseline_team - cumulative_team_reward)
            
            # Distribute penalty proportionally to each agent's solo baseline contribution
            if self.solo_baseline_team > 0:
                agent_solo_baseline = self.solo_baselines.get(agent_id, 0)
                agent_proportion = agent_solo_baseline / self.solo_baseline_team
                agent_penalty = self.eta * opportunity_cost * agent_proportion
                modified_reward -= agent_penalty
        
        self.modified_rewards[agent_id] = modified_reward
        self.cumulated_modified_rewards[agent_id] += self.modified_rewards[agent_id]

    return self.cumulated_pure_rewards, self.cumulated_modified_rewards