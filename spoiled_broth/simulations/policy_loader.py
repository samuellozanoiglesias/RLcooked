"""
Policy loader for simulation runs.

Loads MultiRLModule policies from RLlib checkpoints for inference during simulations.
This module mirrors the checkpoint loading logic from rllib_controller.py but exposes
a clean dictionary of callable policy modules keyed by agent_id.

Author: Samuel Lozano
"""

import os
from pathlib import Path
from typing import Dict, Optional, Any

import torch
import numpy as np
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule


class PolicyInferrer:
    """Wraps a single RLlib policy module for inference."""

    def __init__(self, policy_module, action_space_size: int):
        self.policy_module = policy_module
        self.action_space_size = action_space_size

    def get_action(self, obs: np.ndarray) -> int:
        """Sample an action from the policy given an observation vector."""
        input_dict = {"obs": torch.tensor(obs, dtype=torch.float32)}
        action_output = self.policy_module.forward_inference(input_dict)
        action_logits = action_output["action_dist_inputs"]
        action_dist_class = self.policy_module.get_inference_action_dist_cls()
        action_dist = action_dist_class.from_logits(action_logits)
        return action_dist.sample().item()


def load_policies(
    num_agents: int,
    checkpoint_dir: Path,
    game_version: str,
    action_space_size: int,
    custom_checkpoints: Optional[Dict[str, Any]] = None,
) -> Dict[str, PolicyInferrer]:
    """
    Load RL policies for all agents from checkpoint directories.

    Args:
        num_agents: Number of agents to load policies for.
        checkpoint_dir: Default checkpoint directory.
        game_version: Game version string (used to detect competition mode).
        action_space_size: Size of the flat action space.
        custom_checkpoints: Optional per-agent override dict. Format:
            {"ai_rl_1": {"loaded_agent_id": "policy_ai_rl_1",
                         "checkpoint_number": "final",
                         "path": "/path/to/training"}, ...}

    Returns:
        Dict mapping agent_id -> PolicyInferrer.
    """
    policies: Dict[str, PolicyInferrer] = {}

    for i in range(1, num_agents + 1):
        agent_id = f"ai_rl_{i}"

        # Resolve checkpoint directory and policy id for this agent
        if custom_checkpoints and agent_id in custom_checkpoints and custom_checkpoints[agent_id] is not None:
            info = custom_checkpoints[agent_id]
            chk_path = Path(info["path"]) / f"checkpoint_{info['checkpoint_number']}"
            policy_id = info["loaded_agent_id"]
        else:
            chk_path = checkpoint_dir
            policy_id = f"policy_{agent_id}"

        rl_module_path = os.path.join(
            chk_path,
            "learner_group",
            "learner",
            "rl_module",
        )

        try:
            multi_module = MultiRLModule.from_checkpoint(rl_module_path)
            if policy_id not in multi_module.keys():
                raise ValueError(
                    f"Policy '{policy_id}' not found in checkpoint. "
                    f"Available: {list(multi_module.keys())}"
                )
            policy_module = multi_module[policy_id]
            policies[agent_id] = PolicyInferrer(policy_module, action_space_size)
            print(f"Loaded policy '{policy_id}' for {agent_id} from {chk_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load policy for {agent_id}: {e}") from e

    return policies
