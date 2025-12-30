"""Reinforcement learning agent using PPO."""

from typing import Dict, Any, Optional
import numpy as np
import torch

from .base_agent import BaseAgent
from ..models.actor_critic import ActorCritic


class RLAgent(BaseAgent):
    """
    PPO-based reinforcement learning agent.

    Wraps the ActorCritic model for use in the game environment.
    """

    # Observation dimension: 12 features (expanded for action masking)
    OBS_DIM = 12

    def __init__(
        self,
        agent_id: str,
        obs_dim: int = 12,
        action_dim: int = 2,
        hidden_dim: int = 64,
        device: str = "cpu",
    ):
        super().__init__(agent_id)
        self.device = torch.device(device)
        self.obs_dim = obs_dim

        self.model = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

    def _obs_to_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        """Convert observation dict to tensor."""
        # Normalize all features to roughly [0, 1] range
        obs_vector = np.array([
            observation["card_value"] / 10.0,           # [0] Card value (1-10)
            observation["turn_position"] / 20.0,        # [1] Position in turn order
            observation["is_dealer"],                   # [2] Is dealer (0 or 1)
            observation["players_remaining"] / 20.0,    # [3] Players still in game
            observation["total_players"] / 20.0,        # [4] Total players at table
            observation["my_lives"] / 3.0,              # [5] My remaining lives
            observation["num_swaps_before_me"] / 20.0,  # [6] How many swapped before me
            observation["num_actions_before_me"] / 20.0, # [7] Actions taken before me
            observation["last_action_was_swap"],        # [8] Did previous player swap?
            observation["swap_ratio"],                  # [9] Ratio of swaps so far
            observation.get("was_swapped_on", 0),       # [10] Was I swapped on this round?
            observation.get("card_before_swap", observation["card_value"]) / 10.0,  # [11] Original card
        ], dtype=np.float32)
        return torch.FloatTensor(obs_vector).to(self.device)

    def _compute_action_mask(self, observation: Dict[str, Any]) -> torch.Tensor:
        """
        Compute action mask based on domain knowledge (common knowledge rules).

        Rules:
        1. King (10), Horse (9), Jack (8) - FORCE STAY (card >= 8)
        2. Was swapped on and got better card - FORCE STAY

        Returns:
            Boolean tensor of shape (2,) where True = action is valid
        """
        mask = torch.ones(2, dtype=torch.bool, device=self.device)
        card_value = observation["card_value"]

        # Rule 1: King, Horse, Jack (>= 8) - FORCE STAY
        if card_value >= 8:
            mask[1] = False  # Disable swap action

        # Rule 2: Was swapped on and got better card - FORCE STAY
        if observation.get("was_swapped_on", 0):
            card_before = observation.get("card_before_swap", card_value)
            if card_value > card_before:
                mask[1] = False  # Disable swap action

        return mask

    def select_action(
        self,
        observation: Dict[str, Any],
        deterministic: bool = False,
    ) -> int:
        """Select action using the learned policy with action masking."""
        obs_tensor = self._obs_to_tensor(observation)
        action_mask = self._compute_action_mask(observation)

        with torch.no_grad():
            action_probs, _, _ = self.model(
                obs_tensor.unsqueeze(0),
                action_mask=action_mask.unsqueeze(0),
            )

        if deterministic:
            action = action_probs.squeeze(0).argmax(dim=-1).item()
        else:
            dist = torch.distributions.Categorical(action_probs.squeeze(0))
            action = dist.sample().item()

        return int(action)

    def get_action_and_value(
        self,
        observation: Dict[str, Any],
    ) -> tuple:
        """Get action, log probability, value estimate, and action mask."""
        obs_tensor = self._obs_to_tensor(observation)
        action_mask = self._compute_action_mask(observation)

        action_probs, value, _ = self.model(
            obs_tensor.unsqueeze(0),
            action_mask=action_mask.unsqueeze(0),
        )
        action_probs = action_probs.squeeze(0)
        value = value.squeeze()

        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob, value, action_mask

    def reset(self) -> None:
        """Reset for new episode (no-op for feedforward model)."""
        pass

    def save(self, path: str) -> None:
        """Save model weights."""
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        """Load model weights."""
        state_dict = torch.load(path, map_location=self.device)
        # Handle both direct state dict and checkpoint format
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        self.model.load_state_dict(state_dict)
