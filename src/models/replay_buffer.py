"""Rollout buffer for storing PPO training data."""

from typing import Dict, List, Optional
import numpy as np

from .gae import compute_gae


class RolloutBuffer:
    """
    Buffer for storing rollout data for PPO.
    
    Stores:
    - observations
    - actions
    - rewards
    - dones (episode termination flags)
    - log_probs (for importance sampling ratio)
    - values (for GAE computation)
    
    After collection, computes:
    - advantages (via GAE)
    - returns (discounted cumulative rewards)
    """
    
    def __init__(self):
        self.observations: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.action_masks: List[np.ndarray] = []  # For action masking

        self.advantages: Optional[np.ndarray] = None
        self.returns: Optional[np.ndarray] = None
    
    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
        action_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Add a single transition to the buffer."""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        if action_mask is not None:
            self.action_masks.append(action_mask)
    
    def compute_returns_and_advantages(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        last_value: float = 0.0,
    ) -> None:
        """
        Compute returns and advantages using GAE.
        
        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            last_value: Value estimate for the state after the last transition
        """
        values = np.array(self.values + [last_value])
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        
        # Compute advantages using GAE
        self.advantages = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        
        # Compute returns (advantages + values)
        self.returns = self.advantages + np.array(self.values)
    
    def get(self) -> Dict[str, np.ndarray]:
        """
        Get all data from the buffer.

        Returns:
            Dictionary containing all buffer data as numpy arrays.
        """
        assert self.advantages is not None, "Must compute returns and advantages first"
        assert self.returns is not None, "Must compute returns and advantages first"

        result = {
            "observations": np.array(self.observations),
            "actions": np.array(self.actions),
            "rewards": np.array(self.rewards),
            "dones": np.array(self.dones),
            "log_probs": np.array(self.log_probs),
            "values": np.array(self.values),
            "advantages": self.advantages,
            "returns": self.returns,
        }

        # Include action masks if available
        if self.action_masks:
            result["action_masks"] = np.array(self.action_masks)

        return result
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.action_masks = []
        self.advantages = None
        self.returns = None
    
    def __len__(self) -> int:
        return len(self.observations)
