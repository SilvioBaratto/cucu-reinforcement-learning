"""Proximal Policy Optimization (PPO) implementation from scratch."""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional
import numpy as np

from .actor_critic import ActorCritic
from .replay_buffer import RolloutBuffer
from .gae import compute_gae


class PPO:
    """
    Proximal Policy Optimization algorithm.
    
    Key components:
    1. Clipped surrogate objective (prevents too large policy updates)
    2. Value function loss
    3. Entropy bonus (encourages exploration)
    """
    
    def __init__(
        self,
        model: ActorCritic,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.05,
        max_grad_norm: float = 0.5,
        epochs_per_update: int = 4,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.epochs_per_update = epochs_per_update
        self.batch_size = batch_size
        
        # Rollout buffer
        self.buffer = RolloutBuffer()
    
    def collect_rollout(
        self,
        obs: torch.Tensor,
        action: int,
        reward: float,
        done: bool,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Store a transition in the rollout buffer."""
        self.buffer.add(
            obs=obs.cpu().numpy().flatten(),
            action=action,
            reward=reward,
            done=done,
            log_prob=log_prob.detach().cpu().item(),
            value=value.detach().cpu().item(),
            action_mask=action_mask.cpu().numpy() if action_mask is not None else None,
        )
    
    def update(self) -> Dict[str, float]:
        """
        Perform PPO update using collected rollouts.
        
        Returns:
            Dictionary of training metrics.
        """
        # Compute returns and advantages
        self.buffer.compute_returns_and_advantages(
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        
        # Get training data
        data = self.buffer.get()

        obs = torch.FloatTensor(data["observations"]).to(self.device)
        actions = torch.LongTensor(data["actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(data["log_probs"]).to(self.device)
        advantages = torch.FloatTensor(data["advantages"]).to(self.device)
        returns = torch.FloatTensor(data["returns"]).to(self.device)

        # Get action masks if available
        action_masks = None
        if "action_masks" in data:
            action_masks = torch.BoolTensor(data["action_masks"]).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Training metrics
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        # Multiple epochs over the data
        for _ in range(self.epochs_per_update):
            # Generate random indices for mini-batches
            indices = np.random.permutation(len(obs))

            for start in range(0, len(obs), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_obs = obs[batch_indices].unsqueeze(1)  # Add seq dim
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Get batch action masks if available
                batch_action_masks = None
                if action_masks is not None:
                    batch_action_masks = action_masks[batch_indices].unsqueeze(1)  # Add seq dim

                # Evaluate actions with action masks
                new_log_probs, values, entropy = self.model.evaluate_actions(
                    batch_obs, batch_actions, action_mask=batch_action_masks
                )
                
                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(values, batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                num_updates += 1
        
        # Clear buffer after update
        self.buffer.clear()
        
        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
        }
    
    def save(self, path: str) -> None:
        """Save model and optimizer state."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str) -> None:
        """Load model and optimizer state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
