"""Actor-Critic network for PPO training."""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class ActorCritic(nn.Module):
    """
    Actor-Critic network with feedforward architecture.

    Simplified from LSTM-based version because:
    1. Cucù observations capture most relevant game state
    2. Feedforward networks train more stably with PPO
    3. Avoids policy collapse issues seen with LSTM

    Architecture:
        Observation -> Shared MLP -> Actor (policy) head
                                  -> Critic (value) head
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,  # Kept for backward compatibility, ignored
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head: outputs action logits (softmax applied separately for masking)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

        # Critic head: estimates state value V(s)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for name, param in self.named_parameters():
            if "weight" in name:
                # Use larger gain for better gradient flow
                nn.init.orthogonal_(param, gain=1.0)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through the network.

        Args:
            obs: Observation tensor of shape (batch, seq_len, obs_dim) or (batch, obs_dim)
            hidden: Ignored (kept for backward compatibility)
            action_mask: Boolean tensor of valid actions (True = valid, False = masked)

        Returns:
            action_probs: Action probabilities (masked actions have ~0 probability)
            values: State values
            hidden: None (no recurrent state)
        """
        # Handle sequence dimension if present
        if obs.dim() == 3:
            batch_size, seq_len, obs_dim = obs.shape
            obs = obs.view(-1, obs_dim)
            features = self.shared(obs)
            action_logits = self.actor(features).view(batch_size, seq_len, -1)
            values = self.critic(features).view(batch_size, seq_len, -1)
        else:
            features = self.shared(obs)
            action_logits = self.actor(features)
            values = self.critic(features)

        # Apply action mask before softmax
        if action_mask is not None:
            # Set masked actions to very negative logits
            action_logits = action_logits.masked_fill(~action_mask, float('-inf'))

        action_probs = torch.softmax(action_logits, dim=-1)

        return action_probs, values, None

    def get_action(
        self,
        obs: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """
        Sample an action from the policy.

        Args:
            obs: Observation tensor
            hidden: Ignored (kept for backward compatibility)
            deterministic: If True, return argmax action
            action_mask: Boolean tensor of valid actions (True = valid)

        Returns:
            action: Sampled action
            log_prob: Log probability of the action
            value: State value estimate
            entropy: Policy entropy
            hidden: None (no recurrent state)
        """
        action_probs, values, _ = self.forward(obs, hidden, action_mask)

        # Handle sequence dimension
        if action_probs.dim() == 3:
            action_probs = action_probs[:, -1, :]
            values = values[:, -1, :]

        # Create categorical distribution
        dist = torch.distributions.Categorical(action_probs)

        if deterministic:
            action = action_probs.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, values.squeeze(-1), entropy, None

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.

        Args:
            obs: Observation tensor
            actions: Actions taken
            hidden: Ignored (kept for backward compatibility)
            action_mask: Boolean tensor of valid actions (True = valid)

        Returns:
            log_probs: Log probabilities of the actions
            values: State value estimates
            entropy: Policy entropy
        """
        action_probs, values, _ = self.forward(obs, hidden, action_mask)

        # Handle sequence dimension
        if action_probs.dim() == 3:
            action_probs = action_probs[:, -1, :]
            values = values[:, -1, :]

        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values.squeeze(-1), entropy
