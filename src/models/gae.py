"""Generalized Advantage Estimation (GAE) implementation."""

import numpy as np


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> np.ndarray:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    GAE balances bias vs variance in advantage estimation:
    - λ=0: High bias, low variance (TD(0))
    - λ=1: Low bias, high variance (Monte Carlo)
    - λ=0.95: Good balance for most tasks
    
    Formula:
        A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
        where δ_t = r_t + γV(s_{t+1}) - V(s_t)
    
    Args:
        rewards: Array of rewards (T,)
        values: Array of value estimates (T+1,) - includes bootstrap value
        dones: Array of done flags (T,)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
    
    Returns:
        advantages: Array of advantage estimates (T,)
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    
    for t in reversed(range(T)):
        # Mask for non-terminal states
        non_terminal = 1.0 - float(dones[t])
        
        # TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * values[t + 1] * non_terminal - values[t]
        
        # GAE: A_t = δ_t + (γλ)(1-d_t)A_{t+1}
        gae = delta + gamma * gae_lambda * non_terminal * gae
        advantages[t] = gae
    
    return advantages


def compute_returns(
    rewards: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    last_value: float = 0.0,
) -> np.ndarray:
    """
    Compute discounted returns (Monte Carlo style).
    
    Args:
        rewards: Array of rewards (T,)
        dones: Array of done flags (T,)
        gamma: Discount factor
        last_value: Bootstrap value for incomplete episodes
    
    Returns:
        returns: Array of discounted returns (T,)
    """
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float32)
    running_return = last_value
    
    for t in reversed(range(T)):
        if dones[t]:
            running_return = 0.0
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return
    
    return returns
