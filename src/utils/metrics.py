"""Evaluation metrics for Cucù agents."""

from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict


def compute_win_rate(
    episode_rewards: List[Dict[str, float]],
    agent_id: str,
) -> float:
    """
    Compute win rate for a specific agent.
    
    Args:
        episode_rewards: List of reward dicts from multiple episodes
        agent_id: ID of the agent to evaluate
    
    Returns:
        Win rate as a float between 0 and 1
    """
    wins = 0
    total = 0
    
    for rewards in episode_rewards:
        if agent_id not in rewards:
            continue
        
        total += 1
        # Winner has highest reward (50 for winning)
        max_reward = max(rewards.values())
        if rewards[agent_id] == max_reward and max_reward > 0:
            wins += 1
    
    return wins / total if total > 0 else 0.0


def compute_survival_rate(
    episode_rewards: List[Dict[str, float]],
    agent_id: str,
) -> float:
    """
    Compute average survival rate (rounds before elimination).
    
    Args:
        episode_rewards: List of reward dicts from multiple episodes
        agent_id: ID of the agent to evaluate
    
    Returns:
        Average survival rate
    """
    survivals = []
    
    for rewards in episode_rewards:
        if agent_id not in rewards:
            continue
        
        # Estimate rounds survived based on reward
        # Each survival gives +1, each loss gives -10
        reward = rewards[agent_id]
        if reward >= 50:  # Winner
            survivals.append(1.0)
        elif reward <= -50:  # Eliminated
            # Rough estimate of survival
            survivals.append(max(0, (reward + 50) / 10))
        else:
            survivals.append(0.5)
    
    return float(np.mean(survivals)) if survivals else 0.0


def compute_action_distribution(
    observations: List[Dict],
    actions: List[int],
) -> Dict[int, Dict[int, float]]:
    """
    Compute action distribution by card value.
    
    Args:
        observations: List of observation dicts
        actions: List of actions taken
    
    Returns:
        Dictionary mapping card_value -> {action -> probability}
    """
    action_counts = defaultdict(lambda: defaultdict(int))
    
    for obs, action in zip(observations, actions):
        card_value = obs["card_value"]
        action_counts[card_value][action] += 1
    
    distribution = {}
    for card_value, counts in action_counts.items():
        total = sum(counts.values())
        distribution[card_value] = {
            action: count / total
            for action, count in counts.items()
        }
    
    return dict(distribution)


def compute_swap_threshold(
    observations: List[Dict],
    actions: List[int],
    threshold_prob: float = 0.5,
) -> float:
    """
    Estimate the effective swap threshold from action data.
    
    Args:
        observations: List of observation dicts
        actions: List of actions taken
        threshold_prob: Probability threshold for considering it a "swap" card
    
    Returns:
        Estimated card value threshold below which agent swaps
    """
    distribution = compute_action_distribution(observations, actions)
    
    threshold = 0
    for card_value in sorted(distribution.keys()):
        swap_prob = distribution[card_value].get(1, 0)
        if swap_prob >= threshold_prob:
            threshold = card_value
        else:
            break
    
    return threshold


def compare_agents(
    episode_results: List[Dict[str, float]],
    agent_ids: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple agents.
    
    Returns:
        Dictionary mapping agent_id -> {metric -> value}
    """
    comparison = {}
    
    for agent_id in agent_ids:
        comparison[agent_id] = {
            "win_rate": compute_win_rate(episode_results, agent_id),
            "survival_rate": compute_survival_rate(episode_results, agent_id),
            "avg_reward": np.mean([
                r.get(agent_id, 0) for r in episode_results
            ]),
        }
    
    return comparison
