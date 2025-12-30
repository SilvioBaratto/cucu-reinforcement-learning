"""Phase 3: Strategy analysis and evaluation."""

import argparse
import os
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import torch

from src.cucu_env import CucuEnv
from src.agents import RandomAgent, ThresholdAgent, RLAgent
from src.models import ActorCritic
from src.multi_agent import MultiAgentWrapper
from src.utils.metrics import (
    compute_win_rate,
    compute_action_distribution,
    compute_swap_threshold,
    compare_agents,
)


def load_trained_agent(model_path: str, device: str = "cpu") -> RLAgent:
    """Load a trained RL agent."""
    agent = RLAgent("player_0", device=device)
    checkpoint = torch.load(model_path, map_location=device)
    agent.model.load_state_dict(checkpoint["model_state_dict"])
    agent.model.eval()
    return agent


def analyze_policy(
    agent: RLAgent,
    num_samples: int = 10000,
) -> Dict:
    """
    Analyze the learned policy.
    
    Returns:
        Dictionary containing policy analysis results
    """
    print("Analyzing learned policy...")
    
    observations = []
    actions = []
    
    # Sample actions for each card value
    for card_value in range(1, 11):
        for _ in range(num_samples // 10):
            obs = {
                "card_value": card_value,
                "turn_position": np.random.randint(0, 4),
                "is_dealer": np.random.randint(0, 2),
                "players_remaining": np.random.randint(2, 5),
                "total_players": 4,
                "my_lives": np.random.randint(1, 4),
                "num_swaps_before_me": np.random.randint(0, 4),
                "num_actions_before_me": np.random.randint(0, 4),
                "last_action_was_swap": np.random.randint(0, 2),
                "swap_ratio": np.random.random(),
                "was_swapped_on": 0,  # Not swapped on for analysis
                "card_before_swap": card_value,  # Same as current card
            }

            action = agent.select_action(obs)
            observations.append(obs)
            actions.append(action)
    
    # Compute statistics
    distribution = compute_action_distribution(observations, actions)
    threshold = compute_swap_threshold(observations, actions)
    
    return {
        "action_distribution": distribution,
        "estimated_threshold": threshold,
    }


def plot_policy(distribution: Dict[int, Dict[int, float]], save_path: Optional[str] = None):
    """Plot action probabilities by card value."""
    card_values = sorted(distribution.keys())
    swap_probs = [distribution[cv].get(1, 0) for cv in card_values]
    stay_probs = [distribution[cv].get(0, 0) for cv in card_values]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(card_values))
    width = 0.35
    
    ax.bar(x - width/2, stay_probs, width, label="Stay", color="green", alpha=0.7)
    ax.bar(x + width/2, swap_probs, width, label="Swap/Draw", color="red", alpha=0.7)
    
    ax.set_xlabel("Card Value")
    ax.set_ylabel("Probability")
    ax.set_title("Learned Policy: Action Probabilities by Card Value")
    ax.set_xticks(x)
    ax.set_xticklabels(["Ace", "2", "3", "4", "5", "6", "7", "Jack", "Horse", "King"])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Policy plot saved to {save_path}")
    else:
        plt.show()


def compare_to_baselines(
    agent: RLAgent,
    num_players: int = 4,
    num_games: int = 5000,
) -> Dict[str, Dict[str, float]]:
    """Compare trained agent to baseline strategies."""
    print(f"\nComparing to baselines over {num_games} games...")
    
    wrapper = MultiAgentWrapper(num_players=num_players)
    
    # Agents to compare
    agents_to_test = {
        "RL Agent": agent,
        "Random": RandomAgent("test"),
        "Threshold-3": ThresholdAgent("test", threshold=3),
        "Threshold-5": ThresholdAgent("test", threshold=5),
        "Threshold-7": ThresholdAgent("test", threshold=7),
    }
    
    results = {}
    
    for name, test_agent in agents_to_test.items():
        print(f"  Testing {name}...")
        
        # Test agent as player_0 vs threshold-5 opponents
        opponents = {
            f"player_{i}": ThresholdAgent(f"player_{i}", threshold=5)
            for i in range(1, num_players)
        }
        
        all_agents = {"player_0": test_agent, **opponents}
        wrapper.set_agents(all_agents)
        
        episode_results = []
        for _ in range(num_games):
            if hasattr(test_agent, "reset"):
                test_agent.reset()
            rewards = wrapper.run_episode()
            episode_results.append(rewards)
        
        win_rate = compute_win_rate(episode_results, "player_0")
        avg_reward = np.mean([r.get("player_0", 0) for r in episode_results])
        
        results[name] = {
            "win_rate": win_rate,
            "avg_reward": avg_reward,
        }
    
    return results


def print_comparison_table(results: Dict[str, Dict[str, float]]):
    """Print comparison results as a table."""
    print("\n" + "=" * 50)
    print("Agent Comparison Results")
    print("=" * 50)
    print(f"{'Agent':<15} {'Win Rate':<12} {'Avg Reward':<12}")
    print("-" * 50)
    
    for agent_name, metrics in sorted(
        results.items(),
        key=lambda x: x[1]["win_rate"],
        reverse=True
    ):
        print(f"{agent_name:<15} {metrics['win_rate']:.4f}       {metrics['avg_reward']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Cucù agent")
    parser.add_argument("--model", type=str, default="models/best_model.pt")
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--games", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Cucù Strategy Analysis")
    print("=" * 60)
    
    # Load trained agent
    if os.path.exists(args.model):
        print(f"\nLoading model from {args.model}")
        agent = load_trained_agent(args.model, args.device)
        
        # Analyze policy
        analysis = analyze_policy(agent)
        
        print(f"\nEstimated swap threshold: {analysis['estimated_threshold']}")
        print("\nAction distribution by card value:")
        for cv, dist in sorted(analysis["action_distribution"].items()):
            swap_prob = dist.get(1, 0)
            print(f"  Card {cv}: P(swap) = {swap_prob:.3f}")
        
        # Plot policy
        os.makedirs(args.output_dir, exist_ok=True)
        plot_policy(
            analysis["action_distribution"],
            save_path=os.path.join(args.output_dir, "policy_plot.png")
        )
        
        # Compare to baselines
        comparison = compare_to_baselines(agent, args.players, args.games)
        print_comparison_table(comparison)
    else:
        print(f"\nNo trained model found at {args.model}")
        print("Running baseline comparison only...")
        
        # Create untrained agent for comparison
        agent = RLAgent("player_0", device=args.device)
        comparison = compare_to_baselines(agent, args.players, args.games)
        print_comparison_table(comparison)


if __name__ == "__main__":
    main()
