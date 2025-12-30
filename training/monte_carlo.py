"""Phase 1: Monte Carlo simulations for baseline strategy discovery."""

import argparse
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm
import numpy as np

from src.cucu_env import CucuEnv
from src.agents import RandomAgent, ThresholdAgent
from src.multi_agent import MultiAgentWrapper
from src.utils.metrics import compute_win_rate, compare_agents


def run_monte_carlo_simulation(
    num_games: int = 100000,
    num_players: int = 4,
    threshold_range: Tuple[int, int] = (2, 8),
) -> Dict[int, Dict[str, float]]:
    """
    Run Monte Carlo simulations to find optimal threshold.
    
    Tests different threshold values and measures win rates.
    
    Args:
        num_games: Number of games per threshold
        num_players: Number of players in each game
        threshold_range: Range of thresholds to test
    
    Returns:
        Results for each threshold value
    """
    results = {}
    
    for threshold in range(threshold_range[0], threshold_range[1] + 1):
        print(f"\nTesting threshold = {threshold}")
        
        wrapper = MultiAgentWrapper(num_players=num_players)
        
        # Set up agents: one threshold agent vs random agents
        agents = {}
        test_agent_id = "player_0"
        
        for i in range(num_players):
            agent_id = f"player_{i}"
            if i == 0:
                agents[agent_id] = ThresholdAgent(agent_id, threshold=threshold)
            else:
                agents[agent_id] = RandomAgent(agent_id)
        
        wrapper.set_agents(agents)
        
        # Run games
        episode_results = []
        for _ in tqdm(range(num_games), desc=f"Threshold {threshold}"):
            rewards = wrapper.run_episode()
            episode_results.append(rewards)
        
        # Compute metrics
        win_rate = compute_win_rate(episode_results, test_agent_id)
        avg_reward = np.mean([r.get(test_agent_id, 0) for r in episode_results])
        
        results[threshold] = {
            "win_rate": win_rate,
            "avg_reward": avg_reward,
        }
        
        print(f"  Win rate: {win_rate:.4f}")
        print(f"  Avg reward: {avg_reward:.2f}")
    
    return results


def find_optimal_threshold(results: Dict[int, Dict[str, float]]) -> int:
    """Find the threshold with highest win rate."""
    best_threshold = max(results.keys(), key=lambda t: results[t]["win_rate"])
    return best_threshold


def test_position_effect(
    num_games: int = 50000,
    num_players: int = 6,
    threshold: int = 5,
) -> Dict[int, float]:
    """
    Test how position affects win rate.
    
    Args:
        num_games: Number of games
        num_players: Number of players
        threshold: Threshold to use for all agents
    
    Returns:
        Win rate by position
    """
    print(f"\nTesting position effect with {num_players} players...")
    
    wrapper = MultiAgentWrapper(num_players=num_players)
    
    # All threshold agents
    agents = {
        f"player_{i}": ThresholdAgent(f"player_{i}", threshold=threshold)
        for i in range(num_players)
    }
    wrapper.set_agents(agents)
    
    # Run games
    episode_results = []
    for _ in tqdm(range(num_games)):
        rewards = wrapper.run_episode()
        episode_results.append(rewards)
    
    # Compute win rate by position
    position_wins = defaultdict(int)
    position_total = defaultdict(int)
    
    for rewards in episode_results:
        max_reward = max(rewards.values())
        for agent_id, reward in rewards.items():
            position = int(agent_id.split("_")[1])
            position_total[position] += 1
            if reward == max_reward and max_reward > 0:
                position_wins[position] += 1
    
    position_win_rates = {
        pos: position_wins[pos] / position_total[pos]
        for pos in range(num_players)
    }
    
    print("\nWin rates by position:")
    for pos, rate in sorted(position_win_rates.items()):
        print(f"  Position {pos}: {rate:.4f}")
    
    return position_win_rates


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo simulations for Cucù")
    parser.add_argument("--games", type=int, default=10000, help="Games per test")
    parser.add_argument("--players", type=int, default=4, help="Number of players")
    parser.add_argument("--test-position", action="store_true", help="Test position effect")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Cucù Monte Carlo Simulation")
    print("=" * 60)
    
    # Find optimal threshold
    results = run_monte_carlo_simulation(
        num_games=args.games,
        num_players=args.players,
    )
    
    optimal = find_optimal_threshold(results)
    print(f"\n{'=' * 60}")
    print(f"OPTIMAL THRESHOLD: {optimal}")
    print(f"Win rate: {results[optimal]['win_rate']:.4f}")
    print(f"{'=' * 60}")
    
    # Test position effect
    if args.test_position:
        test_position_effect(
            num_games=args.games,
            num_players=args.players,
            threshold=optimal,
        )


if __name__ == "__main__":
    main()
