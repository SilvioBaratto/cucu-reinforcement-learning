"""
Strategy Analysis: Extract optimal Cucù strategies from trained PPO agent.

Analyzes the learned policy to determine:
1. Optimal swap thresholds by card value
2. Position effects (early vs late, dealer vs non-dealer)
3. Table size effects (4, 6, 8 players)
4. Life management strategies
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents import RLAgent
from src.models import ActorCritic


CARD_NAMES = {
    1: "Ace (Cucù)",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "Jack",
    9: "Horse",
    10: "King",
}


def load_model(model_path: str, device: str = "cpu") -> RLAgent:
    """Load trained RL agent."""
    agent = RLAgent("analysis", device=device)
    checkpoint = torch.load(model_path, map_location=device)
    agent.model.load_state_dict(checkpoint["model_state_dict"])
    agent.model.eval()
    return agent


def get_swap_probability(
    agent: RLAgent,
    card_value: int,
    turn_position: int,
    is_dealer: int,
    players_remaining: int,
    total_players: int,
    my_lives: int,
    num_swaps_before: int = 0,
    num_actions_before: int = 0,
    last_action_swap: int = 0,
    swap_ratio: float = 0.0,
    was_swapped_on: int = 0,
    card_before_swap: Optional[int] = None,
) -> float:
    """Get swap probability for a given game state."""
    if card_before_swap is None:
        card_before_swap = card_value

    obs = {
        "card_value": card_value,
        "turn_position": turn_position,
        "is_dealer": is_dealer,
        "players_remaining": players_remaining,
        "total_players": total_players,
        "my_lives": my_lives,
        "num_swaps_before_me": num_swaps_before,
        "num_actions_before_me": num_actions_before,
        "last_action_was_swap": last_action_swap,
        "swap_ratio": swap_ratio,
        "was_swapped_on": was_swapped_on,
        "card_before_swap": card_before_swap,
    }

    obs_tensor = agent._obs_to_tensor(obs)
    action_mask = agent._compute_action_mask(obs)

    with torch.no_grad():
        action_probs, _, _ = agent.model(
            obs_tensor.unsqueeze(0),
            action_mask=action_mask.unsqueeze(0),
        )

    probs = action_probs.squeeze(0).cpu().numpy()
    return probs[1]  # Probability of swap action


def analyze_by_card_value(agent: RLAgent, total_players: int = 4) -> Dict[int, float]:
    """Analyze swap probability for each card value."""
    results = {}

    for card_value in range(1, 11):
        # Average over different positions and game states
        swap_probs = []
        for turn_pos in range(total_players):
            is_dealer = 1 if turn_pos == total_players - 1 else 0
            for lives in [1, 2, 3]:
                prob = get_swap_probability(
                    agent,
                    card_value=card_value,
                    turn_position=turn_pos,
                    is_dealer=is_dealer,
                    players_remaining=total_players,
                    total_players=total_players,
                    my_lives=lives,
                )
                swap_probs.append(prob)

        results[card_value] = np.mean(swap_probs)

    return results


def analyze_by_position(agent: RLAgent, total_players: int = 4) -> Dict[str, Dict[int, float]]:
    """Analyze swap probability by position (early, middle, late, dealer)."""
    results = {"early": {}, "middle": {}, "late": {}, "dealer": {}}

    for card_value in range(1, 8):  # Only swappable cards
        for position_type in results.keys():
            probs = []

            if position_type == "early":
                positions = [0] if total_players <= 4 else [0, 1]
            elif position_type == "middle":
                positions = [1] if total_players <= 4 else [2, 3]
            elif position_type == "late":
                positions = [total_players - 2]
            else:  # dealer
                positions = [total_players - 1]

            for pos in positions:
                is_dealer = 1 if pos == total_players - 1 else 0
                prob = get_swap_probability(
                    agent,
                    card_value=card_value,
                    turn_position=pos,
                    is_dealer=is_dealer,
                    players_remaining=total_players,
                    total_players=total_players,
                    my_lives=3,
                    num_actions_before=pos,
                )
                probs.append(prob)

            results[position_type][card_value] = np.mean(probs)

    return results


def analyze_by_lives(agent: RLAgent, total_players: int = 4) -> Dict[int, Dict[int, float]]:
    """Analyze how remaining lives affect swap decisions."""
    results = {1: {}, 2: {}, 3: {}}

    for lives in [1, 2, 3]:
        for card_value in range(1, 8):
            probs = []
            for pos in range(total_players):
                is_dealer = 1 if pos == total_players - 1 else 0
                prob = get_swap_probability(
                    agent,
                    card_value=card_value,
                    turn_position=pos,
                    is_dealer=is_dealer,
                    players_remaining=total_players,
                    total_players=total_players,
                    my_lives=lives,
                )
                probs.append(prob)
            results[lives][card_value] = np.mean(probs)

    return results


def analyze_swap_context(agent: RLAgent, total_players: int = 4) -> Dict[str, Dict[int, float]]:
    """Analyze how others' swaps affect decisions."""
    results = {"no_swaps": {}, "many_swaps": {}}

    for card_value in range(1, 8):
        # No swaps before
        prob_no_swaps = get_swap_probability(
            agent,
            card_value=card_value,
            turn_position=2,
            is_dealer=0,
            players_remaining=total_players,
            total_players=total_players,
            my_lives=3,
            num_swaps_before=0,
            num_actions_before=2,
            swap_ratio=0.0,
        )
        results["no_swaps"][card_value] = prob_no_swaps

        # Many swaps before
        prob_many_swaps = get_swap_probability(
            agent,
            card_value=card_value,
            turn_position=2,
            is_dealer=0,
            players_remaining=total_players,
            total_players=total_players,
            my_lives=3,
            num_swaps_before=2,
            num_actions_before=2,
            swap_ratio=1.0,
        )
        results["many_swaps"][card_value] = prob_many_swaps

    return results


def estimate_threshold(swap_probs: Dict[int, float], cutoff: float = 0.5) -> int:
    """Estimate effective swap threshold from probabilities."""
    for card in range(1, 11):
        if swap_probs.get(card, 0) < cutoff:
            return card
    return 10


def print_analysis(
    agent: RLAgent,
    total_players: int,
    title: str,
):
    """Print comprehensive analysis for a table size."""
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")

    # By card value
    print("\n📊 SWAP PROBABILITY BY CARD VALUE")
    print("-" * 50)
    card_probs = analyze_by_card_value(agent, total_players)
    for card, prob in sorted(card_probs.items()):
        bar = "█" * int(prob * 20)
        marker = "← ALWAYS STAY (action masked)" if card >= 8 else ""
        print(f"  {CARD_NAMES[card]:12s}: {prob:5.1%} {bar} {marker}")

    threshold = estimate_threshold(card_probs)
    print(f"\n  ➤ Effective Threshold: SWAP if card < {threshold} ({CARD_NAMES.get(threshold, 'N/A')})")

    # By position
    print("\n📍 SWAP PROBABILITY BY POSITION (cards 1-7)")
    print("-" * 50)
    position_probs = analyze_by_position(agent, total_players)
    for pos_type in ["early", "middle", "late", "dealer"]:
        avg_prob = np.mean(list(position_probs[pos_type].values()))
        print(f"  {pos_type.capitalize():8s}: {avg_prob:5.1%}")

    # By lives
    print("\n❤️  SWAP PROBABILITY BY LIVES REMAINING (cards 1-7)")
    print("-" * 50)
    lives_probs = analyze_by_lives(agent, total_players)
    for lives in [3, 2, 1]:
        avg_prob = np.mean(list(lives_probs[lives].values()))
        hearts = "❤️ " * lives
        print(f"  {lives} lives {hearts}: {avg_prob:5.1%}")

    # Context effects
    print("\n🔄 EFFECT OF OTHERS' SWAPS (cards 1-7)")
    print("-" * 50)
    context_probs = analyze_swap_context(agent, total_players)
    avg_no_swaps = np.mean(list(context_probs["no_swaps"].values()))
    avg_many_swaps = np.mean(list(context_probs["many_swaps"].values()))
    print(f"  No one swapped before:    {avg_no_swaps:5.1%}")
    print(f"  Everyone swapped before:  {avg_many_swaps:5.1%}")
    diff = avg_many_swaps - avg_no_swaps
    if abs(diff) > 0.05:
        direction = "MORE" if diff > 0 else "LESS"
        print(f"  ➤ Agent swaps {direction} when others have swapped ({abs(diff):+.1%})")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Cucù strategy from trained model")
    parser.add_argument("--model", type=str, default="models/best_model.pt")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        return

    print("Loading trained model...")
    agent = load_model(args.model, args.device)

    # Analyze for each table size
    for players in [4, 6, 8]:
        print_analysis(agent, players, f"ANALYSIS FOR {players} PLAYERS")



if __name__ == "__main__":
    main()
