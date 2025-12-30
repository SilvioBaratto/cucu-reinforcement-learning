"""Threshold-based heuristic agent."""

from typing import Dict, Any

from .base_agent import BaseAgent


class ThresholdAgent(BaseAgent):
    """
    Agent that swaps if card value is below a threshold.
    
    This is a simple but effective heuristic baseline.
    """
    
    def __init__(
        self,
        agent_id: str,
        threshold: int = 5,
        position_aware: bool = False,
    ):
        super().__init__(agent_id)
        self.threshold = threshold
        self.position_aware = position_aware
    
    def select_action(self, observation: Dict[str, Any]) -> int:
        """
        Swap if card is below threshold, respecting common knowledge rules.

        If position_aware, adjust threshold based on position:
        - Earlier positions need lower threshold (less info)
        - Later positions can have higher threshold (more info)
        """
        card_value = observation["card_value"]

        # Common knowledge rule 1: Never swap King, Horse, Jack (>= 8)
        if card_value >= 8:
            return 0  # Stay

        # Common knowledge rule 2: If swapped on and got better card, stay
        if observation.get("was_swapped_on", 0):
            card_before = observation.get("card_before_swap", card_value)
            if card_value > card_before:
                return 0  # Stay

        if self.position_aware:
            position = observation.get("turn_position", observation.get("position", 0))
            players_remaining = observation["players_remaining"]
            # Adjust threshold: later positions can be more conservative
            position_ratio = position / max(players_remaining - 1, 1)
            adjusted_threshold = self.threshold + int(position_ratio * 2)
            threshold = min(adjusted_threshold, 7)  # Cap at 7 (below Jack)
        else:
            threshold = self.threshold

        return 1 if card_value < threshold else 0


class PositionAwareAgent(ThresholdAgent):
    """Convenience class for position-aware threshold agent."""
    
    def __init__(self, agent_id: str, base_threshold: int = 5):
        super().__init__(
            agent_id=agent_id,
            threshold=base_threshold,
            position_aware=True,
        )
