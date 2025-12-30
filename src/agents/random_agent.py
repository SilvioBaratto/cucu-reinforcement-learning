"""Random baseline agent."""

import random
from typing import Dict, Any

from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Agent that randomly chooses to stay or swap, respecting common knowledge rules."""

    def __init__(self, agent_id: str, swap_probability: float = 0.5):
        super().__init__(agent_id)
        self.swap_probability = swap_probability

    def select_action(self, observation: Dict[str, Any]) -> int:
        """Randomly select stay (0) or swap (1), with common knowledge constraints."""
        card_value = observation["card_value"]

        # Common knowledge rule 1: Never swap King, Horse, Jack (>= 8)
        if card_value >= 8:
            return 0  # Stay

        # Common knowledge rule 2: If swapped on and got better card, stay
        if observation.get("was_swapped_on", 0):
            card_before = observation.get("card_before_swap", card_value)
            if card_value > card_before:
                return 0  # Stay

        # Otherwise, random choice
        return 1 if random.random() < self.swap_probability else 0
