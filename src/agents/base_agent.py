"""Abstract base class for all agents."""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseAgent(ABC):
    """Base class for Cucù agents."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
    
    @abstractmethod
    def select_action(self, observation: Dict[str, Any]) -> int:
        """
        Select an action given the current observation.
        
        Args:
            observation: Dictionary containing:
                - card_value: int (1-10)
                - position: int
                - is_dealer: int (0 or 1)
                - players_remaining: int
                - my_lives: int
                - actions_this_round: np.ndarray
        
        Returns:
            Action: 0 = STAY, 1 = CUCU/DRAW
        """
        pass
    
    def reset(self) -> None:
        """Reset agent state for a new episode."""
        pass
    
    def update(self, *args, **kwargs) -> None:
        """Update agent (for learning agents)."""
        pass
