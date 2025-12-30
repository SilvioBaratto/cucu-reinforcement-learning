"""PettingZoo-compatible Cucù game environment."""

from typing import Dict, List, Optional, Any
import numpy as np
from pettingzoo import AECEnv
from gymnasium import spaces

from .cards import Deck, Card, CardValue, SpecialEffect


class CucuEnv(AECEnv):
    """
    Cucù card game environment for multi-agent reinforcement learning.
    
    Follows PettingZoo AEC (Agent Environment Cycle) API.
    """
    
    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "cucu_v1",
    }
    
    def __init__(
        self,
        num_players: int = 4,
        starting_lives: int = 3,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        assert 4 <= num_players <= 20, "Cucù requires 4-20 players"
        
        self.num_players = num_players
        self.starting_lives = starting_lives
        self.render_mode = render_mode
        
        # Agent setup
        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        self.agents = self.possible_agents.copy()
        
        # Action space: 0 = STAY, 1 = CUCU/DRAW
        self.action_spaces = {
            agent: spaces.Discrete(2) for agent in self.possible_agents
        }
        
        # Observation space
        self.observation_spaces = {
            agent: spaces.Dict({
                "card_value": spaces.Discrete(11),  # 0 = no card, 1-10 = card values
                "position": spaces.Discrete(num_players),
                "is_dealer": spaces.Discrete(2),
                "players_remaining": spaces.Discrete(num_players + 1),
                "my_lives": spaces.Discrete(starting_lives + 1),
                "actions_this_round": spaces.MultiBinary(num_players),
            })
            for agent in self.possible_agents
        }
        
        self.deck: Deck = Deck()
        self.player_cards: Dict[str, Optional[Card]] = {}
        self.player_lives: Dict[str, int] = {}
        self.actions_this_round: List[int] = []
        self.current_player_idx: int = 0
        self.dealer_idx: int = 0

        # Swap tracking for action masking
        self.original_cards_this_round: Dict[str, int] = {}  # Card value at round start
        self.was_swapped_on: Dict[str, bool] = {}  # Was player swapped on this round
        
        self._cumulative_rewards: Dict[str, float] = {}
        self.terminations: Dict[str, bool] = {}
        self.truncations: Dict[str, bool] = {}
        self.infos: Dict[str, Dict] = {}
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> None:
        """Reset the environment for a new game."""
        if seed is not None:
            np.random.seed(seed)
        
        self.agents = self.possible_agents.copy()
        self.deck = Deck()
        
        self.player_lives = {
            agent: self.starting_lives for agent in self.agents
        }
        self.player_cards = {agent: None for agent in self.agents}
        
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # Deal cards
        self._deal_cards()
        
        # Set starting position (right of dealer, counter-clockwise)
        self.dealer_idx = 0
        self.current_player_idx = (self.dealer_idx - 1) % len(self.agents)
        self.actions_this_round = []
        
        self._agent_selector = iter(self._get_turn_order())
        self.agent_selection = next(self._agent_selector)
    
    def _deal_cards(self) -> None:
        """Deal one card to each player."""
        # Reset deck if not enough cards for all players
        if len(self.deck) < len(self.agents):
            self.deck.reset()
        else:
            self.deck.shuffle()

        # Reset swap tracking for new round
        self.was_swapped_on = {agent: False for agent in self.agents}
        self.original_cards_this_round = {}

        for agent in self.agents:
            card = self.deck.draw()
            self.player_cards[agent] = card
            # Record original card value for this round
            self.original_cards_this_round[agent] = card.value if card else 0
    
    def _get_turn_order(self) -> List[str]:
        """Get turn order: player to RIGHT of dealer starts, dealer is last."""
        n = len(self.agents)
        order = []
        for i in range(1, n + 1):
            idx = (self.dealer_idx + i) % n
            order.append(self.agents[idx])
        return order
    
    def observe(self, agent: str) -> Dict[str, Any]:
        """Get observation for an agent."""
        card = self.player_cards.get(agent)
        card_value = card.value if card else 0

        # My position in the current turn order (0 = first to act)
        turn_order = self._get_turn_order()
        turn_position = turn_order.index(agent) if agent in turn_order else 0

        # Am I the dealer (last to act)?
        is_dealer = 1 if turn_position == len(self.agents) - 1 else 0

        # Actions that happened before me this round
        num_actions_before_me = len(self.actions_this_round)
        num_swaps_before_me = sum(self.actions_this_round)

        # Did the player right before me swap?
        last_action_was_swap = 0
        if num_actions_before_me > 0:
            last_action_was_swap = self.actions_this_round[-1]

        # Ratio of swaps (useful signal)
        swap_ratio = num_swaps_before_me / max(num_actions_before_me, 1)

        # Swap tracking for action masking
        was_swapped_on = 1 if self.was_swapped_on.get(agent, False) else 0
        card_before_swap = self.original_cards_this_round.get(agent, card_value)

        return {
            "card_value": card_value,
            "turn_position": turn_position,
            "is_dealer": is_dealer,
            "players_remaining": len(self.agents),
            "total_players": self.num_players,
            "my_lives": self.player_lives.get(agent, 0),
            "num_swaps_before_me": num_swaps_before_me,
            "num_actions_before_me": num_actions_before_me,
            "last_action_was_swap": last_action_was_swap,
            "swap_ratio": swap_ratio,
            "was_swapped_on": was_swapped_on,
            "card_before_swap": card_before_swap,
        }
    
    def step(self, action: int) -> None:
        """Execute one step in the environment."""
        if not self.agents:
            return  # Game already over

        agent = self.agent_selection

        if agent not in self.terminations:
            return  # Invalid agent

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        # Record action
        self.actions_this_round.append(action)

        # Process action
        if action == 1:  # CUCU or DRAW
            self._process_swap_or_draw(agent)

        # Move to next agent
        try:
            self.agent_selection = next(self._agent_selector)
        except StopIteration:
            # Round complete, resolve
            self._resolve_round()
    
    def _process_swap_or_draw(self, agent: str) -> None:
        """Process a swap request or deck draw."""
        agent_idx = self.agents.index(agent)
        
        if agent_idx == self.dealer_idx:
            # Dealer draws from deck
            new_card = self.deck.cut_and_draw()
            if new_card and new_card.value == CardValue.KING:
                # Instant loss
                self.player_lives[agent] -= 1
                self.infos[agent]["drew_king"] = True
            elif new_card:
                self.player_cards[agent] = new_card
        else:
            # Try to swap with next player
            self._attempt_swap(agent_idx)
    
    def _attempt_swap(self, requester_idx: int) -> None:
        """Attempt to swap cards, handling special cards."""
        if len(self.agents) < 2:
            return  # Can't swap with no other players

        current_idx = (requester_idx + 1) % len(self.agents)
        requester = self.agents[requester_idx]
        visited = set()  # Prevent infinite loop

        while current_idx != requester_idx and current_idx not in visited:
            visited.add(current_idx)
            target = self.agents[current_idx]
            target_card = self.player_cards[target]

            if target_card is None:
                # No card to swap with, move to next
                current_idx = (current_idx + 1) % len(self.agents)
                continue

            if target_card.value == CardValue.KING:
                # Blocked - requester keeps their card
                self.infos[requester]["blocked_by_king"] = True
                return
            elif target_card.value == CardValue.HORSE:
                # Pass to next player
                current_idx = (current_idx + 1) % len(self.agents)
                # Check if we've reached or passed the dealer
                if current_idx == self.dealer_idx or current_idx == requester_idx:
                    # Requester must draw from deck
                    new_card = self.deck.cut_and_draw()
                    if new_card and new_card.value == CardValue.KING:
                        self.player_lives[requester] -= 1
                    elif new_card:
                        self.player_cards[requester] = new_card
                    return
            else:
                # Swap cards
                self.player_cards[requester], self.player_cards[target] = \
                    self.player_cards[target], self.player_cards[requester]
                # Mark target as swapped on (they received requester's card)
                self.was_swapped_on[target] = True
                return
    
    def _resolve_round(self) -> None:
        """Resolve the round - find lowest card(s) and deduct lives."""
        # Find minimum card value among ACTIVE players only
        active_cards = [
            self.player_cards[agent]
            for agent in self.agents
            if self.player_cards.get(agent) is not None
        ]
        min_value = min(card.value for card in active_cards if card is not None)
        
        # All players with minimum lose a life
        for agent in self.agents:
            card = self.player_cards[agent]
            if card and card.value == min_value:
                self.player_lives[agent] -= 1
                self._cumulative_rewards[agent] -= 10
            else:
                self._cumulative_rewards[agent] += 1
        
        # Remove eliminated players
        for agent in self.agents.copy():
            if self.player_lives[agent] <= 0:
                self.terminations[agent] = True
                self._cumulative_rewards[agent] -= 50
                self.agents.remove(agent)
        
        # Check for game end
        if len(self.agents) <= 1:
            if self.agents:
                winner = self.agents[0]
                self._cumulative_rewards[winner] += 50
                self.terminations[winner] = True
            # Clear agents to signal game over
            self.agents = []
            return
        
        # Start new round
        self._deal_cards()
        self.dealer_idx = (self.dealer_idx + 1) % len(self.agents)
        self.current_player_idx = (self.dealer_idx - 1) % len(self.agents)
        self.actions_this_round = []
        self._agent_selector = iter(self._get_turn_order())
        self.agent_selection = next(self._agent_selector)
    
    def render(self) -> Optional[str]:
        """Render the game state."""
        if self.render_mode == "ansi":
            lines = ["=" * 40, "Cucù Game State", "=" * 40]
            for agent in self.possible_agents:
                card = self.player_cards.get(agent)
                lives = self.player_lives.get(agent, 0)
                status = "ELIMINATED" if lives <= 0 else f"Lives: {lives}"
                card_str = str(card.value) if card else "None"
                is_dealer = agent in self.agents and self.agents.index(agent) == self.dealer_idx
                dealer = " (DEALER)" if is_dealer else ""
                lines.append(f"{agent}{dealer}: {card_str} - {status}")
            return "\n".join(lines)
        return None
    
    def close(self) -> None:
        """Clean up resources."""
        pass
