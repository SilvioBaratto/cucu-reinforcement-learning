"""Card definitions and deck management for Cucù."""

from dataclasses import dataclass
from enum import IntEnum
from typing import List
import random


class CardValue(IntEnum):
    """Card values from lowest to highest."""
    ACE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    JACK = 8
    HORSE = 9
    KING = 10


class SpecialEffect:
    """Special card effects."""
    BLOCK = "block"  # King: blocks swap
    PASS = "pass"    # Horse: passes swap to next player


SPECIAL_CARDS = {
    CardValue.KING: SpecialEffect.BLOCK,
    CardValue.HORSE: SpecialEffect.PASS,
}


@dataclass
class Card:
    """Represents a single card."""
    value: CardValue
    suit: str  # "coins", "cups", "swords", "clubs"
    
    @property
    def is_special(self) -> bool:
        return self.value in SPECIAL_CARDS
    
    @property
    def special_effect(self) -> str | None:
        return SPECIAL_CARDS.get(self.value)
    
    def __lt__(self, other: "Card") -> bool:
        return self.value < other.value
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return False
        return self.value == other.value
    
    def __repr__(self) -> str:
        return f"Card({self.value.name}, {self.suit})"


class Deck:
    """A Sicilian/Neapolitan deck of 40 cards."""
    
    SUITS = ["coins", "cups", "swords", "clubs"]
    
    def __init__(self):
        self.cards: List[Card] = []
        self.reset()
    
    def reset(self) -> None:
        """Reset deck with all 40 cards."""
        self.cards = [
            Card(value, suit)
            for suit in self.SUITS
            for value in CardValue
        ]
        self.shuffle()
    
    def shuffle(self) -> None:
        """Shuffle the deck."""
        random.shuffle(self.cards)
    
    def draw(self) -> Card | None:
        """Draw a card from the top of the deck."""
        if self.cards:
            return self.cards.pop()
        return None
    
    def cut_and_draw(self) -> Card | None:
        """Cut the deck and draw the top card (dealer action)."""
        if not self.cards:
            return None
        cut_point = random.randint(0, len(self.cards) - 1)
        self.cards = self.cards[cut_point:] + self.cards[:cut_point]
        return self.draw()
    
    def __len__(self) -> int:
        return len(self.cards)
