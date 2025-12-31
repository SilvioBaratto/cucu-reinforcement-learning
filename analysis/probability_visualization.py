"""
Probability Visualization for Cucù Card Game.

Creates animated MP4 videos showing how winning probabilities evolve
during a game based on the number of lives each player has.

Inspired by the bingo-probability-analysis visualization approach.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from functools import lru_cache


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class GameConfig:
    """Configuration for Cucù game visualization."""
    num_players: int = 4
    initial_lives: int = 3
    num_values: int = 10
    copies_per_value: int = 4

    @property
    def total_cards(self) -> int:
        return self.num_values * self.copies_per_value


@dataclass
class PlayerState:
    """State of a single player."""
    player_id: int
    lives: int
    is_active: bool = True
    color: str = "blue"

    def lose_life(self) -> None:
        self.lives -= 1
        if self.lives <= 0:
            self.is_active = False


@dataclass
class GameState:
    """Complete game state at a moment in time."""
    round_num: int
    players: List[PlayerState]
    cards_dealt: Optional[Dict[int, int]] = None
    loser_this_round: Optional[List[int]] = None

    @property
    def active_players(self) -> List[PlayerState]:
        return [p for p in self.players if p.is_active]

    @property
    def num_active(self) -> int:
        return len(self.active_players)

    def is_game_over(self) -> bool:
        return self.num_active <= 1


@dataclass
class GameHistory:
    """History of a complete game."""
    config: GameConfig
    states: List[GameState] = field(default_factory=list)
    probabilities: List[Dict[int, float]] = field(default_factory=list)
    winner: Optional[int] = None


# =============================================================================
# PROBABILITY CALCULATIONS
# =============================================================================

class ProbabilityCalculator:
    """Calculate winning probabilities based on game state."""

    def __init__(self, config: GameConfig):
        self.config = config

    @lru_cache(maxsize=1000)
    def prob_win_from_state(
        self,
        my_lives: int,
        opponent_lives_tuple: Tuple[int, ...],
    ) -> float:
        """
        Calculate probability of winning from a given state.

        Uses dynamic programming / recursive calculation.

        Args:
            my_lives: Number of lives I have
            opponent_lives_tuple: Tuple of opponent lives

        Returns:
            Probability of winning
        """
        opponent_lives = list(opponent_lives_tuple)

        # Base cases
        if my_lives <= 0:
            return 0.0

        active_opponents = [l for l in opponent_lives if l > 0]
        if len(active_opponents) == 0:
            return 1.0

        # Total active players including myself
        num_active = 1 + len(active_opponents)

        # In each round, one random player loses a life
        # Probability I lose = 1/num_active
        # Probability opponent i loses = 1/num_active

        p_i_lose = 1.0 / num_active

        # Expected value calculation
        # P(win) = P(I lose) * P(win | I lose) + sum(P(opponent j loses) * P(win | j loses))

        total_prob = 0.0

        # Case 1: I lose a life
        new_my_lives = my_lives - 1
        if new_my_lives > 0:
            total_prob += p_i_lose * self.prob_win_from_state(
                new_my_lives, tuple(opponent_lives)
            )
        # If I die, P(win) = 0, which adds nothing

        # Case 2: Each opponent loses a life
        for i, _ in enumerate(active_opponents):
            new_opp_lives = opponent_lives.copy()
            # Find and update the correct opponent
            idx = 0
            for j, l in enumerate(new_opp_lives):
                if l > 0:
                    if idx == i:
                        new_opp_lives[j] -= 1
                        break
                    idx += 1

            total_prob += p_i_lose * self.prob_win_from_state(
                my_lives, tuple(new_opp_lives)
            )

        return total_prob

    def calculate_all_probabilities(self, state: GameState) -> Dict[int, float]:
        """Calculate win probability for each active player."""
        probabilities = {}

        active_players = [p for p in state.players if p.is_active]

        for player in active_players:
            # Get opponent lives
            opponent_lives = [
                p.lives for p in active_players if p.player_id != player.player_id
            ]

            prob = self.prob_win_from_state(player.lives, tuple(opponent_lives))
            probabilities[player.player_id] = prob

        # Inactive players have 0 probability
        for player in state.players:
            if not player.is_active:
                probabilities[player.player_id] = 0.0

        return probabilities


# =============================================================================
# GAME SIMULATION
# =============================================================================

class GameSimulator:
    """Simulate Cucù games with probability tracking."""

    def __init__(self, config: GameConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.prob_calculator = ProbabilityCalculator(config)

    def create_initial_state(self) -> GameState:
        """Create initial game state."""
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i) for i in range(self.config.num_players)]
        players = [
            PlayerState(
                player_id=i,
                lives=self.config.initial_lives,
                color=f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"
            )
            for i, c in enumerate(colors)
        ]
        return GameState(round_num=0, players=players)

    def deal_cards(self, state: GameState) -> Dict[int, int]:
        """Deal cards to active players."""
        active_ids = [p.player_id for p in state.active_players]
        deck = list(range(1, self.config.num_values + 1)) * self.config.copies_per_value
        self.rng.shuffle(deck)
        return {pid: deck[i] for i, pid in enumerate(active_ids)}

    def play_round(self, state: GameState) -> GameState:
        """Play a single round."""
        # Deal cards
        cards = self.deal_cards(state)

        # Find minimum card
        min_card = min(cards.values())

        # All players with min card lose a life
        losers = [pid for pid, card in cards.items() if card == min_card]

        # Create new player states
        new_players = []
        for p in state.players:
            new_p = PlayerState(
                player_id=p.player_id,
                lives=p.lives,
                is_active=p.is_active,
                color=p.color
            )
            if p.player_id in losers:
                new_p.lose_life()
            new_players.append(new_p)

        return GameState(
            round_num=state.round_num + 1,
            players=new_players,
            cards_dealt=cards,
            loser_this_round=losers
        )

    def simulate_game(self) -> GameHistory:
        """Simulate a complete game with probability tracking."""
        history = GameHistory(config=self.config)

        state = self.create_initial_state()
        history.states.append(state)
        history.probabilities.append(
            self.prob_calculator.calculate_all_probabilities(state)
        )

        while not state.is_game_over():
            state = self.play_round(state)
            history.states.append(state)
            history.probabilities.append(
                self.prob_calculator.calculate_all_probabilities(state)
            )

        # Determine winner
        winners = [p.player_id for p in state.active_players]
        history.winner = winners[0] if winners else None

        return history


# =============================================================================
# VISUALIZATION
# =============================================================================

class ProbabilityVisualizer:
    """Create animated visualizations of game probability evolution."""

    def __init__(self, output_dir: Path = Path("output")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _setup_figure(self) -> Tuple[Any, Any, Any]:
        """Set up the figure with probability plot and game visualization."""
        fig = plt.figure(figsize=(14, 8), facecolor='#1a1a2e')

        # Left: Probability evolution plot
        ax_prob = fig.add_axes((0.08, 0.12, 0.4, 0.75))
        ax_prob.set_facecolor('#16213e')

        # Right: Game state visualization
        ax_game = fig.add_axes((0.55, 0.12, 0.4, 0.75))
        ax_game.set_facecolor('#16213e')

        return fig, ax_prob, ax_game

    def _draw_player_box(
        self,
        ax: Any,
        x: float,
        y: float,
        player: PlayerState,
        probability: float,
        is_loser: bool = False,
        initial_lives: int = 3,
    ) -> None:
        """Draw a player's box with probability prominently displayed."""
        # Box dimensions
        box_width = 0.38
        box_height = 0.22

        # Box background
        if not player.is_active:
            color = '#444444'
            alpha = 0.5
        elif is_loser:
            color = '#ff4444'
            alpha = 0.9
        else:
            color = player.color
            alpha = 0.85

        rect = FancyBboxPatch(
            (x - box_width/2, y - box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor=color,
            edgecolor='white',
            linewidth=2,
            alpha=alpha
        )
        ax.add_patch(rect)

        # Player name
        ax.text(
            x, y + box_height/2 + 0.03,
            f"Player {player.player_id + 1}",
            ha='center', va='bottom',
            fontsize=12, fontweight='bold',
            color='white'
        )

        # Win probability - BIG and prominent
        if player.is_active:
            prob_text = f"{probability:.1%}"
            prob_color = 'white'
            fontsize = 28
        else:
            prob_text = "OUT"
            prob_color = '#888888'
            fontsize = 22

        ax.text(
            x, y + 0.02,
            prob_text,
            ha='center', va='center',
            fontsize=fontsize, fontweight='bold',
            color=prob_color
        )

        # Lives (hearts) below the probability
        hearts = ""
        for i in range(initial_lives):
            if i < player.lives:
                hearts += "\u2764 "
            else:
                hearts += "\u2661 "

        ax.text(
            x, y - box_height/2 - 0.04,
            hearts.strip(),
            ha='center', va='top',
            fontsize=14,
            color='#ff6b6b' if player.is_active else '#666666'
        )

    def animate_game(
        self,
        history: GameHistory,
        interval: int = 1500,
        fps: float = 2,
        filename: str = "cucu_probability_evolution.mp4",
        output_format: str = "mp4",
    ) -> Path:
        """
        Create animated video/gif of game probability evolution.

        Args:
            history: Complete game history with probabilities
            interval: Milliseconds between frames
            fps: Frames per second for output
            filename: Output filename
            output_format: "mp4" or "gif"

        Returns:
            Path to saved animation
        """
        fig, ax_prob, ax_game = self._setup_figure()

        num_players = history.config.num_players
        num_frames = len(history.states)

        # Prepare probability data for plotting
        prob_data = {i: [] for i in range(num_players)}
        for probs in history.probabilities:
            for pid in range(num_players):
                prob_data[pid].append(probs.get(pid, 0.0))

        # Get player colors
        colors = [history.states[0].players[i].color for i in range(num_players)]

        # Initialize plot elements
        lines = {}
        for i in range(num_players):
            line, = ax_prob.plot([], [], color=colors[i], linewidth=2.5,
                                 label=f'Player {i+1}', marker='o', markersize=6)
            lines[i] = line

        # Title
        title = fig.suptitle('', fontsize=16, fontweight='bold', color='white', y=0.96)

        def init():
            """Initialize animation."""
            ax_prob.set_xlim(-0.5, num_frames - 0.5)
            ax_prob.set_ylim(-0.05, 1.05)
            ax_prob.set_xlabel('Round', fontsize=12, color='white')
            ax_prob.set_ylabel('Win Probability', fontsize=12, color='white')
            ax_prob.set_title('Probability Evolution', fontsize=14,
                             fontweight='bold', color='white', pad=10)
            ax_prob.tick_params(colors='white')
            ax_prob.spines['bottom'].set_color('white')
            ax_prob.spines['left'].set_color('white')
            ax_prob.spines['top'].set_visible(False)
            ax_prob.spines['right'].set_visible(False)
            ax_prob.grid(True, alpha=0.3, color='white')
            ax_prob.legend(loc='upper right', facecolor='#16213e',
                          edgecolor='white', labelcolor='white')

            ax_game.set_xlim(0, 1)
            ax_game.set_ylim(0, 1)
            ax_game.set_aspect('equal')
            ax_game.axis('off')
            ax_game.set_title('Current Game State', fontsize=14,
                             fontweight='bold', color='white', pad=10)

            for line in lines.values():
                line.set_data([], [])

            return list(lines.values())

        def animate(frame_idx):
            """Update animation frame."""
            state = history.states[frame_idx]
            probs = history.probabilities[frame_idx]

            # Update title
            if frame_idx == 0:
                title.set_text('Cucù Game - Starting State')
            elif state.is_game_over():
                winner = history.winner
                if winner is not None:
                    title.set_text(f'Cucù Game - Player {winner + 1} Wins!')
                else:
                    title.set_text('Cucù Game - Game Over')
            else:
                title.set_text(f'Cucù Game - Round {state.round_num}')

            # Update probability lines
            for i in range(num_players):
                x_data = list(range(frame_idx + 1))
                y_data = prob_data[i][:frame_idx + 1]
                lines[i].set_data(x_data, y_data)

            # Update game state visualization
            ax_game.clear()
            ax_game.set_xlim(0, 1)
            ax_game.set_ylim(0, 1)
            ax_game.set_aspect('equal')
            ax_game.axis('off')
            ax_game.set_title('Current Game State', fontsize=14,
                             fontweight='bold', color='white', pad=10)

            # Calculate positions for players (2x2 grid for 4 players)
            positions = [
                (0.25, 0.7),   # Player 1
                (0.75, 0.7),   # Player 2
                (0.25, 0.3),   # Player 3
                (0.75, 0.3),   # Player 4
            ]

            # Extend for more players
            if num_players > 4:
                positions = [
                    (0.2 + 0.3 * (i % 3), 0.7 - 0.4 * (i // 3))
                    for i in range(num_players)
                ]

            # Draw each player
            for i, player in enumerate(state.players):
                x, y = positions[i] if i < len(positions) else (0.5, 0.5)
                is_loser = i in state.loser_this_round if state.loser_this_round else False

                self._draw_player_box(
                    ax_game, x, y, player, probs.get(i, 0.0), is_loser,
                    initial_lives=history.config.initial_lives
                )

            return list(lines.values())

        # Create animation
        anim = animation.FuncAnimation(
            fig, animate,
            init_func=init,
            frames=num_frames,
            interval=interval,
            blit=False,
            repeat=True
        )

        # Save animation
        output_path = self.output_dir / filename
        try:
            if output_format == "gif":
                output_path = output_path.with_suffix('.gif')
                writer = animation.PillowWriter(fps=fps)
                anim.save(str(output_path), writer=writer, dpi=100)
            else:
                output_path = output_path.with_suffix('.mp4')
                writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
                anim.save(str(output_path), writer=writer, dpi=120)
            print(f"Animation saved to: {output_path}")
        except Exception as e:
            print(f"Error saving animation: {e}")

        plt.close(fig)
        return output_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Create probability visualization."""
    import argparse

    parser = argparse.ArgumentParser(description="Cucù Probability Visualization")
    parser.add_argument("--players", type=int, default=4, help="Number of players")
    parser.add_argument("--lives", type=int, default=3, help="Initial lives per player")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--duration", type=float, default=7, help="Animation duration in seconds")
    parser.add_argument("--format", type=str, default="mp4", choices=["mp4", "gif"], help="Output format")
    args = parser.parse_args()

    print("=" * 60)
    print("CUCÙ PROBABILITY VISUALIZATION")
    print("=" * 60)

    config = GameConfig(num_players=args.players, initial_lives=args.lives)
    visualizer = ProbabilityVisualizer(output_dir=Path(args.output))

    print(f"\nCreating {args.format.upper()} animation...")
    simulator = GameSimulator(config, seed=args.seed)
    history = simulator.simulate_game()

    # Calculate fps for target duration
    num_frames = len(history.states)
    fps = num_frames / args.duration

    ext = "gif" if args.format == "gif" else "mp4"
    visualizer.animate_game(
        history,
        fps=fps,
        filename=f"cucu_probability_evolution.{ext}",
        output_format=args.format
    )

    print("\n" + "=" * 60)
    print("Visualization created!")
    print(f"Output: {visualizer.output_dir.absolute()}/cucu_probability_evolution.{ext}")
    print("=" * 60)


if __name__ == "__main__":
    main()
