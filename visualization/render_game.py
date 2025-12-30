"""
Render Cucù game as GIF or MP4 using HTML + Playwright for clean visuals.
"""

import argparse
import os
import shutil
import subprocess
import random
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
from playwright.sync_api import sync_playwright, Playwright, Browser, Page

from src.cucu_env import CucuEnv
from src.agents import RLAgent, ThresholdAgent
from visualization.utils import CSS, CARD_NAMES, CARD_FULL, card_color, player_position


# =============================================================================
# DATA
# =============================================================================

class ActionType(Enum):
    NONE = 0
    STAY = 1
    SWAP = 2


@dataclass
class PlayerState:
    agent_id: str
    card_value: Optional[int]
    lives: int
    is_dealer: bool
    action: ActionType
    is_rl_agent: bool
    is_active: bool
    lost_life: bool = False
    is_winner: bool = False


@dataclass
class GameFrame:
    round_num: int
    phase: str
    current_player: Optional[str]
    players: List[PlayerState]
    message: str
    show_cards: bool = False


# =============================================================================
# HTML RENDERER
# =============================================================================


def render_html(frame: GameFrame) -> str:
    """Generate HTML for a frame."""
    players_html = ""

    for i, p in enumerate(frame.players):
        x, y = player_position(i, len(frame.players))

        # Classes
        classes = ["player-box"]
        if p.is_winner:
            classes.append("winner")
        elif p.lost_life:
            classes.append("lost")
        elif frame.current_player == p.agent_id:
            classes.append("current")
        elif p.is_rl_agent:
            classes.append("rl")
        if not p.is_active:
            classes.append("inactive")

        # Name
        name = "RL" if p.is_rl_agent else f"P{p.agent_id.split('_')[1]}"
        name_class = "player-name rl" if p.is_rl_agent else "player-name"
        dealer = '<span class="dealer-badge">D</span>' if p.is_dealer else ""

        # Card - only show if player has one (active players only)
        if p.card_value:
            show = frame.show_cards or p.is_rl_agent or p.action != ActionType.NONE
            if show:
                color = card_color(p.card_value)
                card_html = f'<div class="card face-up {color}">{CARD_NAMES[p.card_value]}</div>'
            else:
                card_html = '<div class="card face-down">?</div>'
        else:
            card_html = ''  # No card for eliminated players

        # Lives
        hearts = ""
        for j in range(3):
            cls = "heart" if j < p.lives else "heart empty"
            hearts += f'<span class="{cls}">♥</span>'

        # Action
        action_html = ""
        if p.action != ActionType.NONE:
            act = "STAY" if p.action == ActionType.STAY else "SWAP"
            act_cls = "stay" if p.action == ActionType.STAY else "swap"
            action_html = f'<div class="action {act_cls}">{act}</div>'

        players_html += f'''
        <div class="player" style="left: {x}px; top: {y}px;">
            {action_html}
            <div class="{' '.join(classes)}">
                <div class="{name_class}">{name}{dealer}</div>
                {card_html}
                <div class="lives">{hearts}</div>
            </div>
        </div>
        '''

    html = f'''<!DOCTYPE html>
<html>
<head>
    <style>{CSS}</style>
</head>
<body>
    <div class="table">
        <div class="center-info">
            <div class="round">Round {frame.round_num}</div>
            <div class="phase">{frame.phase}</div>
        </div>
        {players_html}
    </div>
    <div class="message">{frame.message}</div>
</body>
</html>'''

    return html


class HTMLRenderer:
    def __init__(self) -> None:
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None

    def start(self) -> None:
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch()
        self.page = self.browser.new_page(viewport={'width': 800, 'height': 800})

    def render_frame(self, frame: GameFrame, path: str) -> None:
        html = render_html(frame)
        if self.page is not None:
            self.page.set_content(html)
            self.page.screenshot(path=path)

    def stop(self) -> None:
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()


# =============================================================================
# GAME SIMULATION
# =============================================================================

class GameSimulator:
    def __init__(self, num_players: int = 4, model_path: Optional[str] = None, device: str = "cpu"):
        self.num_players = num_players
        self.frames: List[GameFrame] = []
        self.env = CucuEnv(num_players=num_players, starting_lives=3)

        self.rl_agent = RLAgent("player_0", device=device)
        if model_path and os.path.exists(model_path):
            self.rl_agent.load(model_path)
            self.rl_agent.model.eval()
            print(f"Loaded: {model_path}")

        self.opponents = {f"player_{i}": ThresholdAgent(f"player_{i}", threshold=5)
                          for i in range(1, num_players)}
        self.agents = {"player_0": self.rl_agent, **self.opponents}
        self.round_num = 0
        self.actions: Dict[str, ActionType] = {}

    def _states(
        self,
        show: bool = False,
        current: Optional[str] = None,
        losers: Optional[List[str]] = None,
        winner: Optional[str] = None,
    ) -> List[PlayerState]:
        states: List[PlayerState] = []
        for i in range(self.num_players):
            aid = f"player_{i}"
            lives = self.env.player_lives.get(aid, 0)
            is_active = lives > 0

            # Determine if this player is the dealer
            is_dealer = False
            if self.env.agents and aid in self.env.agents:
                is_dealer = self.env.agents[self.env.dealer_idx] == aid

            # Use round_cards for reveal (shows cards used that round)
            # Otherwise use current env cards
            if show and aid in self.round_cards:
                card_value = self.round_cards[aid]
            else:
                card = self.env.player_cards.get(aid)
                card_value = card.value if (card and is_active) else None

            states.append(PlayerState(
                agent_id=aid,
                card_value=card_value,
                lives=lives,
                is_dealer=is_dealer,
                action=self.actions.get(aid, ActionType.NONE),
                is_rl_agent=(aid == "player_0"),
                is_active=is_active,
                lost_life=bool(losers and aid in losers),
                is_winner=(winner == aid),
            ))
        return states

    def _frame(self, phase, msg, current=None, show=False, losers=None, winner=None):
        self.frames.append(GameFrame(
            round_num=self.round_num,
            phase=phase,
            current_player=current,
            players=self._states(show, current, losers, winner),
            message=msg,
            show_cards=show,
        ))

    def simulate(self):
        self.env.reset()
        self.round_num = 1
        self.actions = {}
        self.round_cards = {}  # Store final cards for reveal

        self._frame("Dealing", "Game starting!")
        for _ in range(4): self._frame("Dealing", "Game starting!")

        while self.env.agents and len(self.env.agents) > 1:
            turn_order = self.env._get_turn_order()
            active_players = list(self.env.agents)  # Snapshot of active players

            # Track lives before round
            lives_before = {aid: self.env.player_lives.get(aid, 0) for aid in active_players}

            # Clear actions for new round
            self.actions = {}
            self.round_cards = {}

            for idx, aid in enumerate(turn_order):
                if aid not in self.env.agents:
                    continue
                if self.env.agent_selection != aid:
                    continue

                obs = self.env.observe(aid)
                cv = obs["card_value"]

                # Store current card value (this is their card at decision time)
                self.round_cards[aid] = cv

                action = self.agents[aid].select_action(obs)
                self.actions[aid] = ActionType.STAY if action == 0 else ActionType.SWAP
                act_str = "STAY" if action == 0 else "SWAP"

                if aid == "player_0":
                    msg = f"RL Agent: {CARD_FULL[cv]} → {act_str}"
                else:
                    msg = f"Player {aid.split('_')[1]}: {act_str}"

                self._frame(f"Turn {idx+1}", msg, current=aid)
                for _ in range(3): self._frame(f"Turn {idx+1}", msg, current=aid)

                # Before dealer acts, capture all final cards (after swaps received)
                is_dealer = (idx == len(turn_order) - 1)
                if is_dealer:
                    # Update all players' final cards before round resolves
                    for pid in active_players:
                        card = self.env.player_cards.get(pid)
                        if card:
                            self.round_cards[pid] = card.value

                self.env.step(action)

            # Find losers by comparing lives before and after
            losers = []
            for aid in active_players:
                if self.env.player_lives.get(aid, 0) < lives_before.get(aid, 0):
                    losers.append(aid)

            # Find minimum card value for the message
            min_v = None
            if losers:
                for aid in losers:
                    cv = self.round_cards.get(aid)
                    if cv and (min_v is None or cv < min_v):
                        min_v = cv

            loser_str = ", ".join(["RL" if l == "player_0" else f"P{l.split('_')[1]}" for l in losers])
            if losers and min_v is not None:
                msg = f"Lowest: {CARD_FULL.get(min_v, '?')} — {loser_str} loses!"
            else:
                msg = "Round complete"

            self._frame("Reveal", msg, show=True, losers=losers)
            for _ in range(6): self._frame("Reveal", msg, show=True, losers=losers)

            if len(self.env.agents) <= 1:
                break

            self.round_num += 1
            self.actions = {}
            self._frame("Dealing", f"Round {self.round_num}")
            for _ in range(3): self._frame("Dealing", f"Round {self.round_num}")

        # Winner
        winner = None
        for aid in self.env.possible_agents:
            if self.env.player_lives.get(aid, 0) > 0:
                winner = aid
                break

        win_name = "RL Agent" if winner == "player_0" else f"Player {winner.split('_')[1]}" if winner else "None"
        msg = f"🏆 WINNER: {win_name}! 🏆"

        self._frame("Game Over", msg, show=True, winner=winner)
        for _ in range(10): self._frame("Game Over", msg, show=True, winner=winner)

        return winner or "draw"


# =============================================================================
# VIDEO
# =============================================================================

def make_video(frames_dir, output, fps=5):
    if not shutil.which("ffmpeg"):
        print("ffmpeg not found")
        return False

    cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i", f"{frames_dir}/frame_%04d.png",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", output]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def make_gif(frames_dir, output, fps=5):
    """Create high-quality GIF using ffmpeg with palette generation."""
    if not shutil.which("ffmpeg"):
        print("ffmpeg not found")
        return False

    # Two-pass approach for better GIF quality
    palette_filter = f"fps={fps},scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", f"{frames_dir}/frame_%04d.png",
        "-vf", palette_filter,
        "-loop", "0",  # Loop forever
        output
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best_model.pt")
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--output", default="game_visualization.gif")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    print(f"Model: {args.model}")
    print(f"Output: {args.output}")

    frames_dir = "visualization/frames"
    os.makedirs(frames_dir, exist_ok=True)
    for f in os.listdir(frames_dir):
        if f.endswith(".png"): os.remove(os.path.join(frames_dir, f))

    print("Simulating game...")
    sim = GameSimulator(args.players, args.model)
    winner = sim.simulate()
    print(f"Winner: {winner}, Frames: {len(sim.frames)}")

    print("Rendering frames with HTML...")
    renderer = HTMLRenderer()
    renderer.start()

    for i, frame in enumerate(sim.frames):
        renderer.render_frame(frame, f"{frames_dir}/frame_{i:04d}.png")
        if (i+1) % 50 == 0: print(f"  Rendered {i+1}/{len(sim.frames)}")

    renderer.stop()
    print(f"  Rendered {len(sim.frames)} frames")

    # Detect output format
    is_gif = args.output.lower().endswith(".gif")
    format_name = "GIF" if is_gif else "video"
    print(f"Generating {format_name}...")

    if is_gif:
        success = make_gif(frames_dir, args.output, args.fps)
    else:
        success = make_video(frames_dir, args.output, args.fps)

    if success:
        print(f"✓ Saved: {args.output}")
        if not args.keep_frames:
            for f in os.listdir(frames_dir):
                if f.endswith(".png"): os.remove(os.path.join(frames_dir, f))
    else:
        print(f"✗ {format_name} failed, frames in:", frames_dir)


if __name__ == "__main__":
    main()
