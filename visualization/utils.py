"""Utility constants and functions for game visualization."""

import numpy as np

# Card display mappings
CARD_NAMES = {1: "A", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "J", 9: "H", 10: "K"}
CARD_FULL = {1: "Ace", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "Jack", 9: "Horse", 10: "King"}

# CSS styles for HTML rendering
CSS = """
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    width: 800px;
    height: 800px;
    background: #0f0f1a;
    font-family: 'Segoe UI', system-ui, sans-serif;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    position: relative;
}

.table {
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, #1a3a2a 0%, #0f2518 100%);
    border-radius: 50%;
    border: 8px solid #2d5a3d;
    position: relative;
    box-shadow: 0 0 60px rgba(45, 90, 61, 0.3);
}

.center-info {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    text-align: center;
    color: #6a8a7a;
}

.round {
    font-size: 28px;
    font-weight: bold;
    color: #8ab89a;
}

.phase {
    font-size: 16px;
    margin-top: 4px;
    opacity: 0.8;
}

.player {
    position: absolute;
    width: 120px;
    text-align: center;
    transform: translate(-50%, -50%);
}

.player-box {
    background: #1a1a2a;
    border: 2px solid #3a3a4a;
    border-radius: 12px;
    padding: 12px 8px;
    transition: all 0.2s;
}

.player-box.current {
    border-color: #6a6aaa;
    background: #252540;
}

.player-box.rl {
    border-color: #e94560;
    background: #2a1a2a;
}

.player-box.lost {
    border-color: #aa4444;
    background: #2a1a1a;
}

.player-box.winner {
    border-color: #44aa66;
    background: #1a2a1a;
}

.player-box.inactive {
    opacity: 0.4;
}

.player-name {
    font-size: 14px;
    font-weight: bold;
    color: #ffffff;
    margin-bottom: 8px;
}

.player-name.rl {
    color: #e94560;
}

.dealer-badge {
    display: inline-block;
    background: #ff9800;
    color: #000;
    font-size: 10px;
    padding: 1px 5px;
    border-radius: 4px;
    margin-left: 4px;
}

.card {
    width: 50px;
    height: 70px;
    margin: 0 auto 8px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
    font-weight: bold;
}

.card.face-up {
    background: #ffffff;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}

.card.face-down {
    background: linear-gradient(135deg, #1a237e, #283593);
    color: #5c6bc0;
    font-size: 24px;
}

.card.red { color: #c62828; border: 2px solid #c62828; }
.card.blue { color: #1565c0; border: 2px solid #1565c0; }
.card.green { color: #2e7d32; border: 2px solid #2e7d32; }

.lives {
    display: flex;
    justify-content: center;
    gap: 4px;
    font-size: 16px;
}

.heart { color: #e53935; }
.heart.empty { color: #3a3a4a; }

.action {
    position: absolute;
    top: -28px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 12px;
    font-weight: bold;
    padding: 3px 10px;
    border-radius: 4px;
}

.action.stay {
    background: #1b4332;
    color: #4caf50;
    border: 1px solid #4caf50;
}

.action.swap {
    background: #3d2e1a;
    color: #ff9800;
    border: 1px solid #ff9800;
}

.message {
    position: absolute;
    bottom: 20px;
    right: 20px;
    font-size: 14px;
    color: #ffffff;
    font-weight: bold;
    padding: 10px 16px;
    background: #1a1a2a;
    border-radius: 8px;
    border: 1px solid #3a3a4a;
}
"""


def card_color(v: int) -> str:
    """Get CSS color class for a card value."""
    if v >= 8:
        return "green"
    if v >= 4:
        return "blue"
    return "red"


def player_position(idx: int, total: int) -> tuple[float, float]:
    """Calculate position around table for a player."""
    angle = -90 + (360 / total) * idx
    rad = angle * 3.14159 / 180
    r = 290  # Distance from center
    x = 250 + r * np.cos(rad)
    y = 250 + r * np.sin(rad)
    return x, y
