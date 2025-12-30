#!/usr/bin/env python3
"""
Cucù Reinforcement Learning - CLI Application

A modern CLI for training and evaluating RL agents on the Cucù card game.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from src.cucu_env import CucuEnv
from src.agents import RLAgent, ThresholdAgent, RandomAgent
from src.models import ActorCritic, PPO
from src.multi_agent import MultiAgentWrapper
from src.utils.logger import TrainingLogger
from src.utils.metrics import compute_win_rate

console = Console()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cmd_monte_carlo(args: argparse.Namespace) -> None:
    """Run Monte Carlo simulation to find optimal thresholds."""
    console.print(Panel.fit(
        "[bold cyan]Phase 1: Monte Carlo Simulation[/bold cyan]\n"
        f"Games: {args.games} | Players: {args.players}",
        border_style="cyan"
    ))

    if args.seed:
        set_seed(args.seed)

    thresholds = range(1, 11)
    results: dict[int, dict[str, float]] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Testing thresholds...", total=len(thresholds))

        for threshold in thresholds:
            wins = 0
            total_reward = 0.0

            for _ in range(args.games):
                env = CucuEnv(num_players=args.players, starting_lives=3)
                env.reset()

                agents = {
                    "player_0": ThresholdAgent("player_0", threshold=threshold),
                    **{f"player_{i}": ThresholdAgent(f"player_{i}", threshold=5)
                       for i in range(1, args.players)}
                }

                while env.agents and len(env.agents) > 1:
                    agent = env.agent_selection
                    if agent not in env.agents:
                        break
                    obs = env.observe(agent)
                    action = agents[agent].select_action(obs)
                    env.step(action)

                for aid in env.possible_agents:
                    if env.player_lives.get(aid, 0) > 0:
                        if aid == "player_0":
                            wins += 1
                        break

                total_reward += env._cumulative_rewards.get("player_0", 0)

            results[threshold] = {
                "win_rate": wins / args.games,
                "avg_reward": total_reward / args.games,
            }
            progress.update(task, advance=1)

    # Display results
    table = Table(title="Monte Carlo Results", border_style="cyan")
    table.add_column("Threshold", style="cyan", justify="center")
    table.add_column("Win Rate", style="green", justify="center")
    table.add_column("Avg Reward", style="yellow", justify="center")

    best_threshold = max(results, key=lambda t: results[t]["win_rate"])
    for threshold, data in results.items():
        style = "bold" if threshold == best_threshold else ""
        table.add_row(
            str(threshold),
            f"{data['win_rate']:.2%}",
            f"{data['avg_reward']:.2f}",
            style=style,
        )

    console.print(table)
    console.print(f"\n[bold green]Best threshold: {best_threshold}[/bold green] "
                  f"(Win rate: {results[best_threshold]['win_rate']:.2%})")


def cmd_train(args: argparse.Namespace) -> None:
    """Train PPO agent with self-play."""
    import os

    console.print(Panel.fit(
        "[bold green]Phase 2: PPO Self-Play Training[/bold green]\n"
        f"Episodes: {args.episodes} | Players: {args.min_players}-{args.max_players} | Device: {args.device}",
        border_style="green"
    ))

    if args.seed:
        set_seed(args.seed)

    os.makedirs("models", exist_ok=True)
    logger = TrainingLogger(log_dir="results", experiment_name="ppo_selfplay")

    # Create model and PPO trainer
    obs_dim = RLAgent.OBS_DIM
    model = ActorCritic(obs_dim=obs_dim, action_dim=2, hidden_dim=64)
    ppo = PPO(model=model, device=args.device, entropy_coef=0.05, lr=3e-4)

    # Training agent
    training_agent = RLAgent("player_0", obs_dim=obs_dim, device=args.device)
    training_agent.model = model

    if args.resume and Path(args.resume).exists():
        ppo.load(args.resume)
        console.print(f"[yellow]Resumed from {args.resume}[/yellow]")

    best_win_rate = 0.0
    global_step = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[cyan]{task.fields[status]}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[green]Training...",
            total=args.episodes,
            status="Starting..."
        )

        for episode in range(args.episodes):
            num_players = random.randint(args.min_players, args.max_players)
            wrapper = MultiAgentWrapper(num_players=num_players)

            # Curriculum: gradually increase opponent difficulty
            difficulty = min(1.0, episode / (args.episodes * 0.5))
            opponents = _create_opponent_pool(num_players, difficulty)
            all_agents = {"player_0": training_agent, **opponents}
            wrapper.set_agents(all_agents)

            wrapper.env.reset()
            training_agent.reset()

            episode_reward = 0.0
            steps = 0

            while wrapper.env.agents and steps < 1000:
                agent_id = wrapper.env.agent_selection
                obs = wrapper.env.observe(agent_id)

                if agent_id == "player_0":
                    obs_tensor = training_agent._obs_to_tensor(obs)
                    action_mask = training_agent._compute_action_mask(obs)

                    action, log_prob, value, _entropy, _ = model.get_action(
                        obs_tensor.unsqueeze(0),
                        action_mask=action_mask.unsqueeze(0),
                    )

                    action_int = int(action.item())
                    wrapper.env.step(action_int)

                    reward = wrapper.env._cumulative_rewards.get(agent_id, 0) - episode_reward
                    episode_reward = wrapper.env._cumulative_rewards.get(agent_id, 0)
                    done = wrapper.env.terminations.get(agent_id, False)

                    ppo.collect_rollout(
                        obs=obs_tensor,
                        action=action_int,
                        reward=reward,
                        done=done,
                        log_prob=log_prob,
                        value=value,
                        action_mask=action_mask,
                    )

                    global_step += 1
                    steps += 1

                    if len(ppo.buffer) >= args.rollout_steps:
                        metrics = ppo.update()
                        logger.log(metrics, step=global_step, prefix="train")
                else:
                    agent = opponents.get(agent_id)
                    action = agent.select_action(obs) if agent else 0
                    wrapper.env.step(action)

            # Evaluation
            if (episode + 1) % args.eval_freq == 0:
                avg_win_rate = 0.0
                for test_players in [4, 6, 8]:
                    wr = _evaluate_agent(training_agent, test_players, num_games=50)
                    logger.log({f"win_rate_{test_players}p": wr}, step=episode, prefix="eval")
                    avg_win_rate += wr
                avg_win_rate /= 3

                if avg_win_rate > best_win_rate:
                    best_win_rate = avg_win_rate
                    ppo.save("models/best_model.pt")

                progress.update(task, status=f"Win rate: {avg_win_rate:.1%}")

            if (episode + 1) % args.save_freq == 0:
                ppo.save(f"models/checkpoint_{episode+1}.pt")

            progress.update(task, advance=1)

    ppo.save("models/final_model.pt")
    logger.close()

    console.print(f"\n[bold green]Training complete![/bold green]")
    console.print(f"Best win rate: {best_win_rate:.2%}")
    console.print(f"Models saved to: models/")


def _create_opponent_pool(num_players: int, difficulty: float = 0.0) -> dict:
    """Create opponent pool with adjustable difficulty."""
    agents = {}
    for i in range(1, num_players):
        agent_id = f"player_{i}"
        if random.random() > difficulty:
            agents[agent_id] = RandomAgent(agent_id)
        else:
            threshold = random.choice([3, 4, 5])
            agents[agent_id] = ThresholdAgent(agent_id, threshold=threshold)
    return agents


def _evaluate_agent(agent: RLAgent, num_players: int, num_games: int = 100) -> float:
    """Evaluate agent against baseline opponents."""
    wrapper = MultiAgentWrapper(num_players=num_players)
    opponents = {
        f"player_{i}": ThresholdAgent(f"player_{i}", threshold=5)
        for i in range(1, num_players)
    }
    all_agents = {"player_0": agent, **opponents}
    wrapper.set_agents(all_agents)

    results = []
    for _ in range(num_games):
        agent.reset()
        rewards = wrapper.run_episode()
        results.append(rewards)

    return compute_win_rate(results, "player_0")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate trained agent."""
    console.print(Panel.fit(
        "[bold yellow]Phase 3: Agent Evaluation[/bold yellow]\n"
        f"Model: {args.model} | Games: {args.games} | Players: {args.players}",
        border_style="yellow"
    ))

    if args.seed:
        set_seed(args.seed)

    rl_agent = RLAgent("player_0", device=args.device)
    if Path(args.model).exists():
        rl_agent.load(args.model)
        rl_agent.model.eval()
        console.print(f"[green]Loaded model: {args.model}[/green]")
    else:
        console.print(f"[red]Model not found: {args.model}[/red]")
        return

    opponent_types = {
        "threshold_3": lambda i: ThresholdAgent(f"player_{i}", threshold=3),
        "threshold_5": lambda i: ThresholdAgent(f"player_{i}", threshold=5),
        "threshold_7": lambda i: ThresholdAgent(f"player_{i}", threshold=7),
        "random": lambda i: RandomAgent(f"player_{i}"),
    }

    results: dict[str, dict[str, float]] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        for opp_name, opp_factory in opponent_types.items():
            task = progress.add_task(f"[yellow]vs {opp_name}...", total=args.games)
            wins = 0

            for _ in range(args.games):
                env = CucuEnv(num_players=args.players, starting_lives=3)
                env.reset()

                opponents = {f"player_{i}": opp_factory(i) for i in range(1, args.players)}

                while env.agents and len(env.agents) > 1:
                    agent = env.agent_selection
                    if agent not in env.agents:
                        break

                    obs = env.observe(agent)
                    if agent == "player_0":
                        action = rl_agent.select_action(obs)
                    else:
                        action = opponents[agent].select_action(obs)
                    env.step(action)

                for aid in env.possible_agents:
                    if env.player_lives.get(aid, 0) > 0:
                        if aid == "player_0":
                            wins += 1
                        break

                progress.update(task, advance=1)

            results[opp_name] = {"win_rate": wins / args.games}

    table = Table(title="Evaluation Results", border_style="yellow")
    table.add_column("Opponent", style="cyan")
    table.add_column("Win Rate", style="green", justify="center")
    table.add_column("Expected (Random)", style="dim", justify="center")

    expected = 1.0 / args.players
    for opp_name, data in results.items():
        diff = data["win_rate"] - expected
        diff_str = f"({diff:+.1%})" if diff >= 0 else f"([red]{diff:+.1%}[/red])"
        table.add_row(
            opp_name,
            f"{data['win_rate']:.2%}",
            f"{expected:.2%} {diff_str}",
        )

    console.print(table)

    avg_win_rate = sum(r["win_rate"] for r in results.values()) / len(results)
    console.print(f"\n[bold]Average win rate: {avg_win_rate:.2%}[/bold]")


def cmd_render(args: argparse.Namespace) -> None:
    """Render a game visualization."""
    import os
    from visualization.render_game import GameSimulator, HTMLRenderer, make_gif, make_video

    console.print(Panel.fit(
        "[bold magenta]Game Visualization[/bold magenta]\n"
        f"Model: {args.model} | Output: {args.output}",
        border_style="magenta"
    ))

    if args.seed:
        set_seed(args.seed)

    frames_dir = "visualization/frames"
    os.makedirs(frames_dir, exist_ok=True)

    for f in os.listdir(frames_dir):
        if f.endswith(".png"):
            os.remove(os.path.join(frames_dir, f))

    with console.status("[magenta]Simulating game..."):
        sim = GameSimulator(args.players, args.model)
        winner = sim.simulate()

    console.print(f"Winner: [bold]{winner}[/bold] | Frames: {len(sim.frames)}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[magenta]Rendering frames...", total=len(sim.frames))

        renderer = HTMLRenderer()
        renderer.start()

        for i, frame in enumerate(sim.frames):
            renderer.render_frame(frame, f"{frames_dir}/frame_{i:04d}.png")
            progress.update(task, advance=1)

        renderer.stop()

    with console.status("[magenta]Generating output..."):
        is_gif = args.output.lower().endswith(".gif")
        if is_gif:
            success = make_gif(frames_dir, args.output, args.fps)
        else:
            success = make_video(frames_dir, args.output, args.fps)

    if success:
        console.print(f"[bold green]✓ Saved: {args.output}[/bold green]")
        if not args.keep_frames:
            for f in os.listdir(frames_dir):
                if f.endswith(".png"):
                    os.remove(os.path.join(frames_dir, f))
    else:
        console.print(f"[bold red]✗ Failed to generate output[/bold red]")


def cmd_info(_args: argparse.Namespace) -> None:
    """Show project information."""
    import sys

    info = Table.grid(padding=(0, 2))
    info.add_column(style="cyan", justify="right")
    info.add_column(style="white")

    info.add_row("Project", "Cucù Reinforcement Learning")
    info.add_row("Python", sys.version.split()[0])
    info.add_row("PyTorch", torch.__version__)
    info.add_row("CUDA", "Available" if torch.cuda.is_available() else "Not available")
    info.add_row("Device", "cuda" if torch.cuda.is_available() else "cpu")

    models_dir = Path("models")
    if models_dir.exists():
        models = list(models_dir.glob("*.pt"))
        info.add_row("Models", str(len(models)) + " found")
    else:
        info.add_row("Models", "None")

    console.print(Panel(info, title="[bold]System Information[/bold]", border_style="blue"))


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Cucù Reinforcement Learning CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Monte Carlo
    mc_parser = subparsers.add_parser("monte-carlo", help="Run Monte Carlo simulation")
    mc_parser.add_argument("--games", type=int, default=10000, help="Number of games")
    mc_parser.add_argument("--players", type=int, default=4, help="Number of players")
    mc_parser.add_argument("--seed", type=int, help="Random seed")

    # Train
    train_parser = subparsers.add_parser("train", help="Train PPO agent")
    train_parser.add_argument("--episodes", type=int, default=100000, help="Training episodes")
    train_parser.add_argument("--min-players", type=int, default=4, help="Minimum players")
    train_parser.add_argument("--max-players", type=int, default=8, help="Maximum players")
    train_parser.add_argument("--rollout-steps", type=int, default=2048, help="Rollout steps")
    train_parser.add_argument("--eval-freq", type=int, default=1000, help="Evaluation frequency")
    train_parser.add_argument("--save-freq", type=int, default=10000, help="Save frequency")
    train_parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    train_parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    train_parser.add_argument("--seed", type=int, help="Random seed")

    # Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained agent")
    eval_parser.add_argument("--model", default="models/best_model.pt", help="Model path")
    eval_parser.add_argument("--games", type=int, default=5000, help="Number of games")
    eval_parser.add_argument("--players", type=int, default=4, help="Number of players")
    eval_parser.add_argument("--device", default="cpu", help="Device")
    eval_parser.add_argument("--seed", type=int, help="Random seed")

    # Render
    render_parser = subparsers.add_parser("render", help="Render game visualization")
    render_parser.add_argument("--model", default="models/best_model.pt", help="Model path")
    render_parser.add_argument("--output", default="game_visualization.gif", help="Output file")
    render_parser.add_argument("--players", type=int, default=4, help="Number of players")
    render_parser.add_argument("--fps", type=int, default=5, help="Frames per second")
    render_parser.add_argument("--seed", type=int, help="Random seed")
    render_parser.add_argument("--keep-frames", action="store_true", help="Keep frame images")

    # Info
    subparsers.add_parser("info", help="Show system information")

    args = parser.parse_args()

    console.print()
    console.print("[bold blue]🎴 Cucù Reinforcement Learning[/bold blue]")
    console.print()

    if args.command == "monte-carlo":
        cmd_monte_carlo(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "render":
        cmd_render(args)
    elif args.command == "info":
        cmd_info(args)
    else:
        parser.print_help()
        console.print("\n[dim]Examples:[/dim]")
        console.print("  cucu monte-carlo --games 10000")
        console.print("  cucu train --episodes 100000 --device cpu")
        console.print("  cucu evaluate --model models/best_model.pt")
        console.print("  cucu render --seed 3")
        console.print("  cucu info")


if __name__ == "__main__":
    main()
