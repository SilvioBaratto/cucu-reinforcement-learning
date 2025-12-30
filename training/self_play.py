"""Phase 2: Self-play training with PPO."""

import argparse
import os
from typing import Dict, List, Optional
from tqdm import tqdm
import torch
import numpy as np

from src.cucu_env import CucuEnv
from src.agents import RandomAgent, ThresholdAgent, RLAgent, BaseAgent
from src.models import ActorCritic, PPO
from src.multi_agent import MultiAgentWrapper
from src.utils.logger import TrainingLogger
from src.utils.metrics import compute_win_rate


def create_opponent_pool(num_players: int, difficulty: float = 0.0) -> Dict[str, BaseAgent]:
    """Create a pool of opponents with adjustable difficulty.

    Args:
        num_players: Number of players
        difficulty: 0.0 = all random, 1.0 = all skilled threshold agents
    """
    agents = {}

    for i in range(1, num_players):
        agent_id = f"player_{i}"

        # Gradually introduce harder opponents based on difficulty
        import random
        if random.random() > difficulty:
            # Easy opponent: random
            agents[agent_id] = RandomAgent(agent_id)
        else:
            # Harder opponent: threshold agent
            threshold = random.choice([3, 4, 5])
            agents[agent_id] = ThresholdAgent(agent_id, threshold=threshold)

    return agents


def train_ppo(
    num_episodes: int = 100000,
    min_players: int = 4,
    max_players: int = 8,
    rollout_steps: int = 2048,
    save_freq: int = 10000,
    eval_freq: int = 1000,
    log_dir: str = "results",
    model_dir: str = "models",
    device: str = "cpu",
    entropy_coef: float = 0.05,
    lr: float = 3e-4,
) -> None:
    """
    Train PPO agent through self-play with variable table sizes.

    Args:
        num_episodes: Total training episodes
        min_players: Minimum players per game
        max_players: Maximum players per game
        rollout_steps: Steps before each PPO update
        save_freq: Model checkpoint frequency
        eval_freq: Evaluation frequency
        log_dir: Directory for logs
        model_dir: Directory for saved models
        device: Training device
        entropy_coef: Entropy coefficient for exploration (lower = more deterministic)
        lr: Learning rate
    """
    import random

    print("=" * 60)
    print("PPO Self-Play Training (Variable Table Sizes)")
    print(f"Training with {min_players}-{max_players} players")
    print("=" * 60)

    # Setup
    os.makedirs(model_dir, exist_ok=True)
    logger = TrainingLogger(log_dir=log_dir, experiment_name="ppo_selfplay")

    # Create model and PPO trainer
    obs_dim = RLAgent.OBS_DIM  # 12 features (expanded for action masking)
    action_dim = 2  # stay or cucu/draw

    model = ActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=64)
    ppo = PPO(model=model, device=device, entropy_coef=entropy_coef, lr=lr)

    # Training agent (will be reused across different table sizes)
    training_agent = RLAgent("player_0", obs_dim=obs_dim, device=device)
    training_agent.model = model  # Share model with PPO

    # Training loop
    episode = 0
    global_step = 0
    best_win_rate = 0.0

    pbar = tqdm(total=num_episodes, desc="Training")

    while episode < num_episodes:
        # Randomly select number of players for this episode
        num_players = random.randint(min_players, max_players)

        # Create new environment with this number of players
        wrapper = MultiAgentWrapper(num_players=num_players)

        # Curriculum: gradually increase opponent difficulty
        difficulty = min(1.0, episode / (num_episodes * 0.5))  # Ramp up over first 50%
        opponents = create_opponent_pool(num_players, difficulty=difficulty)
        all_agents = {"player_0": training_agent, **opponents}
        wrapper.set_agents(all_agents)

        # Reset for new episode
        wrapper.env.reset()
        training_agent.reset()

        episode_reward = 0
        steps_this_episode = 0

        while wrapper.env.agents and steps_this_episode < 1000:
            agent_id = wrapper.env.agent_selection
            obs = wrapper.env.observe(agent_id)

            if agent_id == "player_0":
                # Training agent
                obs_tensor = training_agent._obs_to_tensor(obs)
                action_mask = training_agent._compute_action_mask(obs)

                action, log_prob, value, entropy, _ = model.get_action(
                    obs_tensor.unsqueeze(0),
                    action_mask=action_mask.unsqueeze(0),
                )

                action_int = int(action.item())
                wrapper.env.step(action_int)

                reward = wrapper.env._cumulative_rewards.get(agent_id, 0) - episode_reward
                episode_reward = wrapper.env._cumulative_rewards.get(agent_id, 0)
                done = wrapper.env.terminations.get(agent_id, False)

                # Store transition with action mask
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
                steps_this_episode += 1

                # Update if buffer is full
                if len(ppo.buffer) >= rollout_steps:
                    metrics = ppo.update()
                    logger.log(metrics, step=global_step, prefix="train")
            else:
                # Opponent agent
                agent = opponents.get(agent_id)
                if agent:
                    action = agent.select_action(obs)
                else:
                    action = 0
                wrapper.env.step(action)

        # Episode complete
        episode += 1
        pbar.update(1)

        # Log episode
        logger.log({
            "reward": episode_reward,
            "steps": steps_this_episode,
            "num_players": num_players,
        }, step=episode, prefix="episode")

        # Evaluation (test on different table sizes)
        if episode % eval_freq == 0:
            avg_win_rate = 0.0
            for test_players in [4, 6, 8]:
                wr = evaluate_agent(training_agent, test_players, num_games=50)
                logger.log({f"win_rate_{test_players}p": wr}, step=episode, prefix="eval")
                avg_win_rate += wr
            avg_win_rate /= 3

            pbar.set_postfix({"avg_win_rate": f"{avg_win_rate:.3f}"})

            if avg_win_rate > best_win_rate:
                best_win_rate = avg_win_rate
                ppo.save(os.path.join(model_dir, "best_model.pt"))

        # Checkpoint
        if episode % save_freq == 0:
            ppo.save(os.path.join(model_dir, f"checkpoint_{episode}.pt"))

    pbar.close()
    logger.close()

    print(f"\nTraining complete!")
    print(f"Best average win rate: {best_win_rate:.4f}")


def evaluate_agent(
    agent: RLAgent,
    num_players: int,
    num_games: int = 100,
) -> float:
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


def main():
    parser = argparse.ArgumentParser(description="PPO Self-Play Training")
    parser.add_argument("--episodes", type=int, default=100000)
    parser.add_argument("--min-players", type=int, default=4)
    parser.add_argument("--max-players", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--save-freq", type=int, default=10000)
    parser.add_argument("--eval-freq", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-dir", type=str, default="results")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--entropy-coef", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    train_ppo(
        num_episodes=args.episodes,
        min_players=args.min_players,
        max_players=args.max_players,
        rollout_steps=args.rollout_steps,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        device=args.device,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        entropy_coef=args.entropy_coef,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
