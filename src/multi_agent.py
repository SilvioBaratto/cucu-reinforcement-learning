"""Multi-agent game wrapper for training."""

from typing import Dict, List, Tuple, Optional, Any, Mapping
import numpy as np

from .cucu_env import CucuEnv
from .agents.base_agent import BaseAgent


class MultiAgentWrapper:
    """
    Wrapper for running multi-agent games with different agent types.
    Handles turn-based play and episode management.
    """
    
    def __init__(
        self,
        num_players: int = 4,
        starting_lives: int = 3,
    ):
        self.env = CucuEnv(
            num_players=num_players,
            starting_lives=starting_lives,
        )
        self.agents: Dict[str, BaseAgent] = {}
    
    def set_agents(self, agents: Mapping[str, BaseAgent]) -> None:
        """Set agents for each player."""
        self.agents = dict(agents)
    
    def run_episode(self, max_steps: int = 10000) -> Dict[str, float]:
        """
        Run a complete game episode.

        Args:
            max_steps: Maximum steps before forcing termination (safety limit)

        Returns:
            Dictionary of cumulative rewards per agent.
        """
        self.env.reset()
        steps = 0

        while self.env.agents and steps < max_steps:
            agent_id = self.env.agent_selection
            observation = self.env.observe(agent_id)

            if agent_id in self.agents:
                action = self.agents[agent_id].select_action(observation)
            else:
                action = 0  # Default: stay

            self.env.step(action)
            steps += 1

        if steps >= max_steps:
            print(f"WARNING: Episode exceeded {max_steps} steps, forcing termination")

        return dict(self.env._cumulative_rewards)
    
    def run_episodes(self, num_episodes: int) -> List[Dict[str, float]]:
        """Run multiple episodes and collect results."""
        results = []
        for _ in range(num_episodes):
            episode_rewards = self.run_episode()
            results.append(episode_rewards)
        return results
    
    def collect_training_data(
        self,
        training_agent_id: str,
        num_steps: int,
    ) -> Tuple[List, List, List, List, List]:
        """
        Collect training data for a specific agent.
        
        Returns:
            observations, actions, rewards, dones, next_observations
        """
        observations = []
        actions = []
        rewards = []
        dones = []
        next_observations = []
        
        self.env.reset()
        steps = 0
        
        while steps < num_steps:
            agent_id = self.env.agent_selection
            obs = self.env.observe(agent_id)
            
            if agent_id == training_agent_id:
                observations.append(obs)
                
                agent = self.agents.get(agent_id)
                action = agent.select_action(obs) if agent else 0
                actions.append(action)
                
                self.env.step(action)
                
                reward = self.env._cumulative_rewards.get(agent_id, 0)
                done = self.env.terminations.get(agent_id, False)
                next_obs = self.env.observe(agent_id)
                
                rewards.append(reward)
                dones.append(done)
                next_observations.append(next_obs)
                
                steps += 1
            else:
                agent = self.agents.get(agent_id)
                action = agent.select_action(obs) if agent else 0
                self.env.step(action)
            
            if not self.env.agents:
                self.env.reset()
        
        return observations, actions, rewards, dones, next_observations
