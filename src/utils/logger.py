"""Training logger with TensorBoard support."""

import os
from typing import TYPE_CHECKING, Dict, Optional
from datetime import datetime

TENSORBOARD_AVAILABLE = False
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    pass

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    """
    Logger for training metrics.
    
    Supports:
    - Console logging
    - TensorBoard logging
    - CSV export
    """
    
    def __init__(
        self,
        log_dir: str = "results",
        experiment_name: Optional[str] = None,
        use_tensorboard: bool = True,
    ):
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        self.writer: Optional["SummaryWriter"] = None

        if self.use_tensorboard and TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=self.log_dir)  # type: ignore[possibly-undefined]
        
        self.metrics_history: Dict[str, list] = {}
        self.step = 0
    
    def log(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Global step (uses internal counter if None)
            prefix: Prefix for metric names
        """
        if step is None:
            step = self.step
            self.step += 1
        
        for name, value in metrics.items():
            full_name = f"{prefix}/{name}" if prefix else name
            
            # Store in history
            if full_name not in self.metrics_history:
                self.metrics_history[full_name] = []
            self.metrics_history[full_name].append((step, value))
            
            # Log to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar(full_name, value, step)
    
    def log_episode(
        self,
        episode: int,
        rewards: Dict[str, float],
        extra_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log episode results."""
        avg_reward = sum(rewards.values()) / len(rewards)
        max_reward = max(rewards.values())
        
        metrics = {
            "episode/avg_reward": avg_reward,
            "episode/max_reward": max_reward,
            "episode/num_players": len(rewards),
        }
        
        if extra_metrics:
            metrics.update(extra_metrics)
        
        self.log(metrics, step=episode)
    
    def print_status(
        self,
        episode: int,
        metrics: Dict[str, float],
    ) -> None:
        """Print training status to console."""
        status = f"Episode {episode}"
        for name, value in metrics.items():
            status += f" | {name}: {value:.4f}"
        print(status)
    
    def save_history(self, filename: str = "metrics.csv") -> None:
        """Save metrics history to CSV."""
        import csv
        
        filepath = os.path.join(self.log_dir, filename)
        
        # Get all unique steps
        all_steps = set()
        for values in self.metrics_history.values():
            for step, _ in values:
                all_steps.add(step)
        all_steps = sorted(all_steps)
        
        # Build data rows
        metric_names = list(self.metrics_history.keys())
        
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step"] + metric_names)
            
            for step in all_steps:
                row = [step]
                for name in metric_names:
                    values = dict(self.metrics_history[name])
                    row.append(values.get(step, ""))
                writer.writerow(row)
    
    def close(self) -> None:
        """Close the logger."""
        if self.writer is not None:
            self.writer.close()
        self.save_history()
