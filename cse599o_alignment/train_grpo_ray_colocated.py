"""
GRPO Skeleton: Colocated Synchronous Training Loop
-------------------------------------------------
Students should complete the TODO parts to:
 - implement rollout generation with reward computation
 - perform policy updates using GRPO/PPO
 - manage replay buffer for training samples

This version combines Generator and Learner into a single actor for simplified
synchronous training, with reward computation happening directly in the generator.
"""

import argparse
import ray
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Dict, Any
import numpy as np


# ===================== Basic setup =====================

G = 4  # group size (number of actions per prompt)
STATE_DIM = 4
ACTION_DIM = 4


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================== Data container =====================

class Trajectory:
    """A single rollout sample with rewards."""
    def __init__(self, state, actions, logps, rewards):
        self.state = state
        self.actions = actions
        self.logps = logps
        self.rewards = rewards


# ===================== Base classes (no @ray.remote) =====================

class Generator:
    """Base generator class for rollout generation and reward computation."""
    def __init__(self):
        self.device = get_device()
        self.model = nn.Sequential(
            nn.Linear(STATE_DIM, 16),
            nn.Tanh(),
            nn.Linear(16, ACTION_DIM),
        ).to(self.device)
    
    def generate_rollout(self, state: torch.Tensor, num_actions: int = G) -> Trajectory:
        """Generate a rollout with actions, log-probs, and rewards."""
        # TODO: Implement rollout generation
        # 1. Sample actions from the current policy
        # 2. Compute log probabilities
        # 3. Compute rewards for the actions
        # 4. Return Trajectory object
        
        # Placeholder implementation
        actions = torch.randint(0, ACTION_DIM, (num_actions,), device=self.device)
        logps = torch.zeros(num_actions, device=self.device)
        rewards = torch.zeros(num_actions, device=self.device)
        
        return Trajectory(state, actions, logps, rewards)
    
    def compute_rewards(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute rewards for generated actions."""
        # TODO: Implement reward model
        # This is where you would implement your reward function
        # For now, return dummy rewards
        return torch.randn(actions.shape[0], device=self.device)


class Learner:
    """Base learner class for policy updates."""
    def __init__(self):
        self.device = get_device()
        self.model = nn.Sequential(
            nn.Linear(STATE_DIM, 16),
            nn.Tanh(),
            nn.Linear(16, ACTION_DIM),
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.replay_buffer = []
    
    def add_trajectory(self, trajectory: Trajectory):
        """Add trajectory to replay buffer."""
        # TODO: Implement replay buffer management
        # Consider buffer size limits, oldest sample removal, etc.
        pass
    
    def compute_advantages(self, trajectories: List[Trajectory]) -> torch.Tensor:
        """Compute advantages for GRPO."""
        # TODO: Implement GRPO advantage computation
        # This should implement the group-relative advantage computation
        # that's central to GRPO algorithm
        return torch.zeros(len(trajectories), device=self.device)
    
    def update_policy(self, trajectories: List[Trajectory]) -> float:
        """Perform one policy update step."""
        # TODO: Implement GRPO/PPO policy update
        # 1. Compute advantages
        # 2. Compute policy gradient loss
        # 3. Perform optimizer step
        # 4. Return loss value
        
        loss = torch.tensor(0.0, device=self.device)
        return float(loss.item())


# ===================== Combined Actor =====================

@ray.remote
class ColocatedWorker(Generator, Learner):
    """Combined Generator and Learner in a single Ray actor."""
    def __init__(self):
        Generator.__init__(self)
        Learner.__init__(self)
        self.step_count = 0
    
    async def training_step(self, state: torch.Tensor) -> Dict[str, Any]:
        """Perform one complete training step: generate rollout + update policy."""
        # Generate rollout with rewards
        trajectory = self.generate_rollout(state)
        
        # Add to replay buffer
        self.add_trajectory(trajectory)
        
        # Update policy if we have enough samples
        loss = 0.0
        if len(self.replay_buffer) >= G:  # Wait for at least one group
            # TODO: Sample trajectories for training
            sampled_trajectories = []  # Implement sampling logic
            loss = self.update_policy(sampled_trajectories)
        
        self.step_count += 1
        
        return {
            'step': self.step_count,
            'loss': loss,
            'trajectory_length': len(trajectory.actions),
            'avg_reward': float(trajectory.rewards.mean()) if len(trajectory.rewards) > 0 else 0.0
        }
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get current training statistics."""
        return {
            'step_count': self.step_count,
            'buffer_size': len(self.replay_buffer),
            'model_parameters': sum(p.numel() for p in self.model.parameters())
        }


# ===================== Training loop =====================

async def run_training(num_steps: int = 10, num_workers: int = 1):
    """Run colocated training with specified number of workers."""
    
    # Create workers
    workers = [ColocatedWorker.remote() for _ in range(num_workers)]
    
    print(f"Starting training with {num_workers} colocated workers for {num_steps} steps...")
    
    for step in range(num_steps):
        # Generate random states for each worker
        states = [torch.randn(STATE_DIM) for _ in range(num_workers)]
        
        # Run training step on all workers in parallel
        futures = [
            worker.training_step.remote(state) 
            for worker, state in zip(workers, states)
        ]
        
        # Wait for all workers to complete
        results = await asyncio.gather(*futures)
        
        # Print progress
        avg_loss = np.mean([r['loss'] for r in results])
        avg_reward = np.mean([r['avg_reward'] for r in results])
        
        print(f"Step {step + 1}/{num_steps}: "
              f"Avg Loss = {avg_loss:.4f}, "
              f"Avg Reward = {avg_reward:.4f}")
    
    # Get final statistics
    stats_futures = [worker.get_statistics.remote() for worker in workers]
    final_stats = await asyncio.gather(*stats_futures)
    
    print("\nFinal Statistics:")
    for i, stats in enumerate(final_stats):
        print(f"Worker {i}: {stats}")


def run_once(num_steps: int = 10, num_workers: int = 1):
    """Entry point for training."""
    import asyncio
    asyncio.run(run_training(num_steps, num_workers))


# ===================== Entry point =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10, 
                       help="Number of training steps")
    parser.add_argument("--workers", type=int, default=1, 
                       help="Number of colocated workers")
    args = parser.parse_args()
    
    ray.init(ignore_reinit_error=True)
    
    try:
        run_once(num_steps=args.steps, num_workers=args.workers)
    finally:
        ray.shutdown()
