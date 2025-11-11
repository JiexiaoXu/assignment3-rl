"""
GRPO Skeleton: Colocated Synchronous Training Loop (Simplified)
--------------------------------------------------------------
Students should complete the TODO parts to:
 - implement rollout generation with reward computation using TransformerLM
 - perform policy updates using GRPO algorithm
 - implement keyword inclusion reward function

This version combines Generator and Learner into a single actor for simplified
synchronous training without replay buffer, training directly on each trajectory.
"""

import argparse
import asyncio
import ray
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import tiktoken
import time
from typing import List, Dict, Any
import numpy as np

from cse599o_basics.model import TransformerLM
from cse599o_basics.optimizer import AdamW
from cse599o_alignment.grpo import (
    compute_group_normalized_reward,
    grpo_microbatch_train_step,
    gradient_clipping
)


# ===================== Basic setup =====================

G = 4  # group size (number of responses per prompt)
VOCAB_SIZE = tiktoken.get_encoding("gpt2").n_vocab
CONTEXT_LENGTH = 256
NUM_LAYERS = 4
D_MODEL = 512
NUM_HEADS = 16
D_FF = 1344
THETA = 10000
CHECKPOINT_PATH = "/homes/iws/cxyzhao/student-version/assignment5-alignment/checkpoint/999"


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================== Data container =====================

class Trajectory:
    """Stores a single rollout trajectory for text generation"""

    def __init__(
        self,
        prompts: List[str],  # shape: [G]
        responses: List[str],  # shape: [G]
        rewards: torch.Tensor,  # shape: [G]
        log_probs: torch.Tensor,  # shape: [G]
        values: Optional[torch.Tensor] = None,  # shape: [G]
    ):
        self.prompts = prompts
        self.responses = responses
        self.rewards = rewards
        self.log_probs = log_probs
        self.values = values


# ===================== Base classes (no @ray.remote) =====================

class Generator:
    """Base class for text generation using TransformerLM"""

    def __init__(self):
        self.device = get_device()
        # TODO: Initialize the TransformerLM model
        # self.model = TransformerLM(...)
        # self.model.load_checkpoint(CHECKPOINT_PATH)
        # self.tokenizer = tiktoken.get_encoding("gpt2")

    def generate_trajectories(self, prompts: List[str]) -> List[Trajectory]:
        """
        Generate G responses for each prompt using TransformerLM.

        TODO: Implement this method
        - For each prompt, generate G responses using self.model
        - Calculate log probabilities for generated tokens
        - Return list of Trajectory objects with prompts, responses, log_probs
        """
        raise NotImplementedError("Students should implement this method")


class Learner:
    """Base learner class for policy gradient updates using TransformerLM."""
    def __init__(self):
        self.device = get_device()
        # TODO: Initialize the same TransformerLM model as Generator
        # self.model = TransformerLM(...)
        # self.model.load_checkpoint(CHECKPOINT_PATH)
        # self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
    

    
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
    
    async def training_step(self, prompts: List[str]) -> Dict[str, Any]:
        """Perform one complete training step: generate rollout + update policy."""
        # Generate trajectories for the batch of prompts
        trajectories = self.generate_trajectories(prompts)
        
        # Update policy using GRPO
        loss = self.update_policy(trajectories)
        
        self.step_count += 1
        
        return {
            'step': self.step_count,
            'loss': loss,
            'num_trajectories': len(trajectories),
            'avg_reward': float(torch.cat([traj.rewards for traj in trajectories]).mean()) if trajectories else 0.0
        }
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get current training statistics."""
        return {
            'step_count': self.step_count,
            'model_parameters': sum(p.numel() for p in self.model.parameters()) if hasattr(self, 'model') else 0
        }


# ===================== Training loop =====================

async def run_training(num_steps: int = 10, num_workers: int = 1):
    """Run colocated GRPO training with text generation."""
    
    # Create workers  
    workers = [ColocatedWorker.remote() for _ in range(num_workers)]
    
    # TODO: Define training prompts
    # training_prompts = ["Write a story about", "Explain the concept of", "Describe how to"]
    
    print(f"Starting GRPO training with {num_workers} colocated workers for {num_steps} steps...")
    
    for step in range(num_steps):
        # TODO: Sample prompts for each worker
        # For now, using placeholder prompts
        prompt_batches = [["Sample prompt"] * G for _ in range(num_workers)]
        
        # Run training step on all workers in parallel
        futures = [
            worker.training_step.remote(prompts) 
            for worker, prompts in zip(workers, prompt_batches)
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
