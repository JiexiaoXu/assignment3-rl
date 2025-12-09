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
from typing import List, Dict, Any, Optional
import numpy as np

from cse599o_basics.tokenizer import BPETokenizer
from cse599o_basics.transformer_lm import TransformerLM
from cse599o_basics.optimizer import AdamW
from cse599o_basics.training_util import load_checkpoint
from cse599o_alignment.grpo import (
    compute_group_normalized_reward,
    grpo_microbatch_train_step,
    compute_grpo_clip_loss
)


# ===================== Basic setup =====================

G = 4  # group size (number of responses per prompt)
VOCAB_SIZE = tiktoken.get_encoding("gpt2").n_vocab
CONTEXT_LENGTH = 512
NUM_LAYERS = 10
D_MODEL = 512
NUM_HEADS = 16
D_FF = 1344
THETA = 10000
CHECKPOINT_PATH = "/local1/jiexiao/checkpoint/checkpoint01_step8000.pth"

N_GRPO_STEPS: int = 100
LEARNING_RATE: float = 5e-4
SAMPLING_TEMPERATURE: float = 0.8
SAMPLING_MAX_TOKENS: int = 60
ADVANTAGE_EPS: float = 1e-8
LOSS_TYPE: str = "grpo_clip"
USE_STD_NORMALIZATION: bool = True


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def keyword_inclusion_reward_fn(response: str, keywords: List[str]) -> float:
    if not keywords:
        return 0.0
    
    response_lower = response.lower()
    keywords_lower = [kw.lower() for kw in keywords]

    # reward only if all keywords are included
    if all(kw in response_lower for kw in keywords_lower):
        return {
            "reward": 1.0,
        }
    else:
        return {
            "reward": 0.0,
        }

def generate_text(
    prompt: str,
    model: TransformerLM,
    tokenizer: BPETokenizer,
    context_length: int,
    max_gen_length: int,
    temperature: float,
    top_p_threshold: float,
) -> tuple[str, torch.Tensor]:
    def temperature_softmax(logits, temperature):
        logits = logits / max(temperature, 1e-8)
        return torch.softmax(logits, dim=-1)

    def top_p_filtering(softmax_logits, top_p=top_p_threshold):
        sorted_logits, sorted_indices = torch.sort(softmax_logits, descending=True)
        cumulative_probs = torch.cumsum(sorted_logits, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        softmax_logits[indices_to_remove] = 0.0

        # Renormalize the probabilities
        softmax_logits = softmax_logits / softmax_logits.sum(dim=-1, keepdim=True)
        return softmax_logits

    model.eval()
    with torch.no_grad():
        # Tokenize the prompt
        accum_output = tokenizer.encode(prompt)
        accum_output = torch.tensor(accum_output, dtype=torch.long, device="cuda")
        eos_token_id = tokenizer.encode("<|endoftext|>")[0]

        log_probs = []

        # Generate tokens
        while True:
            if len(accum_output) >= max_gen_length:
                break

            input_tensor = accum_output[-context_length:].unsqueeze(0)
            logits = model(input_tensor)

            # Only take last token's logits
            last_token_logits = logits[0, -1, :]
            softmax_logits = temperature_softmax(last_token_logits, temperature)
            softmax_logits = top_p_filtering(softmax_logits, top_p=top_p_threshold)

            # sample from last token's distribution, stop if EOS token is generated
            sampled_token = torch.multinomial(softmax_logits, num_samples=1)

            # Get log probabilities
            token_log_prob = torch.log(softmax_logits[sampled_token])
            log_probs.append(token_log_prob)

            if sampled_token.item() == eos_token_id:
                break

            accum_output = torch.cat([accum_output, sampled_token], dim=0)

        # Decode the generated text
        generated_text = tokenizer.decode(accum_output.tolist())
        total_log_prob = torch.sum(torch.stack(log_probs)) if log_probs else torch.tensor(0.0, device=accum_output.device)
    return generated_text, total_log_prob

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
        self.model = TransformerLM(
            vocab_size=VOCAB_SIZE,
            context_length=CONTEXT_LENGTH,
            num_layers=NUM_LAYERS,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            dff=D_FF,
            theta=THETA,
            dtype=torch.float32,
            device=self.device,
        )
        load_checkpoint(CHECKPOINT_PATH, self.model, None)
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def generate_trajectories(self, prompts: List[str]) -> List[Trajectory]:
        """
        Generate G responses for each prompt using TransformerLM.

        TODO: Implement this method
        - For each prompt, generate G responses using self.model
        - Calculate log probabilities for generated tokens
        - Return list of Trajectory objects with prompts, responses, log_probs
        """

        trajectories: List[Trajectory] = []

        for prompt in prompts:
            # use the last work of prompt as keywords
            keyword = prompt.strip().split()[-1]
            prompts = [keyword] * G

            responses = []
            log_probs = []
            for _ in range(G):
                response, log_prob = generate_text(
                    prompt=prompt,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    context_length=CONTEXT_LENGTH,
                    max_gen_length=SAMPLING_MAX_TOKENS,
                    temperature=SAMPLING_TEMPERATURE,
                    top_p_threshold=0.9,
                )
                responses.append(response)
                log_probs.append(log_prob)

            trajectory = Trajectory(
                prompts=prompts,
                responses=responses,
                rewards=torch.zeros(G, device=self.device),  # Placeholder
                log_probs=log_probs,
            )
            trajectories.append(trajectory)

        return trajectories

class Learner:
    """Base learner class for policy gradient updates using TransformerLM."""
    def __init__(self):
        self.device = get_device()
        # TODO: Initialize the same TransformerLM model as Generator
        self.model = TransformerLM(
            vocab_size=VOCAB_SIZE,
            context_length=CONTEXT_LENGTH,
            num_layers=NUM_LAYERS,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            d_ff=D_FF,
            theta=THETA,
            dtype=torch.float32,
            device=self.device,
        )
        load_checkpoint(CHECKPOINT_PATH, self.model, None)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=1e-5,
            weight_decay=0.01,
        )
    

    
    def compute_advantages(self, trajectories: List[Trajectory]) -> torch.Tensor:
        """Compute advantages for GRPO."""
        # TODO: Implement GRPO advantage computation
        # This should implement the group-relative advantage computation
        # that's central to GRPO algorithm
        rollout_responses = torch.cat([traj.responses for traj in trajectories])
        repeated_ground_truths = torch.cat([traj.prompts for traj in trajectories])
        group_size = G

        advantages, _, _ = compute_group_normalized_reward(
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            reward_fn=keyword_inclusion_reward_fn,
            group_size=group_size,
            normalized_by_std=USE_STD_NORMALIZATION,
            advantage_eps=ADVANTAGE_EPS,
        )

        return advantages
    
    def update_policy(self, trajectories: List[Trajectory]) -> float:
        """Perform one policy update step."""
        # TODO: Implement GRPO/PPO policy update
        # 1. Compute advantages
        # 2. Compute policy gradient loss
        # 3. Perform optimizer step
        # 4. Return loss value

        advantages = self.compute_advantages(trajectories)
        loss = grpo_microbatch_train_step(
            model=self.model,
            optimizer=self.optimizer,
            trajectories=trajectories,
            advantages=advantages,
            loss_type=LOSS_TYPE,
            cliprange=0.2,
            max_grad_norm=1.0,
        )
        
        return loss


# ===================== Combined Actor =====================

@ray.remote
class ColocatedWorker(Generator, Learner):
    """Combined Generator and Learner in a single Ray actor."""
    def __init__(self):
        Generator.__init__(self)
        Learner.__init__(self)
        self.step_count = 0
    
    def training_step(self, prompts: List[str]) -> Dict[str, Any]:
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
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current training statistics."""
        return {
            'step_count': self.step_count,
            'model_parameters': sum(p.numel() for p in self.model.parameters()) if hasattr(self, 'model') else 0
        }


# ===================== Training loop =====================

def run_training(num_steps: int = 10, num_workers: int = 1):
    """Run colocated GRPO training with text generation."""
    
    # Create workers  
    workers = [ColocatedWorker.remote() for _ in range(num_workers)]    

    # TODO: Define training prompts
    base_prompt = "Generate a story that includes "
    with open("/homes/iws/jiexiao/cse599o/hw3/assignment3-rl/cse599o_alignment/prompts/keywords.txt", "r") as f:
        keywords = [line.strip() for line in f.readlines()]
    prompts = [base_prompt + kw for kw in keywords]


    for step in range(num_steps):
        for worker in workers:
            # Sample a batch of prompts for each worker
            prompt_batch = np.random.choice(prompts, size=2, replace=False).tolist()
            result = ray.get(worker.training_step.remote(prompt_batch))
            print(f"Worker Step {result['step']}: Loss={result['loss']:.4f}, Avg Reward={result['avg_reward']:.4f}")


def run_once(num_steps: int = 10, num_workers: int = 1):
    """Entry point for training."""
    run_training(num_steps, num_workers)


# ===================== Entry point =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10, 
                       help="Number of training steps")
    parser.add_argument("--workers", type=int, default=1, 
                       help="Number of colocated workers")
    args = parser.parse_args()
    
    ray.init(
        ignore_reinit_error=True,
        runtime_env={
            "working_dir": ".", 
            "excludes": [
                ".venv",        
                ".git",         
                "__pycache__",
                "tests/fixtures" 
            ]
        }
    )
    
    try:
        run_once(num_steps=args.steps, num_workers=args.workers)
    finally:
        ray.shutdown()
