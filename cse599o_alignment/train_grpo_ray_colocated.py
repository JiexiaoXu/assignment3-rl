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
import torch.nn.functional as F
from torch.distributions import Categorical
import tiktoken
import time
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path

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

def keyword_inclusion_reward_fn(response: str, keywords: List[str]) -> Dict[str, float]:
    """
    Simple deterministic reward: 1.0 only if all keywords are present in the response.
    Matches the reward_fn signature used by compute_group_normalized_reward.
    """
    if not keywords:
        return {"reward": 0.0}

    response_lower = response.lower()
    keywords_lower = [kw.lower() for kw in keywords]
    reward = 1.0 if all(kw in response_lower for kw in keywords_lower) else 0.0
    return {"reward": reward}

def generate_text(
    prompt: str,
    model: TransformerLM,
    tokenizer: BPETokenizer,
    context_length: int,
    max_gen_length: int,
    temperature: float,
    top_p_threshold: float,
) -> tuple[str, torch.Tensor, torch.Tensor]:
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
        accum_output = torch.tensor(accum_output, dtype=torch.long, device=get_device())
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
            log_probs.append(token_log_prob.squeeze())

            if sampled_token.item() == eos_token_id:
                break

            accum_output = torch.cat([accum_output, sampled_token], dim=0)

        # Decode the generated text
        generated_text = tokenizer.decode(accum_output.tolist())
        response_len = len(accum_output) - len(tokenizer.encode(prompt))
        response_log_probs = torch.stack(log_probs) if log_probs else torch.tensor([], device=accum_output.device)
    return generated_text, accum_output, response_log_probs

# ===================== Data container =====================

class Trajectory:
    """Stores a single rollout trajectory for text generation"""

    def __init__(
        self,
        prompts: List[str],  # shape: [G]
        responses: List[str],  # shape: [G]
        rewards: torch.Tensor,  # shape: [G]
        log_probs: List[torch.Tensor],  # per-response token log-probs
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
            keyword = prompt.strip().split()[-1]
            prompt_group = [prompt] * G
            responses = []
            log_probs = []
            rewards = []
            for _ in range(G):
                response, token_ids, resp_log_probs = generate_text(
                    prompt=prompt,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    context_length=CONTEXT_LENGTH,
                    max_gen_length=SAMPLING_MAX_TOKENS,
                    temperature=SAMPLING_TEMPERATURE,
                    top_p_threshold=0.9,
                )
                responses.append(response)
                # Store per-token log probs (old policy) for the generated response
                log_probs.append(resp_log_probs)
                reward_dict = keyword_inclusion_reward_fn(response, [keyword])
                rewards.append(reward_dict["reward"])

            trajectory = Trajectory(
                prompts=prompt_group,
                responses=responses,
                rewards=torch.tensor(rewards, device=self.device),
                log_probs=[lp.detach() for lp in log_probs],
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
            lr=LEARNING_RATE,
            weight_decay=0.01,
        )

    def _compute_response_log_probs(
        self, prompt: str, response: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-token log-probabilities for the response portion."""
        prompt_ids = self.tokenizer.encode(prompt)
        response_ids = self.tokenizer.encode(response)
        all_ids = prompt_ids + response_ids

        if len(all_ids) < 2:
            empty = torch.zeros(1, device=self.device)
            return empty, empty

        input_ids = torch.tensor(all_ids[:-1], device=self.device, dtype=torch.long).unsqueeze(0)
        target_ids = torch.tensor(all_ids[1:], device=self.device, dtype=torch.long).unsqueeze(0)

        logits = self.model(input_ids)
        log_probs_all = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs_all.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)  # (1, seq_len-1)

        response_start = max(len(prompt_ids) - 1, 0)
        response_log_probs = token_log_probs[:, response_start:]
        response_mask = torch.ones_like(response_log_probs, dtype=torch.float32)
        return response_log_probs.squeeze(0), response_mask.squeeze(0)

    def compute_advantages(self, trajectories: List[Trajectory]) -> torch.Tensor:
        """Compute advantages for GRPO."""
        # TODO: Implement GRPO advantage computation
        # This should implement the group-relative advantage computation
        # that's central to GRPO algorithm
        rollout_responses: List[str] = []
        repeated_ground_truths: List[List[str]] = []

        for traj in trajectories:
            for prompt, response in zip(traj.prompts, traj.responses):
                rollout_responses.append(response)
                keyword = prompt.strip().split()[-1]
                repeated_ground_truths.append([keyword])

        advantages, _, _ = compute_group_normalized_reward(
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            reward_fn=keyword_inclusion_reward_fn,
            group_size=G,
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

        if not trajectories:
            return 0.0

        advantages = self.compute_advantages(trajectories)

        policy_log_probs_list = []
        old_log_probs_list = []
        response_masks = []

        for traj in trajectories:
            for prompt, response, old_lp in zip(traj.prompts, traj.responses, traj.log_probs):
                policy_lp, mask = self._compute_response_log_probs(prompt, response)
                policy_log_probs_list.append(policy_lp)
                response_masks.append(mask)
                old_log_probs_list.append(old_lp.to(self.device))

        max_len = max(lp.shape[0] for lp in policy_log_probs_list)

        def pad_to_len(t: torch.Tensor, length: int) -> torch.Tensor:
            if t.numel() == length:
                return t
            return F.pad(t, (0, length - t.numel()))

        policy_log_probs = torch.stack([pad_to_len(lp, max_len) for lp in policy_log_probs_list])
        old_log_probs = torch.stack([pad_to_len(lp, max_len) for lp in old_log_probs_list])
        response_mask = torch.stack([pad_to_len(mask, max_len) for mask in response_masks])

        self.optimizer.zero_grad()
        loss, _ = grpo_microbatch_train_step(
            policy_log_probs=policy_log_probs,
            response_mask=response_mask,
            gradient_accumulation_steps=1,
            loss_type=LOSS_TYPE,
            advantages=advantages.unsqueeze(-1).to(self.device),
            old_log_probs=old_log_probs,
            cliprange=0.2,
        )

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return float(loss.item())


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
    keywords_path = Path(__file__).resolve().parent / "prompts" / "keywords.txt"
    with open(keywords_path, "r") as f:
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
