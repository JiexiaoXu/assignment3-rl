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
CHECKPOINT_PATH = "/local1/jiexiao/checkpoint/checkpoint.pth"

N_GRPO_STEPS: int = 100
LEARNING_RATE: float = 5e-4
SAMPLING_TEMPERATURE: float = 1.0
SAMPLING_MAX_TOKENS: int = 60
ADVANTAGE_EPS: float = 1e-8
LOSS_TYPE: str = "grpo_clip"
USE_STD_NORMALIZATION: bool = True
SAMPLING_TOP_P: float = 0.9


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_tokenizer():
    encoding = tiktoken.get_encoding("gpt2")
    vocab = {i: encoding.decode_single_token_bytes(i) for i in range(encoding.n_vocab)}
    merges = list(encoding._mergeable_ranks.items())
    special_tokens = ["<|endoftext|>"]
    tokenizer = BPETokenizer(vocab, merges, special_tokens)
    return tokenizer

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
            token_log_prob = torch.log(softmax_logits[sampled_token] + 1e-8)
            log_probs.append(token_log_prob.squeeze())

            accum_output = torch.cat([accum_output, sampled_token], dim=0)
            if sampled_token.item() == eos_token_id:
                break


        # Decode the generated text
        prompt_ids = tokenizer.encode(prompt)
        response_tokens = accum_output[len(prompt_ids):]
        completion_text = tokenizer.decode(response_tokens.tolist())
        response_log_probs = torch.stack(log_probs) if log_probs else torch.tensor([], device=accum_output.device)
    return completion_text, response_log_probs


def compute_response_log_probs(
    prompt: str,
    response: str,
    model: TransformerLM,
    temperature: float = SAMPLING_TEMPERATURE,
    top_p_threshold: float = SAMPLING_TOP_P,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token log-probs of a response under a truncated distribution."""
    device = get_device()
    tokenizer = get_tokenizer()

    prompt_ids = tokenizer.encode(prompt)
    response_ids = tokenizer.encode(response)
    all_ids = prompt_ids + response_ids

    if len(all_ids) < 2:
        empty = torch.zeros(1, device=device)
        return empty, empty

    input_ids = torch.tensor(all_ids[:-1], device=device, dtype=torch.long).unsqueeze(0)
    target_ids = torch.tensor(all_ids[1:], device=device, dtype=torch.long).unsqueeze(0)
    logits = model(input_ids)  # (1, seq_len-1, vocab)

    def filtered_log_prob(step_logits, target_token):
        scaled = step_logits / max(temperature, 1e-8)
        probs = torch.softmax(scaled, dim=-1)
        if top_p_threshold < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            to_remove = cumulative > top_p_threshold
            to_remove[..., 1:] = to_remove[..., :-1].clone()
            to_remove[..., 0] = False
            probs[sorted_idx[to_remove]] = 0.0
            probs = probs / probs.sum(dim=-1, keepdim=True)
        return torch.log(probs[target_token] + 1e-8)

    token_log_probs = []
    for step in range(target_ids.shape[1]):
        token_log_probs.append(filtered_log_prob(logits[0, step, :], target_ids[0, step]))

    token_log_probs = torch.stack(token_log_probs).unsqueeze(0)  # (1, seq_len-1)

    response_start = max(len(prompt_ids) - 1, 0)
    response_log_probs = token_log_probs[:, response_start:]
    response_mask = torch.ones_like(response_log_probs, dtype=torch.float32)
    return response_log_probs.squeeze(0), response_mask.squeeze(0)

# ===================== Data container =====================

class Trajectory:
    """Stores a single rollout trajectory for text generation"""

    def __init__(
        self,
        prompts: List[str],  # shape: [G]
        responses: List[str],  # shape: [G]
        rewards: torch.Tensor,  # shape: [G]
        log_probs: List[torch.Tensor],  # per-response token log-probs shape: [G][seq_len]
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
        self.actor_model = TransformerLM(
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
        load_checkpoint(CHECKPOINT_PATH, self.actor_model, None)
        self.tokenizer = get_tokenizer()

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
                response, resp_log_probs = generate_text(
                    prompt=prompt,
                    model=self.actor_model,
                    tokenizer=self.tokenizer,
                    context_length=CONTEXT_LENGTH,
                    max_gen_length=SAMPLING_MAX_TOKENS,
                    temperature=SAMPLING_TEMPERATURE,
                    top_p_threshold=SAMPLING_TOP_P,
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
        self.frozen_model = TransformerLM(
            vocab_size=VOCAB_SIZE,
            context_length=CONTEXT_LENGTH,
            num_layers=NUM_LAYERS,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            dff=D_FF,
            theta=THETA,
            dtype=torch.float32,
            device=get_device(),
        )
        load_checkpoint(CHECKPOINT_PATH, self.frozen_model, None)
        self.frozen_model.eval()
        for param in self.frozen_model.parameters():
            param.requires_grad = False

        # TODO: Initialize the same TransformerLM model as Generator
        self.learner_model = TransformerLM(
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
        load_checkpoint(CHECKPOINT_PATH, self.learner_model, None)
        self.optimizer = AdamW(
            self.learner_model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=0.01,
        )

    def compute_advantages(self, trajectories: List[Trajectory]) -> torch.Tensor:
        """Compute advantages for GRPO."""
        advantage_list: List[float] = []

        for traj in trajectories:
            rewards = traj.rewards.to(self.device)
            if rewards.numel() == 0:
                continue
            group_mean = rewards.mean()
            if USE_STD_NORMALIZATION and rewards.numel() > 1:
                group_std = rewards.std(unbiased=True)
                denom = group_std + ADVANTAGE_EPS
            else:
                denom = torch.tensor(1.0, device=self.device)
            advantage_list.extend(((rewards - group_mean) / denom).tolist())

        return torch.tensor(advantage_list, device=self.device)
    
    def update_policy(self, trajectories: List[Trajectory], k=1) -> float:
        """Perform one policy update step."""
        # TODO: Implement GRPO/PPO policy update
        # 1. Compute advantages
        # 2. Compute policy gradient loss
        # 3. Perform optimizer step
        # 4. Return loss value

        def pad_to_len(t: torch.Tensor, length: int) -> torch.Tensor:
            if t.numel() == length:
                return t
            return F.pad(t, (0, length - t.numel()))

        if not trajectories:
            return 0.0, 0.0

        # Compute Advantages
        advantages = self.compute_advantages(trajectories)

        policy_log_probs_list = []
        old_log_probs_list = []
        ref_log_probs_list = []
        response_masks = []

        # Compute new log probs
        for traj in trajectories:
            for prompt, response, old_lp in zip(traj.prompts, traj.responses, traj.log_probs):
                policy_lp, mask = compute_response_log_probs(
                    prompt,
                    response,
                    self.learner_model,
                    temperature=SAMPLING_TEMPERATURE,
                    top_p_threshold=SAMPLING_TOP_P,
                )
                policy_log_probs_list.append(policy_lp)
                with torch.no_grad():
                    ref_lp, _ = compute_response_log_probs(
                        prompt,
                        response,
                        self.frozen_model,
                        temperature=SAMPLING_TEMPERATURE,
                        top_p_threshold=SAMPLING_TOP_P,
                    )
                    ref_log_probs_list.append(ref_lp.to(self.device))
                response_masks.append(mask)
                old_log_probs_list.append(old_lp.to(self.device))

        max_len = max(lp.shape[0] for lp in policy_log_probs_list)

        policy_log_probs = torch.stack([pad_to_len(lp, max_len) for lp in policy_log_probs_list])
        old_log_probs = torch.stack([pad_to_len(lp, max_len) for lp in old_log_probs_list])
        ref_log_probs = torch.stack([pad_to_len(lp, max_len) for lp in ref_log_probs_list])
        response_mask = torch.stack([pad_to_len(mask, max_len) for mask in response_masks])

        self.optimizer.zero_grad()
        loss, _ = grpo_microbatch_train_step(
            policy_log_probs=policy_log_probs,
            response_mask=response_mask,
            gradient_accumulation_steps=k,
            loss_type=LOSS_TYPE,
            advantages=advantages.unsqueeze(-1).to(self.device),
            old_log_probs=old_log_probs,
            cliprange=0.2,
        )

        self.optimizer.step()

        with torch.no_grad():
            kl_per_token = (policy_log_probs - ref_log_probs)
            kl_batch = (kl_per_token * response_mask).sum(dim=1)  # sum over tokens per rollout
            kl_value = kl_batch.mean()

        return float(loss.item()), float(kl_value.item())


# ===================== Combined Actor =====================

@ray.remote(num_gpus=1)
class ColocatedWorker(Generator, Learner):
    """Combined Generator and Learner in a single Ray actor."""
    def __init__(self):
        Generator.__init__(self)
        Learner.__init__(self)
        self.step_count = 0
        self.gen_times = []
        self.learn_times = []
        self.sync_times = []
        self.kl_values = []
        self.total_samples = 0
        self.start_time = time.time()

    
    def training_step(self, prompts: List[str], k=1) -> Dict[str, Any]:
        """Perform one complete training step: generate rollout + update policy."""
        # Generate trajectories for the batch of prompts
        gen_start = torch.cuda.Event(enable_timing=True)
        gen_end = torch.cuda.Event(enable_timing=True)
        learn_start = torch.cuda.Event(enable_timing=True)
        learn_end = torch.cuda.Event(enable_timing=True)
        sync_start = torch.cuda.Event(enable_timing=True)
        sync_end = torch.cuda.Event(enable_timing=True)
    
        gen_start.record()
        trajectories = self.generate_trajectories(prompts)
        gen_end.record()
        torch.cuda.synchronize()
        self.gen_times.append(gen_start.elapsed_time(gen_end))

        
        # Update policy using GRPO
        learn_start.record()
        loss, kl_value = self.update_policy(trajectories, k=k)
        learn_end.record()
        torch.cuda.synchronize()
        self.learn_times.append(learn_start.elapsed_time(learn_end))

        num_samples_step = 0
        for traj in trajectories:
            # each response in traj.responses is one sample
            num_samples_step += len(traj.responses)

        # Weight Synchronization 
        sync_start.record()
        self.actor_model.load_state_dict(self.learner_model.state_dict())
        sync_end.record()
        torch.cuda.synchronize()
        self.sync_times.append(sync_start.elapsed_time(sync_end))

        self.kl_values.append(kl_value)

        self.step_count += 1
        self.total_samples += num_samples_step
        elapsed_time = time.time() - self.start_time
        samples_per_sec = self.total_samples / elapsed_time
        
        return {
            'step': self.step_count,
            'loss': loss,
            'num_trajectories': len(trajectories),
            'avg_reward': float(torch.cat([traj.rewards for traj in trajectories]).mean()) if trajectories else 0.0,
            'samples_per_sec': samples_per_sec,
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current training statistics."""
        return {
            'step_count': self.step_count,
            'model_parameters': sum(p.numel() for p in self.actor_model.parameters()),
            'avg_gen_time_ms': np.mean(self.gen_times) if self.gen_times else 0.0,
            'avg_learn_time_ms': np.mean(self.learn_times) if self.learn_times else 0.0,
            'avg_sync_time_ms': np.mean(self.sync_times) if self.sync_times else 0.0,
            'kl_values': self.kl_values,
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
            prompt_batch = np.random.choice(prompts, size=4, replace=False).tolist()
            result = ray.get(worker.training_step.remote(prompt_batch, k=1))

            if step != 0 and step % 2 == 0:

                print(f"Worker Step {result['step']}: Loss={result['loss']:.4f}, Avg Reward={result['avg_reward']:.4f}, Samples/sec={result['samples_per_sec']:.2f}")
                worker_stats = ray.get(worker.get_statistics.remote())
                print("Worker Statistics:")
                for k, v in worker_stats.items():
                    print(f"  {k}: {v}")
                    
                    # plot KL values if needed
                    import matplotlib.pyplot as plt
                    if worker_stats['kl_values']:
                        plt.plot(worker_stats['kl_values'])
                        plt.title(f"Worker KL Divergence over Steps")
                        plt.xlabel("Step")
                        plt.ylabel("KL Divergence")
                        plt.savefig(f"kl_divergence.png")


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
    
    ray.init(runtime_env={
        "excludes": [
            ".git/**",                           # git metadata and objects
            ".venv/**",                          # virtual environment
            "submission_*/**",                   # submission folders (6.9GB)
            "checkpoint/**",                     # checkpoint folder (731MB)
            "tests/fixtures/**",                 # test fixtures (large model files)
            "wandb/**",                          # wandb logs
            "*.nsys-rep",                        # profiling files
            "*.pt", "*.pth", "*.safetensors",   # model weight files
            "*.tar", "*.zip", "*.gz",           # archives
            "__pycache__/**",                   # Python cache
            "*.egg-info/**"                     # package info
        ]
    }, ignore_reinit_error=True)
    
    try:
        run_once(num_steps=args.steps, num_workers=args.workers)
    finally:
        ray.shutdown()
