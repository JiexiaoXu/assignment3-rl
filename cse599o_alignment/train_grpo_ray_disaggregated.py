"""
GRPO Skeleton: Minimal Asynchronous Training Loop
------------------------------------------------
Students should complete the TODO parts to:
 - implement rollout generation with text generation using TransformerLM (Generator)
 - compute rewards for text responses (Scorer)
 - perform policy updates using GRPO algorithm (Learner)
 - synchronize model weights between Generator and Learner
"""

import asyncio
import argparse
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
from cse599o_alignment.train_grpo_ray_colocated import keyword_inclusion_reward_fn, compute_response_log_probs, get_tokenizer



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
SAMPLING_TEMPERATURE: float = 0.8
SAMPLING_MAX_TOKENS: int = 60
ADVANTAGE_EPS: float = 1e-8
LOSS_TYPE: str = "grpo_clip"
USE_STD_NORMALIZATION: bool = True
SAMPLING_TOP_P: float = 0.9


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================== Data container =====================

class Trajectory:
    """Stores a single rollout trajectory for text generation"""
    def __init__(
        self,
        version: int,
        prompts: List[str],  # shape: [G]
        responses: List[str],  # shape: [G]
        rewards: torch.Tensor,  # shape: [G]
        log_probs: List[torch.Tensor],  # per-response token log-probs shape: [G][seq_len]
        values: Optional[torch.Tensor] = None,  # shape: [G]
    ):
        self.version = version
        self.prompts = prompts
        self.responses = responses
        self.rewards = rewards
        self.log_probs = log_probs
        self.values = values


# ===================== Actors =====================

@ray.remote
class TrajectoryQueue:
    """Buffer between Generator and Scorer."""
    def __init__(self):
        self.q = []

    def put(self, traj: Trajectory):
        # TODO: implement trajectory queuing
        self.q.append(traj)

    def get(self):
        # TODO: implement trajectory retrieval with timeout
        if len(self.q) == 0:
            return None
        return self.q.pop(0)


@ray.remote
class ReplayBuffer:
    """Stores scored trajectories for the Learner."""
    def __init__(self):
        self.data = []

    def put(self, traj: Trajectory):
        # TODO: store completed trajectories here
        self.data.append(traj)

    def sample(self, k: int, version: int):
        # TODO: sample k trajectories for training
        if len(self.data) == 0:
            return []
        
        valid_traj = [traj for traj in self.data if 0 <= version - traj.version <= 1]
        if len(valid_traj) == 0:
            return []
        
        k = min(k, len(valid_traj))
        indices = np.random.choice(len(valid_traj), size=k, replace=False)
        return [valid_traj[i] for i in indices]

    def size(self) -> int:
        return len(self.data)


@ray.remote
class Scorer:
    """Assigns rewards to generated text responses."""
    def __init__(self, traj_q, replay_buf):
        self.traj_q = traj_q
        self.replay_buf = replay_buf
        self.running = False

    def run(self):
        """Continuously fetch trajectories, assign rewards, and store them."""
        self.running = True
        while self.running:
            # TODO: Get trajectories from queue, compute rewards, store in replay buffer
            # This should implement a reward function that evaluates text quality
            # e.g., keyword inclusion, safety, helpfulness, etc.
            traj = ray.get(self.traj_q.get.remote())
            if traj is None:
                time.sleep(0.1)
                continue
            
            rewards = []
            for prompt, response in zip(traj.prompts, traj.responses):
                keyword = prompt.strip().split()[-1]
                reward_dict = keyword_inclusion_reward_fn(response, [keyword])
                rewards.append(reward_dict["reward"])
            traj.rewards = torch.tensor(rewards, dtype=torch.float32)
            self.replay_buf.put.remote(traj)
            

    def stop(self):
        self.running = False


@ray.remote(num_gpus=1)
class Learner:
    """Learns policy updates from the replay buffer using TransformerLM."""
    def __init__(self, replay_buf):
        self.device = get_device()
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
        self.version = 0
        self.replay_buf = replay_buf

        self.step_time = []

    def step(self):
        """One GRPO/PPO-style update step."""
        # TODO: sample from replay buffer, compute advantages, update model
        # This should implement GRPO policy gradient updates for text generation
        current_version = self.version
        sampled_trajectories = ray.get(self.replay_buf.sample.remote(k=2, version=current_version))

        if not sampled_trajectories:
            return 0.0
        
        start = time.time()

        old_log_probs_list: List[torch.Tensor] = []
        policy_log_probs_list: List[torch.Tensor] = []
        response_masks: List[torch.Tensor] = []
        advantages_list: List[float] = []

        for traj in sampled_trajectories:
            rewards = traj.rewards.to(self.device)
            if rewards.numel() == 0:
                continue
            group_mean = rewards.mean()
            if USE_STD_NORMALIZATION and rewards.numel() > 1:
                group_std = rewards.std(unbiased=True)
                denom = group_std + ADVANTAGE_EPS
            else:
                denom = torch.tensor(1.0, device=self.device)

            for reward, prompt, response, old_lp in zip(rewards, traj.prompts, traj.responses, traj.log_probs):
                policy_lp, mask = compute_response_log_probs(
                    prompt,
                    response,
                    self.learner_model,
                    temperature=SAMPLING_TEMPERATURE,
                    top_p_threshold=SAMPLING_TOP_P,
                )
                policy_log_probs_list.append(policy_lp)
                response_masks.append(mask)
                old_log_probs_list.append(old_lp.to(self.device))
                adv = (reward - group_mean) / denom
                advantages_list.append(adv.item())

        advantages = torch.tensor(advantages_list, device=self.device)
        if policy_log_probs_list == []:
            return 0.0

        # Update Policy
        max_len = max(lp.shape[0] for lp in policy_log_probs_list)

        def pad_to_len(t: torch.Tensor, length: int) -> torch.Tensor:
            if t.numel() == length:
                return t
            return torch.nn.functional.pad(t, (0, length - t.numel()))

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
        torch.nn.utils.clip_grad_norm_(self.learner_model.parameters(), 1.0)
        self.optimizer.step()

        self.version += 1

        end = time.time()
        self.step_time.append(end - start)

        return float(loss.item())

    def get_weights(self):
        # TODO: Return model weights for synchronization with Generator
        return {k: v.cpu() for k, v in self.learner_model.state_dict().items()}

    def get_version(self):
        return self.version
    
    def get_avg_step_time(self):
        if not self.step_time:
            return 0.0
        return sum(self.step_time) / len(self.step_time)


@ray.remote(num_gpus=1)
class Generator:
    """Generates text responses using TransformerLM policy."""
    def __init__(self, traj_q):
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
        self.traj_q = traj_q
        self.version = 0
        self.gen_time = []
        self.sync_time = []

    def generate(self, prompts: List[str]):
        """Generate text responses and send to Scorer."""
        from cse599o_alignment.train_grpo_ray_colocated import generate_text

        start = time.time()

        for prompt in prompts:
            prompt_group = [prompt] * G
            responses = []
            log_probs = []
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

            trajectory = Trajectory(
                version=self.version,
                prompts=prompt_group,
                responses=responses,
                rewards=torch.zeros(len(responses)),
                log_probs=[lp.detach() for lp in log_probs],
            )
            self.traj_q.put.remote(trajectory)

        end = time.time()
        self.gen_time.append(end - start)


    def update(self, weights: Dict, version: int):
        """Load updated learner weights."""
        # TODO: Update model weights from learner
        start = time.time()
        sd = self.actor_model.state_dict()
        for n, w in weights.items():
            sd[n] = w.to(self.device)
        self.actor_model.load_state_dict(sd)
        self.version = version
        end = time.time()
        self.sync_time.append(end - start)

    def get_stats(self):
        avg_gen_time = sum(self.gen_time) / len(self.gen_time) if self.gen_time else 0.0
        avg_sync_time = sum(self.sync_time) / len(self.sync_time) if self.sync_time else 0.0
        return avg_gen_time, avg_sync_time
            


# ===================== Training loop =====================

def run_training(num_steps: int = 3):
    """Run disaggregated GRPO training with text generation."""
    traj_q = TrajectoryQueue.remote()
    replay_buf = ReplayBuffer.remote()
    learner = Learner.remote(replay_buf)
    scorer = Scorer.remote(traj_q, replay_buf)
    generator = Generator.remote(traj_q)

    # TODO: Driver code for the training loop
    # Define training prompts
    base_prompt = "Generate a story that includes "
    from pathlib import Path
    keywords_path = Path(__file__).resolve().parent / "prompts" / "keywords.txt"
    with open(keywords_path, "r") as f:
        keywords = [line.strip() for line in f.readlines()]
    prompts = [base_prompt + kw for kw in keywords]

    scorer.run.remote()

    # Populate the buffer
    prompt_batch = np.random.choice(prompts, size=2, replace=False).tolist()
    generator.generate.remote(prompt_batch)
    # wait until scorer has placed something in replay buffer
    while ray.get(replay_buf.size.remote()) == 0:
        time.sleep(0.05)

    for step in range(num_steps):
        next_prompt = np.random.choice(prompts, size=2, replace=False).tolist()
        generator.generate.remote(next_prompt)

        # Learner update
        loss = 0.0
        # ensure learner has data; retry if buffer temporarily empty
        while True:
            loss = ray.get(learner.step.remote())
            if loss != 0.0:
                break
            time.sleep(0.05)

        weights = ray.get(learner.get_weights.remote())
        version = ray.get(learner.get_version.remote())
        generator.update.remote(weights, version)
        print(f"[Step {step+1}] Loss={loss:.4f}, Version={version}")

        if step % 2 == 0:
            avg_gen_time, avg_sync_time = ray.get(generator.get_stats.remote())
            avg_learn_time = ray.get(learner.get_avg_step_time.remote())
            print(f"  Avg Gen Time: {avg_gen_time:.4f}s, Avg Sync Time: {avg_sync_time:.4f}s, Avg Learn Time: {avg_learn_time:.4f}s")

    scorer.stop.remote()


def run_once(num_steps: int = 3):
    """Entry point for training."""
    run_training(num_steps)



# ===================== Entry point =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3)
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
    run_once(num_steps=args.steps)
