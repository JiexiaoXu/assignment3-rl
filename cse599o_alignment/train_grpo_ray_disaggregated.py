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
from cse599o_alignment.train_grpo_ray_colocated import keyword_inclusion_reward_fn, compute_response_log_probs



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
        self.q = list[Trajectory]

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

    def sample(self, k: int):
        # TODO: sample k trajectories for training
        if len(self.data) == 0:
            return []
        k = min(k, len(self.data))
        indices = np.random.choice(len(self.data), size=k, replace=False)
        return [self.data[i] for i in indices]


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
            traj = ray.get(self.traj_q.get().remote())
            if traj is None:
                asyncio.sleep(0.1)
                continue

            for prompt, response in zip(traj.prompts, traj.responses):
                keyword = prompt.strip().split()[-1]
                reward = keyword_inclusion_reward_fn(response, keyword)
                traj.rewards.append(reward)
            self.replay_buf.put.remote(traj)
            

    def stop(self):
        self.running = False


@ray.remote
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

    def step(self):
        """One GRPO/PPO-style update step."""
        # TODO: sample from replay buffer, compute advantages, update model
        # This should implement GRPO policy gradient updates for text generation
        sampled_trajectories = ray.get(self.replay_buf.sample.remote(k=8))

        if not sampled_trajectories:
            return 0.0

        rollout_responses: List[str] = []
        repeated_ground_truths: List[List[str]] = []
        old_log_probs_list: List[torch.Tensor] = []
        policy_log_probs_list: List[torch.Tensor] = []
        response_masks: List[torch.Tensor] = []

        # Compute advantages
        for traj in sampled_trajectories:
            for prompt, response, old_lp in zip(traj.prompts, traj.responses, traj.log_probs):
                rollout_responses.append(response)
                keyword = prompt.strip().split()[-1]
                repeated_ground_truths.append([keyword])
                policy_lp, mask = compute_response_log_probs(prompt, response, self.learner_model)
                policy_log_probs_list.append(policy_lp)
                response_masks.append(mask)
                old_log_probs_list.append(old_lp.to(self.device))

        advantages, _, _ = compute_group_normalized_reward(
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            reward_fn=keyword_inclusion_reward_fn,
            group_size=G,
            normalized_by_std=USE_STD_NORMALIZATION,
            advantage_eps=ADVANTAGE_EPS,
        )

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
        return float(loss.item())

    def get_weights(self):
        # TODO: Return model weights for synchronization with Generator
        return {k: v.cpu() for k, v in self.learner_model.state_dict().items()}

    def get_version(self):
        return self.version


@ray.remote
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
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.traj_q = traj_q
        self.version = 0

    def generate(self, prompts: List[str]):
        """Generate text responses and send to Scorer."""
        from cse599o_alignment.train_grpo_ray_colocated import generate_text

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
                    top_p_threshold=0.9,
                )
                responses.append(response)
                # Store per-token log probs (old policy) for the generated response
                log_probs.append(resp_log_probs)

            trajectory = Trajectory(
                prompts=prompt_group,
                responses=responses,
                rewards=torch.zeros(G),  # placeholder, to be filled by Scorer
                log_probs=[lp.detach() for lp in log_probs],
            )
            self.traj_q.put.remote(trajectory)


    def update(self, weights: Dict, version: int):
        """Load updated learner weights."""
        # TODO: Update model weights from learner
        sd = self.actor_model.state_dict()
        for n, w in weights.items():
            sd[n] = w.to(self.device)
        self.actor_model.load_state_dict(sd)
        self.version = version


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
    for step in range(num_steps):
        prompt_batch = np.random.choice(prompts, size=2, replace=False).tolist()
        generator.generate.remote(prompt_batch)
        time.sleep(0.1)
        # Learner update
        loss = ray.get(learner.step.remote())
        weights = ray.get(learner.get_weights.remote())
        version = ray.get(learner.get_version.remote())
        generator.update.remote(weights, version)
        print(f"[Step {step+1}] Loss={loss:.4f}, Version={version}")

    scorer.stop.remote()


def run_once(num_steps: int = 3):
    """Entry point for training."""
    run_training(num_steps)



# ===================== Entry point =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True)
    run_once(num_steps=args.steps)
