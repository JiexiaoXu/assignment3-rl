"""
GRPO Algorithm Implementation - Assignment 3
========================================================
"""

import torch
from einops import repeat
from typing import Literal


def compute_group_normalized_reward(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalized_by_std
):
    """    
    Args:
        reward_fn: Callable[[str, str], dict[str, float]] Scores the rollout responses against
        the ground truths, producing a dict with keys "reward", "format_reward", and
        "answer_reward".
        
        rollout_responses: list[str] Rollouts from the policy. The length of this list is
        rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        
        repeated_ground_truths: list[str] The ground truths for the examples. The length of this
        list is rollout_batch_size, because the ground truth for each example is repeated
        group_size times.
        
        group_size: int Number of responses per question (group).
        
        advantage_eps: float Small constant to avoid division by zero in normalization.
        
        normalized_by_std: bool If True, divide by the per-group standard deviation; otherwise
        subtract only the group mean.
        
    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]].
        
        advantages: shape (rollout_batch_size,). Group-normalized rewards for each rollout
        response.
        
        raw_rewards: shape (rollout_batch_size,). Unnormalized rewards for each rollout
        response.
        
        metadata: your choice of other statistics to log (e.g. mean, std, max/min of rewards).
    """

    ## Get the raw rewards 
    advantage = []
    raw_rewards = []

    rollout_batch_size = len(rollout_responses)
    num_prompts = rollout_batch_size // group_size
    
    for response, gt in zip(rollout_responses, repeated_ground_truths):
        r = reward_fn(response, gt)
        raw_rewards.append(r["reward"])
    
    for i in range(num_prompts):
        group_rewards = raw_rewards[i*group_size:(i+1)*group_size]
        group_mean = sum(group_rewards) / group_size
        if normalized_by_std:
            variance = sum((r - group_mean) ** 2 for r in group_rewards) / (group_size - 1)
            group_std = variance ** 0.5
            denom = group_std + advantage_eps
        else:
            denom = 1.0
        
        for r in group_rewards:
            adv = (r - group_mean) / denom
            advantage.append(adv)

    advantages = torch.tensor(advantage, dtype=torch.float32)
    raw_rewards_tensor = torch.tensor(raw_rewards, dtype=torch.float32)

    metadata = {
        "mean_reward": raw_rewards_tensor.mean().item(),
        "max_reward": raw_rewards_tensor.max().item(),
        "min_reward": raw_rewards_tensor.min().item(),
        "std_reward": raw_rewards_tensor.std().item()
    }

    return advantages, raw_rewards_tensor, metadata

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute GRPO loss with PPO-style clipping for training stability.
    
    Args:
        advantages: torch.Tensor Shape (batch_size, 1), per-example advantages A.
        
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log
        probs from the policy being trained.
        
        old_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log probs
        from the old policy.
        
        cliprange: float Clip parameter ϵ (e.g. 0.2).
    
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
        
        loss: torch.Tensor of shape (batch_size, sequence_length), the per-token clipped
        loss.
        
        metadata: dict containing whatever you want to log. We suggest logging whether each
        token was clipped or not, i.e., whether the clipped policy gradient loss on the RHS of
        the min was lower than the LHS.
    """

    log_multiplier = torch.exp(policy_log_probs - old_log_probs)

    # broadcast advantages to match policy_log_probs shape
    advantages = repeat(advantages, 'b 1 -> b s', s=policy_log_probs.shape[1])

    unclipped_loss = log_multiplier * advantages

    clipped_log_multiplier = torch.clamp(log_multiplier, 1 - cliprange, 1 + cliprange)
    clipped_loss = clipped_log_multiplier * advantages

    loss = -torch.min(unclipped_loss, clipped_loss)

    metadata = {}

    return loss, metadata


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None
) -> torch.Tensor:
    """
    Compute mean of tensor elements where mask is True.
    
    Args:
        tensor: torch.Tensor The data to be averaged.
       
        mask: torch.Tensor Same shape as tensor; positions with 1 are included in the mean.
       
        dim: int | None Dimension over which to average. If None, compute the mean over all
        masked elements.
    
    Returns:
        torch.Tensor The masked mean; shape matches tensor.mean(dim) semantics.
    """
    
    masked_tensor = tensor * mask
    sum_masked = masked_tensor.sum(dim=dim)
    count_masked = mask.sum(dim=dim)
    mean_masked = sum_masked / count_masked
    return mean_masked


def grpo_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        loss_type: Literal["grpo_clip"], # for this assignment, only "grpo_clip" is required
        raw_rewards: torch.Tensor | None=None,
        advantages: torch.Tensor | None=None,
        old_log_probs: torch.Tensor | None=None,
        cliprange: float | None=None
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute one GRPO training microbatch step with gradient accumulation.
    
    Args:
        policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
        policy being trained.
        
        response_mask (batch_size, sequence_length), 1 for response tokens, 0 for
        prompt/padding.
        
        gradient_accumulation_steps Number of microbatches per optimizer step.
        
        loss_type "grpo_clip".
        
        raw_rewards Needed when loss_type == "no_baseline"; shape (batch_size, 1).
        
        advantages Needed when loss_type != "no_baseline"; shape (batch_size, 1).
        
        old_log_probs Required for GRPO-Clip; shape (batch_size, sequence_length).
        
        cliprange Clip parameter ϵ for GRPO-Clip.
    
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
        
        loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
        this so we can log it.
        
        metadata Dict with metadata from the underlying loss call, and any other statistics you
        might want to log.
    """
    
    loss, _ = compute_grpo_clip_loss(
        advantages,
        policy_log_probs,
        old_log_probs,
        cliprange,
    )

    masked_mean_loss = masked_mean(loss, response_mask) / gradient_accumulation_steps
    masked_mean_loss.backward()

    metadata = {
        "microbatch_loss": masked_mean_loss.item()
    }

    return masked_mean_loss, metadata
