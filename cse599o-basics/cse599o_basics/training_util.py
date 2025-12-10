import torch
import math

def cross_entropy_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute the cross-entropy loss between logits and target.
    """
    # make logits numerically stable
    vocab_dim = -1
    logits_max = torch.max(logits, dim=vocab_dim, keepdim=True).values
    logits_stable = logits - logits_max

    # get denominator and numerator in log space
    log_denom = torch.log(torch.sum(torch.exp(logits_stable), dim=vocab_dim))
    
    targets_rearrange = target.unsqueeze(vocab_dim)
    log_numerator = torch.gather(logits_stable, vocab_dim, targets_rearrange).squeeze(vocab_dim)

    log_probs = log_numerator - log_denom

    return torch.mean(-log_probs)

def perplexity(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute the perplexity between logits and target.
    """
    ce_loss = cross_entropy_loss(logits, target)
    return torch.exp(ce_loss)


def cosine_lr_schedule(
    step: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    cos_anneal_steps: int,
) -> float:
    """Compute the learning rate at a given step using a cosine schedule with warmup.
    Args:
        step: Current training step (0-indexed).
        max_lr: Maximum learning rate.
        min_lr: Minimum learning rate.
        warmup_steps: Number of steps to linearly increase the learning rate.
        cos_steps: Number of steps to decay the learning rate using cosine decay.
    Returns:
        Learning rate at the current step.
    """
    assert warmup_steps < cos_anneal_steps, "warmup_steps must be less than cos_anneal_steps"
    
    cos_steps = cos_anneal_steps - warmup_steps
    if step < warmup_steps:
        lr = max_lr * (step / warmup_steps)
    elif step < cos_anneal_steps:
        cos_decay = 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / cos_steps))
        lr = min_lr + (max_lr - min_lr) * cos_decay
    else:
        lr = min_lr
    return lr


def gradient_clip(gradients: torch.Tensor, max_norm: float, eps: float = 1e-6) -> None:
    """Clip gradients of an iterable of parameters to a maximum L2 norm of 1.0.
    """
    total_norm = torch.norm(torch.stack([torch.norm(g) for g in gradients]))
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + eps)
        for g in gradients:
            g.data.mul_(clip_coef)


import numpy as np
from typing import Tuple

def data_load(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: torch.device,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    sampled_input = torch.zeros((batch_size, context_length), dtype=torch.long, device=device)
    sampled_target = torch.zeros((batch_size, context_length), dtype=torch.long, device=device)
    n = len(x) - context_length - 1

    idx = np.random.randint(0, n + 1, size=(batch_size,))
    input_indices = idx[:, None] + np.arange(context_length)
    target_indices = input_indices + 1
    
    sampled_input = torch.tensor(x[input_indices], dtype=torch.long, device=device)
    sampled_target = torch.tensor(x[target_indices], dtype=torch.long, device=device)

    return sampled_input, sampled_target


import os
import typing
def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]) -> None:
    """Save model and optimizer state to a checkpoint file.
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )

from typing import Optional
def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer]) -> int:
    """Load model and optimizer state from a checkpoint file.
    Returns the iteration number to resume training from.
    """
    checkpoint = torch.load(src)

    prefix = "_orig_mod."
    original_state_dict = checkpoint["model_state_dict"]
    
    cleaned_state_dict = {
        key[len(prefix):] if key.startswith(prefix) else key: value
        for key, value in original_state_dict.items()
    }
    
    # Load the cleaned state_dict
    model.load_state_dict(cleaned_state_dict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    iteration = checkpoint["iteration"]
    return iteration