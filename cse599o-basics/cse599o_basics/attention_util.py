import torch
import torch.nn as nn
import math

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute the softmax of the input tensor along the specified dimension.

    Args:
        x: Input tensor.
        dim: Dimension along which to compute the softmax.

    Returns:
        Tensor with the same shape as input, after applying softmax.
    """
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    
    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    assert x_exp_sum.shape == x_max.shape, "Shapes of exp sum and max do not match."
    
    return x_exp / x_exp_sum


@torch.cuda.nvtx.range("scaled dot product attention")
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the scaled dot-product attention.

    Args:
        query: Tensor of shape (batchsize, ..., seq_len, d_k)
        key: Tensor of shape (batchsize, ..., seq_len, d_k)
        value: Tensor of shape (batchsize, ..., seq_len, d_v)
        mask: Optional tensor broadcastable to (batchsize, ..., seq_len_q, seq_len_k). Contains 1s for positions to attend to and 0s elsewhere.

    Returns:
        Tensor of shape (batchsize, ..., d_v) after applying attention.
    """
    
    torch.cuda.nvtx.range_push("compute attention scores")
    normalize = query.shape[-1] ** 0.5
    qTk = torch.einsum("...qd,...kd->...qk", query, key) / normalize
    torch.cuda.nvtx.range_pop()
    
    if mask is not None:
        qTk = qTk.masked_fill(mask == 0, float("-inf"))

    torch.cuda.nvtx.range_push("apply softmax")
    attn_weights = softmax(qTk, dim=-1)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("final matmul")
    output = torch.einsum("...qk,...kd->...qd", attn_weights, value)
    torch.cuda.nvtx.range_pop()
    return output
