import torch
import torch.nn as nn
from cse599o_basics.basic_layer import Linear
from cse599o_basics.attention_util import scaled_dot_product_attention


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        # Construct a RMSNorm module.
        # This function should accept the following parameters:

        # d_model: int                         Hidden dimension of the model
        # eps: float = 1e-5                    Epsilon value for numerical stability
        # device: torch.device | None          Device to store the parameters on
        # dtype: torch.dtype | None            Data type of the parameters

        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process an input tensor of shape (batch_size, sequence_length, d_model)
        and return a tensor of the same shape."""
        in_type = x.dtype
        x = x.to(torch.float32)

        assert (
            self.d_model == x.shape[-1]
        ), f"Expected input with last dimension {self.d_model}, but got {x.shape[-1]}"
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms * self.weight

        return x_norm.to(in_type)


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the SiLU activation function.

        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            Tensor of shape (..., d_model) after applying SiLU activation
        """
        return x * torch.sigmoid(x)


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, dff: int = None, device=None, dtype=None):
        super().__init__()

        self.dff = dff if dff is not None else int(8 * d_model / 3)
        self.W_1 = Linear(d_model, self.dff, device, dtype)
        self.W_2 = Linear(self.dff, d_model, device, dtype)
        self.W_3 = Linear(d_model, self.dff, device, dtype)
        self.silu = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the SwiGLU activation function.

        Args:
            x: Input tensor of shape (..., 2 * d_model)

        Returns:
            Tensor of shape (..., d_model) after applying SwiGLU activation
        """
        out = self.W_1(x)
        out = self.silu(out)
        out = out * self.W_3(x)
        out = self.W_2(out)
        return out


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Construct the RoPE module and create buffers if needed.

        Args:
            theta: Theta value for RoPE.
            d_k: Dimension of query/key vectors (should be even).
            max_seq_len: Maximum sequence length that will be inputted.
            device: torch.device | None. Device to store the buffers on.
        """
        super().__init__()

        # create 2d buffer for cos and sin in cache
        theta_denominator = 1.0 / (
            theta ** (torch.arange(0, d_k, 2, device=device) / d_k)
        )
        theta_numerator = torch.arange(max_seq_len, device=device)
        theta = torch.outer(theta_numerator, theta_denominator)

        cos_cache = torch.cos(theta)
        sin_cache = torch.sin(theta)

        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to an input tensor of shape (..., seq_len, d_k) and
        return a tensor of the same shape.

        Notes:
            - Accept x with an arbitrary number of batch dimensions.
            - token_positions has shape (..., seq_len) and gives absolute
                positions per token along the sequence dimension.
            - Use token_positions to slice (precomputed) cos/sin tensors
            along the sequence dimension.
        """
        cos = self.cos_cache[token_positions].unsqueeze(-3)
        sin = self.sin_cache[token_positions].unsqueeze(-3)

        x1, x2 = x[..., 0::2], x[..., 1::2]

        # Use formula (x1, x2) -> (x1*cos - x2*sin, x1*sin + x2*cos)
        x_rotated__1 = x1 * cos - x2 * sin
        x_rotated__2 = x1 * sin + x2 * cos

        x_rotated = torch.empty_like(x)
        x_rotated[..., 0::2] = x_rotated__1
        x_rotated[..., 1::2] = x_rotated__2

        return x_rotated


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float = None,
        max_seq_len: int = None,
        device=None,
        dtype=None,
    ):
        """
        Construct a multi-head self-attention module.

        Args:
            d_model: Dimension of the model
            num_heads: Number of attention heads
            theta: Optional float. If provided, enables RoPE with the given theta value.
            max_seq_len: Optional int. Required if theta is provided. Maximum sequence length for Ro
            device: torch.device | None. Device to store the parameters on.
            dtype: torch.dtype | None. Data type of the parameters.
        """
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_v = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_o = Linear(d_model, d_model, device=device, dtype=dtype)

        self.rope = None
        if theta is not None and max_seq_len is not None:
            self.rope = RoPE(theta, self.d_k, max_seq_len, device=device)

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Apply multi-head self-attention to an input tensor of shape
        (batch_size, seq_len, d_model) and return a tensor of the same shape.

        Args:
            x: Input tensor of shape (batch_size, ..., seq_len, d_model)
            token_positions: Optional tensor of shape (batch_size, seq_len)

        Returns:
            Tensor of shape (batch_size, seq_len, d_model) after applying multi-head self-attention.
        """
        batch_size = x.shape[0]
        seq_len = x.shape[-2]

        # Compute Q, K, V matrices
        # Before self-attention, reshape Q, K, V to (batch_size, num_heads, seq_len, d_k)
        Q = (
            self.W_q(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.W_k(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.W_v(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

        # Apply RoPE to Q and K
        if self.rope is not None and token_positions is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # Scaled dot-product attention
        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device))
        attention_out = scaled_dot_product_attention(Q, K, V, mask=mask)

        attention_out = (
            attention_out.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )
        output = self.W_o(attention_out)
        return output


class PreNormTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dff: int,
        theta: float = None,
        max_seq_len: int = None,
        device=None,
        dtype=None,
    ):
        """
        Construct a Transformer block with pre-norm architecture.

        Args:
            d_model: Dimension of the model
            num_heads: Number of attention heads
            theta: Optional float. If provided, enables RoPE with the given theta value.
            max_seq_len: Optional int. Required if theta is provided. Maximum sequence length for RoPE
            device: torch.device | None. Device to store the parameters on.
            dtype: torch.dtype | None. Data type of the parameters.
        """
        super().__init__()

        self.attn = MultiHeadSelfAttention(
            d_model, num_heads, theta, max_seq_len, device=device, dtype=dtype
        )
        self.ffn = SwiGLUFFN(d_model, dff, device=device, dtype=dtype)
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Apply the Transformer block to an input tensor of shape
        (batch_size, seq_len, d_model) and return a tensor of the same shape.

        Args:
            x: Input tensor of shape (batch_size, ..., seq_len, d_model)
            token_positions: Optional tensor of shape (batch_size, seq_len)

        Returns:
            Tensor of shape (batch_size, seq_len, d_model) after applying the Transformer block.
        """
        x = x + self.attn(self.norm1(x), token_positions=token_positions)
        x = x + self.ffn(self.norm2(x))
        return x
    
    
class TimedPreNormTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dff: int,
        theta: float = None,
        max_seq_len: int = None,
        device=None,
        dtype=None,
    ):
        """
        Construct a Transformer block with pre-norm architecture.

        Args:
            d_model: Dimension of the model
            num_heads: Number of attention heads
            theta: Optional float. If provided, enables RoPE with the given theta value.
            max_seq_len: Optional int. Required if theta is provided. Maximum sequence length for RoPE
            device: torch.device | None. Device to store the parameters on.
            dtype: torch.dtype | None. Data type of the parameters.
        """
        super().__init__()

        self.attn = MultiHeadSelfAttention(
            d_model, num_heads, theta, max_seq_len, device=device, dtype=dtype
        )
        self.ffn = SwiGLUFFN(d_model, dff, device=device, dtype=dtype)
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        
        self.attn_start = torch.cuda.Event(enable_timing=True)
        self.attn_end = torch.cuda.Event(enable_timing=True)
        self.ffn_start = torch.cuda.Event(enable_timing=True)
        self.ffn_end = torch.cuda.Event(enable_timing=True)

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Apply the Transformer block to an input tensor of shape
        (batch_size, seq_len, d_model) and return a tensor of the same shape.

        Args:
            x: Input tensor of shape (batch_size, ..., seq_len, d_model)
            token_positions: Optional tensor of shape (batch_size, seq_len)

        Returns:
            Tensor of shape (batch_size, seq_len, d_model) after applying the Transformer block.
        """
        
        x_copy = x
        x_norm = self.norm1(x_copy)
        self.attn_start.record()
        attn = self.attn(x_norm, token_positions)
        self.attn_end.record()
        x = x + attn
        
        x_copy = x
        x_norm = self.norm2(x_copy)
        self.ffn_start.record()
        ffn = self.ffn(x_norm)
        self.ffn_end.record()
        x = x + ffn
        return x
