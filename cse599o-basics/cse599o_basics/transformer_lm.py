import torch
import torch.nn as nn

from cse599o_basics.transformer_block import PreNormTransformer, RMSNorm, TimedPreNormTransformer
from cse599o_basics.basic_layer import Embedding, Linear
from cse599o_basics.attention_util import softmax


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_heads: int,
        dff: int,
        num_layers: int,
        theta: float = None,
        device=None,
        dtype=None,
    ):
        """
        Construct a Transformer-based language model.

        Args:
            vocab_size: Size of the vocabulary
            context_length: Maximum context length (sequence length)
            d_model: Dimension of the model
            num_heads: Number of attention heads
            dff: Dimension of the feed-forward network
            num_layers: Number of Transformer blocks
            theta: RoPE parameter (optional)
            device: Device to store the parameters on
            dtype: Data type of the parameters
        """

        super().__init__()
        self.token_embedding = Embedding(
            vocab_size, d_model, device=device, dtype=dtype
        )
        self.layers = nn.ModuleList(
            [
                PreNormTransformer(
                    d_model,
                    num_heads,
                    dff,
                    theta,
                    context_length,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.rmsnorm = RMSNorm(d_model, device=device, dtype=dtype)
        self.output_linear = Linear(d_model, vocab_size, device=device, dtype=dtype)


    import torch.cuda.nvtx as nvtx
    
    @nvtx.range("TransformerLM forward", color="blue")
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer language model.

        Args:
            token_ids: Tensor of shape (batch_size, seq_len) containing token indices.

        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size).
        """
        
        
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        # Get token positions
        token_positions = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        )
        x = self.token_embedding(token_ids)

        # Pre-norm transformer blocks
        for layer in self.layers:
            x = layer(x, token_positions)

        # Final RMSNorm and linear layer
        x = self.rmsnorm(x)
        logits = self.output_linear(x)

        return logits


class TimedTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_heads: int,
        dff: int,
        num_layers: int,
        theta: float = None,
        device=None,
        dtype=None,
    ):
        """
        Construct a Transformer-based language model.

        Args:
            vocab_size: Size of the vocabulary
            context_length: Maximum context length (sequence length)
            d_model: Dimension of the model
            num_heads: Number of attention heads
            dff: Dimension of the feed-forward network
            num_layers: Number of Transformer blocks
            theta: RoPE parameter (optional)
            device: Device to store the parameters on
            dtype: Data type of the parameters
        """

        super().__init__()
        self.token_embedding = Embedding(
            vocab_size, d_model, device=device, dtype=dtype
        )
        self.layers = nn.ModuleList(
            [
                TimedPreNormTransformer(
                    d_model,
                    num_heads,
                    dff,
                    theta,
                    context_length,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.rmsnorm = RMSNorm(d_model, device=device, dtype=dtype)
        self.output_linear = Linear(d_model, vocab_size, device=device, dtype=dtype)
        
        self.ffn_time_ms = []
        self.attn_time_ms = []

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer language model.

        Args:
            token_ids: Tensor of shape (batch_size, seq_len) containing token indices.

        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size).
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        # Get token positions
        token_positions = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        )
        x = self.token_embedding(token_ids)

        # Pre-norm transformer blocks
        for layer in self.layers:
            x = layer(x, token_positions)
            torch.cuda.synchronize()
            self.ffn_time_ms.append(layer.ffn_start.elapsed_time(layer.ffn_end))
            self.attn_time_ms.append(layer.attn_start.elapsed_time(layer.attn_end))

        # Final RMSNorm and linear layer
        x = self.rmsnorm(x)
        logits = self.output_linear(x)

        return logits
