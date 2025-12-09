import torch 
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """Construct a linear transformation module.
        This function should accept the following parameters:
        
        in_features: int
            Final dimension of the input
        out_features: int
            Final dimension of the output
        device: torch.device | None = None
            Device to store the parameters on
        dtype: torch.dtype | None = None
            Data type of the parameters
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize the weight parameter."""
        std = (2.0 / (self.in_features + self.out_features)) ** 0.5
        upper = std * 3
        lower = -upper

        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=lower, b=upper)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation to the input."""
        # i: in_features
        # o: out_features
        return torch.einsum("...i,oi->...o", x, self.weight)
    
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """Construct an embedding module.
        This function should accept the following parameters:
        
        num_embeddings: int
            Number of unique embeddings (vocabulary size)
        embedding_dim: int
            Dimension of each embedding vector
        device: torch.device | None = None
            Device to store the parameters on
        dtype: torch.dtype | None = None
            Data type of the parameters
        """
        super().__init__()
        
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize the embedding weights."""
        std = 1.0
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3, b=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the embedding lookup to the input."""
        return self.weight[x]