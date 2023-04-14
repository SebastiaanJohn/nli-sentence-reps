"""Baseline sentence encoder: averaging word embeddings to obtain sentence representations."""

import torch
from torch import nn


class BaselineEncoder(nn.Module):
    """Baseline sentence encoder: averaging word embeddings to obtain sentence representations."""
    def __init__(self) -> None:
        """Initialize the encoder."""
        super(BaselineEncoder, self).__init__()

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute sentence representations by averaging word embeddings.

        Args:
            embeddings (torch.Tensor): The word embeddings of shape (batch_size, seq_len, embedding_dim).
        """
        return torch.mean(embeddings, dim=1)


