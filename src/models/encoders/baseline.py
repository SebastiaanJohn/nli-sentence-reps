"""Baseline sentence encoder: averaging word embeddings to obtain sentence representations."""

import torch
from torch import nn


class BaselineEncoder(nn.Module):
    """Baseline sentence encoder: averaging word embeddings to obtain sentence representations."""
    def __init__(self, aligned_embeddings: torch.Tensor) -> None:
        """Initialize the encoder.

        Args:
            aligned_embeddings (torch.Tensor): Aligned word embeddings (vocab_size, embedding_dim).
        """
        super(BaselineEncoder, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(aligned_embeddings, freeze=True)

    def forward(self, token_indices: torch.Tensor) -> torch.Tensor:
        """Compute sentence representations by averaging word embeddings.

        Args:
            token_indices (torch.Tensor): Tensor of token indices of shape (batch_size, sequence_length).
        """
        # Convert token indices to embeddings
        embeddings = self.embedding(token_indices)

        # Compute the mask to ignore padding tokens in the averaging process
        mask = token_indices.ne(0).float().unsqueeze(-1)

        # Compute the sum of embeddings along the sequence dimension
        sum_embeddings = (embeddings * mask).sum(dim=1)

        # Compute the number of non-padding tokens in each sequence
        num_tokens = mask.sum(dim=1)

        # Average the embeddings
        averaged_embeddings = sum_embeddings / num_tokens

        return averaged_embeddings


