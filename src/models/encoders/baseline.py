"""Baseline sentence encoder: averaging word embeddings to obtain sentence representations."""

import torch
from torch import nn


class BaselineEncoder(nn.Module):
    """Baseline sentence encoder: averaging word embeddings to obtain sentence representations."""
    def __init__(self, word_embeddings: torch.Tensor) -> None:
        """Initialize the encoder.

        Args:
            word_embeddings (torch.Tensor): The word embeddings.
        """
        super(BaselineEncoder, self).__init__()
        self.word_embeddings = nn.Embedding.from_pretrained(word_embeddings)

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Compute sentence representations by averaging word embeddings.

        Args:
            indices (torch.Tensor): The input indices (batch_size, seq_len).
            lengths (torch.Tensor): Lengths of input sequences (batch_size,).
        """
        # Get the word embeddings (batch_size, seq_len, embedding_dim)
        embeddings = self.word_embeddings(indices)

        # Sum the embeddings along the sequence dimension (dim=1)
        sum_embeddings = torch.sum(embeddings, dim=1)

        # Compute the average by dividing the sum by the sequence lengths
        lengths = lengths.view(-1, 1).to(torch.float32)  # Ensure lengths have the same dtype as sum_embeddings
        avg_embeddings = sum_embeddings / lengths

        return avg_embeddings



