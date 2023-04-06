"""Baseline sentence encoder: averaging word embeddings to obtain sentence representations."""

import torch


class BaselineEncoder(torch.nn.Module):
    """Baseline sentence encoder: averaging word embeddings to obtain sentence representations."""

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """Compute sentence embeddings for a batch of sentences.

        Args:
            sentence: Tensor of shape (batch_size, max_sentence_length, embedding_size) containing
                word embeddings.

        Returns:
            Tensor of shape (batch_size, embedding_size) containing sentence embeddings.
        """
        return sentence.mean(dim=1)

