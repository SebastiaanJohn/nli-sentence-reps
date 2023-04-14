"""Unidirectional LSTM encoder."""

import torch
from torch import nn


class LSTMEncoder(nn.Module):
    """Unidirectional LSTM encoder."""
    def __init__(self, embedding_dim: int, hidden_size: int) -> None:
        """Initialize the LSTM encoder.

        Args:
            embedding_dim (int): Dimension of the word embeddings.
            hidden_size (int): Size of the hidden state of the LSTM.
        """
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute sentence representations using the LSTM.

        Args:
            embeddings (torch.Tensor): Tensor of word embeddings of
                shape (batch_size, sequence_length, embedding_dim).
        """
        # Pass the embeddings through the LSTM
        _, (hidden_state, _) = self.lstm(embeddings)

        # The last hidden state is the sentence representation
        sentence_representation = hidden_state[-1]

        return sentence_representation

