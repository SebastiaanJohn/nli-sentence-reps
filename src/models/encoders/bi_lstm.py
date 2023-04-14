"""Simple bidirectional LSTM (BiLSTM).

where the last hidden state of forward and backward layers are concatenated as the sentence representations.
"""

import torch
from torch import nn


class BiLSTMEncoder(nn.Module):
    """BiLSTM encoder."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, max_pooling: bool = False) -> None:
        """Initialize the BiLSTM encoder.

        Args:
            input_size (int): The size of the input embeddings.
            hidden_size (int): The size of the hidden states.
            num_layers (int): The number of layers in the LSTM. Defaults to 1.
            max_pooling (bool): Whether to use max-pooling to aggregate the word-level hidden states.
                Defaults to False.
        """
        super(BiLSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.max_pooling = max_pooling

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass of the BiLSTM encoder.

        Args:
            embeddings (torch.Tensor): The input word embeddings (batch_size, seq_len, input_size).
        """
        # Pass the embeddings through the LSTM
        lstm_outputs, (hidden_states, _) = self.lstm(embeddings)

        if self.max_pooling:
            # Apply max-pooling to the concatenation of word-level hidden states from both directions
            sentence_repr, _ = torch.max(lstm_outputs, dim=1)
        else:
            # Concatenate the last hidden states of the forward and backward layers
            forward_hidden = hidden_states[0]
            backward_hidden = hidden_states[1]
            sentence_repr = torch.cat((forward_hidden, backward_hidden), dim=1)

        return sentence_repr
