"""Unidirectional LSTM encoder."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class LSTMEncoder(nn.Module):
    """Unidirectional LSTM encoder."""
    def __init__(self, word_embeddings: torch.Tensor, input_dim: int = 300, output_dim: int = 2048):
        """Initialize the LSTM encoder.

        Args:
            word_embeddings (torch.Tensor): The word embeddings (vocab_size, embedding_dim)
            input_dim (int): Dimension of the input word embeddings.
                Defaults to 300.
            output_dim (int): Dimension of the output sentence representations.
                Defaults to 2048.
        """
        super(LSTMEncoder, self).__init__()
        self.word_embeddings = nn.Embedding.from_pretrained(word_embeddings)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=output_dim, batch_first=True)

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Compute sentence representations using the LSTM.

        Args:
            indices (torch.Tensor): The input indices (batch_size, seq_len).
            lengths (torch.Tensor): Lengths of input sequences (batch_size,).
        """
        # Get the word embeddings (batch_size, seq_len, embedding_dim)
        embeddings = self.word_embeddings(indices)

        # # Sort the sequences by length in descending order (required for pack_padded_sequence)
        # lengths, sorted_indices = lengths.sort(descending=True)

        # Move sorted_indices to the same device as embeddings
        # sorted_indices = sorted_indices.to(embeddings.device)

        # sorted_embeddings = embeddings.index_select(0, sorted_indices)

        # Move lengths to CPU
        lengths = lengths.to('cpu')

        # Pack the padded sequence
        packed_embeddings = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)

        # Pass the packed embeddings through the LSTM
        _, (hidden_state, _) = self.lstm(packed_embeddings)

        # The last hidden state is the sentence representation
        sentence_representation = hidden_state[-1]

        # # Undo the sorting by length
        # _, unsorted_indices = sorted_indices.sort()
        # unsorted_representation = sentence_representation.index_select(0, unsorted_indices.to(sentence_representation.device))

        return sentence_representation


