"""Simple bidirectional LSTM (BiLSTM).

where the last hidden state of forward and backward layers are concatenated as the sentence representations.
"""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMEncoder(nn.Module):
    """BiLSTM encoder."""
    def __init__(
        self,
        word_embeddings: torch.Tensor,
        input_dim: int = 300,
        output_dim: int = 2048,
        num_layers: int = 1,
        max_pooling: bool = False
        ) -> None:
        """Initialize the BiLSTM encoder.

        Args:
            word_embeddings (torch.Tensor): The word embeddings.
            input_dim (int): The dimension of the input word embeddings. Defaults to 300.
            output_dim (int): The dimension of the output sentence representations. Defaults to 2048.
            num_layers (int): The number of layers in the LSTM. Defaults to 1.
            max_pooling (bool): Whether to use max-pooling to aggregate the word-level hidden states.
                Defaults to False.
        """
        super(BiLSTMEncoder, self).__init__()
        self.word_embeddings = nn.Embedding.from_pretrained(word_embeddings)
        self.lstm = nn.LSTM(input_dim, output_dim, num_layers, bidirectional=True, batch_first=True)
        self.max_pooling = max_pooling

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass of the BiLSTM encoder.

        Args:
            indices (torch.Tensor): The input indices (batch_size, seq_len).
            lengths (torch.Tensor): Lengths of input sequences (batch_size,).
        """
        # Get the word embeddings (batch_size, seq_len, embedding_dim)
        embeddings = self.word_embeddings(indices)

        # Sort the sequences by length in descending order (required for pack_padded_sequence)
        lengths, sorted_indices = lengths.sort(descending=True)

        # Move sorted_indices to the same device as embeddings
        sorted_indices = sorted_indices.to(embeddings.device)

        sorted_embeddings = embeddings.index_select(0, sorted_indices)

        # Move lengths to CPU
        lengths = lengths.to('cpu')

        # Pack the padded sequence
        packed_embeddings = pack_padded_sequence(sorted_embeddings, lengths, batch_first=True)

        # Pass the packed embeddings through the LSTM
        packed_lstm_outputs, (hidden_states, _) = self.lstm(packed_embeddings)

        # Unpack the packed outputs
        lstm_outputs, _ = pad_packed_sequence(packed_lstm_outputs, batch_first=True)

        if self.max_pooling:
            # Apply max-pooling to the concatenation of word-level hidden states from both directions
            sentence_repr, _ = torch.max(lstm_outputs, dim=1)
        else:
            # Concatenate the last hidden states of the forward and backward layers
            forward_hidden = hidden_states[-2]
            backward_hidden = hidden_states[-1]
            sentence_repr = torch.cat((forward_hidden, backward_hidden), dim=1)

        # Undo the sorting by length
        _, unsorted_indices = sorted_indices.sort()
        unsorted_representation = sentence_repr.index_select(0, unsorted_indices.to(sentence_repr.device))

        return unsorted_representation

