"""Unidirectional LSTM encoder."""

from torch import nn


class UniLSTMEncoder(nn.Module):
    """Unidirectional LSTM applied on the word embeddings.

    The last hidden state is considered as sentence representation.
    """
    pass
