"""Utility functions for the models."""

from torch import nn

from models.encoders.baseline import BaselineEncoder
from models.encoders.bilstm import BiLSTMEncoder
from models.encoders.lstm import LSTMEncoder


def get_encoder(embeddings, encoder_name: str) -> nn.Module:
    """Get the sentence encoder.

    Args:
        embeddings (torch.Tensor): The word embeddings.
        encoder_name (str): The name of the encoder.

    Returns:
        (nn.Module): The encoder.
    """
    if encoder_name == "baseline":
        encoder = BaselineEncoder(embeddings)
    elif encoder_name == "lstm":
        encoder = LSTMEncoder(embeddings, 300, 2048)
    elif encoder_name == "bilstm":
        encoder = BiLSTMEncoder(embeddings, 300, 2048)
    elif encoder_name == "bilstm-max":
        encoder = BiLSTMEncoder(embeddings, 300, 2048, max_pooling=True)
    else:
        error = f"Unknown encoder: {encoder_name}"
        raise ValueError(error)

    return encoder
