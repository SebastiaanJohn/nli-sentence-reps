"""Utility functions for the models."""

from torch import nn

from src.models.encoders.baseline import BaselineEncoder
from src.models.encoders.bilstm import BiLSTMEncoder
from src.models.encoders.lstm import LSTMEncoder


def get_encoder(embeddings, args) -> nn.Module:
    """Get the sentence encoder."""
    if args.encoder == "baseline":
        encoder = BaselineEncoder(embeddings)
    elif args.encoder == "lstm":
        encoder = LSTMEncoder(embeddings, args.embeddings_dim, args.hidden_size)
    elif args.encoder == "bilstm":
        encoder = BiLSTMEncoder(embeddings, args.embeddings_dim, args.hidden_size)
    elif args.encoder == "bilstm-max":
        encoder = BiLSTMEncoder(embeddings, args.embeddings_dim, args.hidden_size, max_pooling=True)
    else:
        error = f"Unknown encoder: {args.encoder}"
        raise ValueError(error)

    return encoder
