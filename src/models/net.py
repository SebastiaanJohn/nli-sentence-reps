"""This module contains the NLIModel class."""

import torch
from torch import nn


class NLIModel(nn.Module):
    """The NLIModel class is a wrapper around the encoder and classifier."""
    def __init__(self, encoder: nn.Module, classifier: nn.Module) -> None:
        """Initialize the NLIModel class.

        Args:
            encoder (nn.Module): The sentence encoder.
            classifier (nn.Module): The classifier.
        """
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, premise: torch.Tensor, hypothesis: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model.

        Args:
            premise (torch.Tensor): The premise sentences.
            hypothesis (torch.Tensor): The hypothesis sentences.

        Returns:
            torch.Tensor: The logits.
        """
        premise_rep = self.encoder(premise)
        hypothesis_rep = self.encoder(hypothesis)
        logits = self.classifier(premise_rep, hypothesis_rep)

        return logits
