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

    def forward(
        self,
        premise: torch.Tensor,
        premise_lengths: torch.Tensor,
        hypothesis: torch.Tensor,
        hypothesis_lengths: torch.Tensor
        ) -> torch.Tensor:
        """Perform a forward pass through the model.

        Args:
            premise (torch.Tensor): The premise sentences.
            premise_lengths (torch.Tensor): The lengths of the premise sentences.
            hypothesis (torch.Tensor): The hypothesis sentences.
            hypothesis_lengths (torch.Tensor): The lengths of the hypothesis sentences.

        Returns:
            torch.Tensor: The logits.
        """
        premise_rep = self.encoder(premise, premise_lengths)
        hypothesis_rep = self.encoder(hypothesis, hypothesis_lengths)
        logits = self.classifier(premise_rep, hypothesis_rep)

        return logits
