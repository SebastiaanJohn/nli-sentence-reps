"""Utility functions for the data module."""

import torch
from torch.nn.utils.rnn import pad_sequence


def snli_collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function for the SNLI dataset.

    Args:
        batch (list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]): List of tuples
            containing the premise, hypothesis, and label of the premise-hypothesis pair.
    """
    premises, hypotheses, labels = zip(*batch)

    # Pad sequences in premises and hypotheses using pad_sequence
    padded_premises = pad_sequence(premises, batch_first=True, padding_value=0)
    padded_hypotheses = pad_sequence(hypotheses, batch_first=True, padding_value=0)

    # Stack labels into a single tensor
    labels = torch.stack(labels)

    return padded_premises, padded_hypotheses, labels
