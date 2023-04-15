"""Utility functions for the data module."""

import torch
from torch.nn.utils.rnn import pad_sequence


def snli_collate_fn(batch):
    """Collate function for the SNLI dataset.

    Args:
        batch (list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]): A batch
            of data from the SNLI dataset.
    """
    premises, hypotheses, labels = zip(*batch)

    # Compute lengths
    premise_lengths = torch.tensor([len(premise) for premise in premises])
    hypothesis_lengths = torch.tensor([len(hypothesis) for hypothesis in hypotheses])

    # Pad sequences in premises and hypotheses using pad_sequence
    padded_premises = pad_sequence(premises, batch_first=True, padding_value=1)
    padded_hypotheses = pad_sequence(hypotheses, batch_first=True, padding_value=1)

    # Stack labels into a single tensor
    labels = torch.stack(labels)

    return padded_premises, padded_hypotheses, premise_lengths, hypothesis_lengths, labels
