"""This module contains the code for building the dataset."""

from pathlib import Path

import torch
from datasets import DatasetDict
from spacy.lang.en import English
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import GloVe


nlp = English()

def tokenize(batch: dict) -> dict:
    """Tokenize the premise and hypothesis of a batch of examples.

    Args:
        batch (dict): A batch of examples from the dataset

    Returns:
        dict: A dictionary containing the tokenized premise and hypothesis
    """
    return {
        "premise": [[token.text.lower() for token in nlp.tokenizer(text)] for text in batch["premise"]],
        "hypothesis": [[token.text.lower() for token in nlp.tokenizer(text)] for text in batch["hypothesis"]],
    }

def tokens_to_indices(batch: dict, token_to_idx: dict[str, int]) -> dict:
    """Convert tokens to indices.

    Args:
        batch (dict): A batch of examples from the dataset
        token_to_idx (dict): A dictionary mapping tokens to indices

    Returns:
        dict: A dictionary containing the tokenized premise and hypothesis
    """
    return {
        "premise": [[token_to_idx[token] for token in example] for example in batch["premise"]],
        "hypothesis": [[token_to_idx[token] for token in example] for example in batch["hypothesis"]],
    }

def build_vocabulary(
    dataset: DatasetDict,
    glove_version: str = "840B",
    word_embedding_dim: int = 300,
    ) -> tuple[dict[str, int], torch.Tensor]:
    """Build the vocabulary from the dataset.

    Args:
        dataset (DatasetDict): The dataset to build the vocabulary from
        glove_version (str): The version of GloVe to use
        word_embedding_dim (int): The dimension of the GloVe embeddings

    Returns:
        token_to_idx (dict): A dictionary mapping tokens to indices
        aligned_embeddings (torch.Tensor): A tensor of aligned embeddings
    """
    # Check if the vocabulary has already been built
    if Path("token_to_idx.pt").exists() and Path("word_embeddings.pt").exists():
        print("Loading vocabulary from disk")
        token_to_idx = torch.load("token_to_idx.pt")
        aligned_embeddings = torch.load("word_embeddings.pt")
        return token_to_idx, aligned_embeddings

    # Load the GloVe embeddings
    glove = GloVe(name=glove_version, dim=word_embedding_dim)

    # Create a dictionary mapping tokens to indices
    token_to_idx = {"<UNK>": 0, "<PAD>": 1}

    # Compute the average embedding for the unknown token
    unk_embedding = glove.vectors.mean(dim=0)

    # Create a list of aligned embeddings
    aligned_embeddings = [unk_embedding, glove["<PAD>"]]

    # Get unique tokens from the dataset
    unique_tokens = {token for item in dataset for token in item["premise"] + item["hypothesis"]}

    # Update the token_to_idx dictionary and aligned_embeddings list
    for token in unique_tokens:
        token_to_idx[token] = len(token_to_idx)
        aligned_embeddings.append(glove[token])

    # Convert the list of aligned embeddings to a torch.Tensor
    aligned_embeddings = torch.stack(aligned_embeddings)

    # Save the token_to_idx dictionary and aligned_embeddings tensor to disk
    torch.save(token_to_idx, "token_to_idx.pt")
    torch.save(aligned_embeddings, "word_embeddings.pt")

    return token_to_idx, aligned_embeddings

def snli_collate_fn(token_to_idx, batch):
    """Collate function for the SNLI dataset.

    Args:
        batch (list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]): A batch
            of data from the SNLI dataset (premise_indices, hypothesis_indices, label)
    """
    # Separate premises, hypotheses, and labels
    premises, hypotheses, labels = zip(*[(item['premise'], item['hypothesis'], item['label']) for item in batch])

    # Convert tokens to indices
    premises = [torch.tensor([token_to_idx.get(token, 0) for token in premise]) for premise in premises]
    hypotheses = [torch.tensor([token_to_idx.get(token,0) for token in hypothesis]) for hypothesis in hypotheses]

    # Compute lengths
    premise_lengths = torch.tensor([len(premise) for premise in premises])
    hypothesis_lengths = torch.tensor([len(hypothesis) for hypothesis in hypotheses])

    # Pad sequences in premises and hypotheses using pad_sequence
    padded_premises = pad_sequence(premises, batch_first=True, padding_value=1)
    padded_hypotheses = pad_sequence(hypotheses, batch_first=True, padding_value=1)

    # Convert labels to tensor
    labels = torch.tensor(labels)

    return padded_premises, padded_hypotheses, premise_lengths, hypothesis_lengths, labels

