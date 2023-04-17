"""This module contains the code for building the dataset and helper functions."""

import argparse
import logging
from functools import partial
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from spacy.lang.en import English
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.vocab import GloVe


nlp = English()

def tokenize(text: str) -> list[str]:
    """Tokenize a sentence.

    Args:
        text (str): The sentence to tokenize

    Returns:
        list[str]: The list of tokens
    """
    return [token.text.lower() for token in nlp.tokenizer(text)]

def tokenize_batch(batch: dict) -> dict:
    """Tokenize the premise and hypothesis of a batch of examples.

    Args:
        batch (dict): A batch of examples from the dataset

    Returns:
        dict: A dictionary containing the tokenized premise and hypothesis
    """
    return {
        "premise": [tokenize(text) for text in batch["premise"]],
        "hypothesis": [tokenize(text) for text in batch["hypothesis"]],
    }

def tokens_to_indices(tokens: list[str], token_to_idx: dict[str, int]) -> torch.Tensor:
    """Convert the tokens in a sentence to indices.

    Args:
        tokens (list[str]): The tokens to convert to indices
        token_to_idx (dict[str, int]): A dictionary mapping tokens to indices

    Returns:
        dict: A dictionary containing the premise and hypothesis as a list of indices
    """
    return torch.tensor([token_to_idx.get(token, 0) for token in tokens])

def build_vocabulary(
    split: str,
    glove_version: str = "840B",
    word_embedding_dim: int = 300,
    ) -> tuple[dict[str, int], torch.Tensor]:
    """Build the vocabulary from the dataset.

    Args:
        split (str): The split of the dataset to use for building the vocabulary
        glove_version (str): The version of GloVe to use
        word_embedding_dim (int): The dimension of the GloVe embeddings

    Returns:
        token_to_idx (dict): A dictionary mapping tokens to indices
        aligned_embeddings (torch.Tensor): A tensor of aligned embeddings
    """
    # Check if the vocabulary has already been built
    if Path(f"./data/token_to_idx_{glove_version}_{word_embedding_dim}.pt").exists() \
        and Path(f"./data/word_embeddings_{glove_version}_{word_embedding_dim}.pt").exists():

        logging.info("Loading vocabulary from disk...")
        token_to_idx = torch.load(
            f"./data/token_to_idx_{glove_version}_{word_embedding_dim}.pt")
        aligned_embeddings = torch.load(
            f"./data/word_embeddings_{glove_version}_{word_embedding_dim}.pt")

        return token_to_idx, aligned_embeddings

    logging.info("Building vocabulary from scratch...")

    # Load the SNLI dataset
    dataset = get_dataset(split=split)

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
    torch.save(token_to_idx, f"./data/token_to_idx_{glove_version}_{word_embedding_dim}.pt")
    torch.save(aligned_embeddings, f"./data/word_embeddings_{glove_version}_{word_embedding_dim}.pt")

    return token_to_idx, aligned_embeddings

def snli_collate_fn(token_to_idx, batch):
    """Collate function for the SNLI dataset.

    Args:
        token_to_idx (dict): A dictionary mapping tokens to indices
        batch (list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]): A batch
            of data from the SNLI dataset (premise_indices, hypothesis_indices, label)
    """
    # Separate premises, hypotheses, and labels
    premises, hypotheses, labels = zip(*[(item['premise'], item['hypothesis'], item['label']) for item in batch])

    # Convert tokens to indices
    premises = [tokens_to_indices(premise, token_to_idx) for premise in premises]
    hypotheses = [tokens_to_indices(hypothesis, token_to_idx) for hypothesis in hypotheses]

    # Compute lengths
    premise_lengths = torch.tensor([len(premise) for premise in premises])
    hypothesis_lengths = torch.tensor([len(hypothesis) for hypothesis in hypotheses])

    # Pad sequences in premises and hypotheses using pad_sequence
    padded_premises = pad_sequence(premises, batch_first=True, padding_value=1)
    padded_hypotheses = pad_sequence(hypotheses, batch_first=True, padding_value=1)

    # Convert labels to tensor
    labels = torch.tensor(labels)

    return padded_premises, padded_hypotheses, premise_lengths, hypothesis_lengths, labels

def get_dataset(split: str) -> Dataset:
    """Get the dataset and tokenize the premise and hypothesis.

    Args:
        split (str): The split to use

    Returns:
        Dataset: Tokenized dataset
    """
    if split not in {"train", "validation", "test"}:
        error = f"Invalid split: {split}. Must be one of 'train', 'validation', or 'test'."
        raise ValueError(error)

    dataset = load_dataset("snli", split=split)
    dataset = dataset.filter(lambda example: example["label"] != -1)
    dataset = dataset.map(tokenize_batch, batched=True, batch_size=1000)

    return dataset

def get_dataloader(split: str, token_to_idx: dict[str, int], args: argparse.Namespace) -> DataLoader:
    """Get the validation dataloader.

    Args:
        split (str): The split to use
        token_to_idx (dict[str, int]): The vocabulary
        args (argparse.Namespace): The arguments

    Returns:
        DataLoader: The validation dataloader
    """
    dataset = get_dataset(split)

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(snli_collate_fn, token_to_idx),
        num_workers=args.num_workers,
    )
