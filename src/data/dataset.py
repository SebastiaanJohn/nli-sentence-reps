"""Dataset class for SNLI dataset."""

import json
from pathlib import Path

import torch
from spacy.lang.en import English
from torch.utils.data import Dataset
from torchtext.vocab import GloVe


nlp = English()

class SNLI(Dataset):
    """SNLI dataset class.

    The dataset consists of the following fields:
        - premise: The premise of the hypothesis.
        - hypothesis: The hypothesis to be proven.
        - label: The label of the premise-hypothesis pair, where 0 is
            entailment, 1 is neutral, and 2 is contradiction.
    """

    def __init__(self, root: str, split: str = 'train', embedding_dim: int = 300):
        """Initialize the dataset.

        Args:
            root (str): Path to the dataset.
            split (str, optional): Split of the dataset to load. Defaults to
                'train'.
            embedding_dim (int, optional): Dimension of the word embeddings.
                Defaults to 300.
        """
        if split not in ['train', 'valid', 'test']:
            error = f"Invalid split: {split}. Must be one of 'train', 'valid', " \
                    f"or 'test'."
            raise ValueError(error)

        if split == 'valid':
            split = 'dev'

        self.root = Path(root + '/snli_1.0')
        self.label_mapping = {
            "entailment": 0,
            "neutral": 1,
            "contradiction": 2
        }
        self.glove = GloVe(name='6B', dim=embedding_dim)
        self.data = self._load_data(split)

    def _load_data(self, split: str) -> list[dict[str, torch.Tensor]]:
        """Load the dataset.

        Args:
            split (str): Split of the dataset to load.

        Returns:
            list[dict[str, torch.Tensor]]: List of items in the dataset with the
                (premise, hypothesis, label).
        """
        with self.root.joinpath(f'snli_1.0_{split}.jsonl').open() as f:
            raw_data = [json.loads(line) for line in f]

        # Extract the required information from the raw data and process it
        data = [
            {
                "premise": self.tokenize_and_embed(item["sentence1"]),
                "hypothesis": self.tokenize_and_embed(item["sentence2"]),
                "label": torch.tensor(self.label_mapping[item["gold_label"]]),
            }
            for item in raw_data if item["gold_label"] != "-"
        ]

        return data

    def tokenize_and_embed(self, text: str) -> torch.Tensor:
        """Tokenize and embed the given text.

        Args:
            text (str): Text to tokenize and embed.

        Returns:
            torch.Tensor: Tensor of word embeddings for the given text
                (shape: (seq_len, embedding_dim)).
        """
        tokens = [token.text.lower() for token in nlp.tokenizer(text)]
        embeddings = [self.glove[token] for token in tokens]

        return torch.stack(embeddings)


    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        """Return an item from the dataset.

        Args:
            index (int): Index of the item to return.

        Returns:
            dict: Item from the dataset at the given index.
        """
        return self.data[index]

if __name__ == '__main__':
    dataset = SNLI(root='data', split='valid')
    print(dataset[0])
    print(dataset[0]['premise'].shape)
    print(dataset[0]['hypothesis'].shape)
    print(dataset[1]['premise'].shape)
