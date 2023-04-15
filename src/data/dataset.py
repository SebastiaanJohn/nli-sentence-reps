"""Dataset class for SNLI dataset."""

import json
from collections import defaultdict
from itertools import islice
from pathlib import Path

import torch
from spacy.lang.en import English
from torch.utils.data import Dataset
from torchtext.vocab import GloVe


class SNLIVocabulary:
    """Vocabulary class for SNLI dataset."""
    def __init__(
        self,
        data: list[dict[str, list[str] | torch.Tensor]],
        tokenizer: English,
        glove_version: str = "840B",
        embedding_dim: int = 300,
        ) -> None:
        """Initialize the vocabulary.

        Args:
            data (list): A list of dictionaries containing the data
            tokenizer (English): The tokenizer to use
            glove_version (str): The version of GloVe to use
            embedding_dim (int): The dimension of the GloVe embeddings
        """
        self.glove = GloVe(name=glove_version, dim=embedding_dim)
        self.nlp = tokenizer
        self.token_to_idx, self.word_embedding = self._build(data)

    def _build(self, data: list[dict[str, list[str] | torch.Tensor]]
                          ) -> tuple[dict[str, int], torch.Tensor]:
        """Build the vocabulary from the dataset.

        Returns:
            token_to_idx (dict): A dictionary mapping tokens to indices
            aligned_embeddings (torch.Tensor): A tensor of aligned embeddings
        """
        # Create a default dictionary for the token to index mapping
        token_to_idx = defaultdict(lambda: len(token_to_idx))
        token_to_idx["<UNK>"] = 0  # Adding an unknown token
        token_to_idx["<PAD>"] = 1  # Adding a padding token

        # Compute the average embedding for the unknown token
        unk_embedding = self.glove.vectors.mean(dim=0)

        # Create a list of aligned embeddings
        aligned_embeddings = [unk_embedding, self.glove["<PAD>"]]
        for item in data:
            for text_field in ["premise", "hypothesis"]:
                for token in item[text_field]:
                    if token not in token_to_idx:
                        token_to_idx[token] = len(token_to_idx)
                        aligned_embeddings.append(self.glove[token])

        # Convert the list of aligned embeddings to a torch.Tensor
        aligned_embeddings = torch.stack(aligned_embeddings)

        return token_to_idx, aligned_embeddings

    def tokenize_and_index(self, text: str) -> torch.Tensor:
        """Tokenize and index a string of text."""
        tokens = [token.text.lower() for token in self.nlp.tokenizer(text)]
        indices = torch.tensor([self.token_to_idx.get(token, 0) for token in tokens])
        return indices

    def token_to_index(self, tokens: list[str]) -> torch.Tensor:
        """Return the index of the given token."""
        indices = torch.tensor([self.token_to_idx.get(token, 0) for token in tokens])
        return indices

class SNLIDataset(Dataset):
    """SNLI dataset class."""
    def __init__(
        self,
        root: str,
        split: str = 'train',
        embedding_dim: int = 300,
        glove_version: str = '840B',
        vocab: SNLIVocabulary | None = None,
        subset: int | None = None,
        ) -> None:
        """Initialize the dataset.

        Args:
            root (str): Path to the dataset directory.
            split (str, optional): Split of the dataset to load. Defaults to
                'train'.
            embedding_dim (int, optional): Dimension of the word embeddings.
                Defaults to 300.
            glove_version (str, optional): Version of GloVe embeddings to use.
                Defaults to '840B'.
            vocab (SNLIVocabulary, optional): Vocabulary to use for the dataset.
                If None, build the vocabulary from the provided data. Defaults
                to None.
            subset (int, optional): Number of examples to use from the dataset.
        """
        if split not in {'train', 'dev', 'test'}:
            error = f"Invalid split: {split}. Must be one of 'train', 'dev', or 'test'."
            raise ValueError(error)

        self.root = Path(root).joinpath('snli_1.0')
        self.label_to_id = {
            "entailment": 0,
            "neutral": 1,
            "contradiction": 2
        }
        self.nlp = English()
        self.data = self._load_data(split, subset)

        if vocab is None:
            self.vocab = SNLIVocabulary(self.data, self.nlp, glove_version, embedding_dim)
        else:
            self.vocab = vocab

    def _load_data(self, split: str, subset: int | None = None
                   ) -> list[dict[str, list[str] | torch.Tensor]]:
        """Load the dataset from the given split.

        Args:
            split (str): Split of the dataset to load.
            subset (int, optional): Number of examples to use from the dataset.
                If None, use the entire dataset. Defaults to None.

        Returns:
            list[dict[str, list[str] | torch.Tensor]]: The dataset as a list of
                dictionaries. Each dictionary contains the premise, hypothesis,
                and label of the premise-hypothesis pair.
        """
        data_file = self.root.joinpath(f'snli_1.0_{split}.jsonl')

        if not data_file.exists():
            error = f"Dataset not found at {self.root}. Please download the dataset " \
                    f"using the script in the data directory of the repository."
            raise FileNotFoundError(error)

        # Generator that yields the required information from the raw data
        def data_gen():
            with data_file.open('r') as f:
                for line in f:
                    item = json.loads(line)
                    if item["gold_label"] != "-":
                        yield {
                            "premise": self._tokenize(item["sentence1"]),
                            "hypothesis": self._tokenize(item["sentence2"]),
                            "label": self.label_to_id[item["gold_label"]],
                        }

        data = list(islice(data_gen(), subset)) if subset else list(data_gen())

        return data

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize a string of text.

        Args:
            text (str): The text to tokenize

        Returns:
            tokens (list): A list of tokens
        """
        return [token.text.lower() for token in self.nlp.tokenizer(text)]

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return an item from the dataset.

        Args:
            index (int): Index of the item to return.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Premise embeddings, hypothesis embeddings, and label.
        """
        item = self.data[index]
        premise_indices = self.vocab.token_to_index(item['premise'])
        hypothesis_indices = self.vocab.token_to_index(item['hypothesis'])
        label = torch.tensor(item['label'])

        return premise_indices, hypothesis_indices, label

