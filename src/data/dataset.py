"""Dataset class for SNLI dataset."""

import json
from collections import defaultdict
from itertools import islice
from pathlib import Path

import torch
import torch.nn as nn
from spacy.lang.en import English
from torch.utils.data import Dataset
from torchtext.vocab import GloVe


nlp = English()

class SNLIDataset(Dataset):
    """SNLI dataset class.

    The dataset consists of the following fields:
        - premise: The premise of the hypothesis.
        - hypothesis: The hypothesis to be proven.
        - label: The label of the premise-hypothesis pair, where 0 is
            entailment, 1 is neutral, and 2 is contradiction.
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        embedding_dim: int = 300,
        glove_embedding_size: str = '840B',
        vocab: dict[str, torch.Tensor | dict[str, int]] | None = None,
        subset: int | None = None,
        ) -> None:
        """Initialize the dataset.

        Args:
            root (str): Path to the dataset.
            split (str, optional): Split of the dataset to load. Defaults to
                'train'.
            embedding_dim (int, optional): Dimension of the word embeddings.
                Defaults to 300.
            glove_embedding_size (str, optional): Size of the GloVe embeddings
                to use. Defaults to '840B'.
            vocab (dict[str, torch.Tensor | dict[str, int]], optional): Vocabulary
                containing the word embeddings and word-to-index mapping. Defaults
                to None.
            subset (int, optional): Number of examples to use from the dataset.
        """
        if split not in {'train', 'valid', 'test'}:
            error = f"Invalid split: {split}. Must be one of 'train', 'valid', or 'test'."
            raise ValueError(error)

        if split == 'valid':
            split = 'dev'

        self.root = Path(root).joinpath('snli_1.0')
        self.label_mapping = {
            "entailment": 0,
            "neutral": 1,
            "contradiction": 2
        }
        self.glove = GloVe(name=glove_embedding_size, dim=embedding_dim)
        self.data = self._load_data(split, subset)

        if vocab is None: # Build the vocabulary from the training data
            self.vocab = self._build_vocabulary()
        else: # Use the vocabulary from the validation or test data
            self.vocab = vocab

        self.embedding = nn.Embedding.from_pretrained(self.vocab["embedding"])
        self.token_to_idx = self.vocab["token_to_idx"]

    def _load_data(self, split: str, subset: int | None) -> list[dict[str, list[str] | torch.Tensor]]:
        """Load the dataset.

        Args:
            split (str): Split of the dataset to load.
            subset (int, optional): Number of examples to use from the dataset.

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
                            "label": torch.tensor(self.label_mapping[item["gold_label"]]),
                        }

        data = list(islice(data_gen(), subset)) if subset else list(data_gen())

        return data

    def _build_vocabulary(self) -> dict[str, torch.Tensor | dict[str, int]]:
        """Build the vocabulary from the dataset.

        Returns:
            dict[str, torch.Tensor | dict[str, int]]: Dictionary containing the
                aligned embedding matrix and the token to index mapping.
        """
        # Create a default dictionary for the token to index mapping
        token_to_idx = defaultdict(lambda: len(token_to_idx))
        token_to_idx["<PAD>"] = 0  # Adding a padding token
        token_to_idx["<UNK>"] = 1  # Adding an unknown token

        # Compute the average embedding for the unknown token
        unk_embedding = self.glove.vectors.mean(dim=0)

        # Initialize the aligned embedding matrix with the size of the vocabulary
        # and the GloVe embedding dimension
        aligned_embeddings = [self.glove["<PAD>"], unk_embedding]

        for item in self.data:
            for text_field in ["premise", "hypothesis"]:
                for token in item[text_field]:
                    if token not in token_to_idx:
                        token_to_idx[token] = len(token_to_idx)
                        aligned_embeddings.append(self.glove[token])

        # Convert the list of aligned embeddings to a torch.Tensor
        aligned_embeddings = torch.stack(aligned_embeddings)

        return {
            "embedding": aligned_embeddings,
            "token_to_idx": token_to_idx
        }


    def _tokenize(self, text: str) -> list[str]:
        """Tokenize the given text.

        Args:
            text (str): Text to tokenize.

        Returns:
            list[str]: List of tokens for the given text.
        """
        tokens = [token.text.lower() for token in nlp.tokenizer(text)]

        return tokens

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
        premise_indices = torch.tensor(
            [self.token_to_idx.get(token, 1) for token in item['premise']]
        )
        hypothesis_indices = torch.tensor(
            [self.token_to_idx.get(token, 1) for token in item['hypothesis']]
        )
        label = item['label']

        premise_embeddings = self.embedding(premise_indices)
        hypothesis_embeddings = self.embedding(hypothesis_indices)

        return premise_embeddings, hypothesis_embeddings, label


if __name__ == "__main__":
    import time

    from torch.utils.data import DataLoader
    start = time.time()
    dataset = SNLIDataset(root='data', subset=1000)
    snli_dataloader = DataLoader(dataset, batch_size=100, shuffle=True, collate_fn=snli_collate_fn)

    for batch in snli_dataloader:
        premise, hypothesis, label = batch
        print(premise.shape, hypothesis.shape, label.shape)
