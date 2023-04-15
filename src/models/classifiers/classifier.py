"""Classifier model for as proposed by Conneau et al. (2017)."""

import torch


class Classifier(torch.nn.Module):
    """Classifier model for as proposed by Conneau et al. (2017)."""

    def __init__(
        self,
        sentence_rep_dim: int,
        hidden_dim: int = 512,
        num_classes: int = 3
        ):
        """Initialize the model.

        Args:
            sentence_rep_dim (int): Dimension of the sentence embeddings.
            hidden_dim (int, optional): Dimension of the hidden layer in the
                classifier. Defaults to 512.
            num_classes (int, optional): Number of classes in the dataset.
                Defaults to 3.
        """
        super().__init__()
        self.sentence_rep_dim = sentence_rep_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(4 * self.sentence_rep_dim, self.hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden_dim, self.num_classes)
        )

    def forward(self, encoded_premise: torch.Tensor, encoded_hypothesis: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        3 matching methods are applied to extract relations between u and v:
        (i) concatenation of the two representa- tions (u, v);
        (ii) element-wise product u * v; and
        (iii) absolute element-wise difference |u - v|.

        The resulting 4 * d-dimensional vector is then passed through a two-layer MLP.

        Args:
            encoded_premise (torch.Tensor): Tensor of premise embeddings.
            encoded_hypothesis (torch.Tensor): Tensor of hypothesis embeddings.

        Returns:
            torch.Tensor: Tensor of concatenated sentence embeddings (u, v, u * v, |u - v|)
        """
        # Concatenate the premise and hypothesis embeddings (u, v)
        concat = torch.cat((encoded_premise, encoded_hypothesis), dim=1)

        # Element-wise product of the premise and hypothesis embeddings (u * v)
        product = encoded_premise * encoded_hypothesis

        # Absolute element-wise difference of the premise and hypothesis embeddings (|u - v|)
        difference = torch.abs(encoded_premise - encoded_hypothesis)

        # Concatenate the concatenated, product, and difference embeddings (u, v, u * v, |u - v|)
        concatenated = torch.cat((concat, product, difference), dim=1)

        # Pass the concatenated embeddings through a two-layer MLP
        return self.mlp(concatenated)
