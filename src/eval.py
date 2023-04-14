"""Evaluation and prediction functions."""

import argparse
import logging

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    eval_data: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model on the given data.

    Args:
        model (nn.Module): The model (encoder + classifier).
        criterion (nn.Module): The loss function.
        eval_data (DataLoader): The evaluation data.
        device (torch.device): The device to use.

    Returns:
        tuple[float, float]: The average loss and accuracy on the evaluation data.
    """
    # Set the model to evaluation mode
    model.eval()

    # Keep track of the evaluation loss and correct predictions
    eval_loss = 0.0
    correct_predictions = 0

    # Disable gradient computation
    with torch.no_grad():
        for batch in tqdm(eval_data, desc="Evaluating"):
            # Unpack the batch
            premise, hypothesis, label = batch

            # Move the batch to the device
            premise = premise.to(device)
            hypothesis = hypothesis.to(device)
            label = label.to(device)

            # Compute the logits
            logits = model(premise, hypothesis)

            # Compute the loss
            loss = criterion(logits, label)

            eval_loss += loss.item()

            # Get the predictions
            predictions = torch.argmax(logits, dim=-1)

            # Count the correct predictions
            correct_predictions += (predictions == label).sum().item()

        # Compute the average evaluation loss
        eval_loss = eval_loss / len(eval_data)

        # Compute the accuracy
        accuracy = correct_predictions / len(eval_data.dataset)

    return eval_loss, accuracy


def predict(
    model: nn.Module,
    premise: str,
    hypothesis: str,
    tokenizer,
    token_to_idx: dict,
    device: torch.device,
) -> tuple[float, torch.Tensor]:
    """Predict the relationship between the premise and hypothesis using the model.

    Args:
        model: The trained model (encoder + classifier).
        premise (str): The premise text.
        hypothesis (str): The hypothesis text.
        tokenizer: The tokenizer function.
        token_to_idx (dict): Dictionary mapping tokens to their indices in the embedding matrix.
        device: The device to use.

    Returns:
        int: The predicted relationship.
        torch.Tensor: The probabilities of the predicted relationship.
    """
    # Set the model to evaluation mode
    model.eval()

    # Tokenize and convert tokens to indices
    premise_indices = tokenizer(premise)
    hypothesis_indices = tokenizer(hypothesis)

    # Convert indices to tensors and add a batch dimension
    premise_tensor = torch.tensor(premise_indices).unsqueeze(0).to(device)
    hypothesis_tensor = torch.tensor(hypothesis_indices).unsqueeze(0).to(device)

    # Disable gradient computation
    with torch.no_grad():
        # Compute the logits
        logits = model(premise_tensor, hypothesis_tensor)

        # Apply softmax to get probabilities
        probabilities = torch.softmax(logits, dim=-1)

        # Get the predicted relationship
        prediction = torch.argmax(logits, dim=-1).item()

    return prediction, probabilities.squeeze()

def main(args):
    # Load the model
    model = torch.load(args.model_path)

    # Load the tokenizer
    tokenizer = torch.load(args.tokenizer_path)

    # Load the token_to_idx dictionary
    token_to_idx = torch.load(args.token_to_idx_path)

    # Move the model to the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Predict the relationship between the premise and hypothesis
    prediction, probabilities = predict(
        model,
        args.premise,
        args.hypothesis,
        tokenizer,
        token_to_idx,
        device,
    )

    # Print the prediction and probabilities
    print(f"Predicted relationship: {prediction}")
    print(f"Probabilities: {probabilities}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--embeddings_dim", type=int, default=300)
    parser.add_argument("--glove_embedding_size", type=str, default="840B")
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--encoder", type=str, default="baseline", choices=["baseline", "uni_lstm"])
    parser.add_argument("--hidden_size", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main(args)
