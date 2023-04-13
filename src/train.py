"""The train module contains the train function."""

import argparse
import logging
from pathlib import Path

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from .data import SNLI
from .models.classifiers import Classifier
from .models.encoders import BaselineEncoder, UniLSTMEncoder


def train_step(
    encoder: nn.Module,
    classifier: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    batch: torch.Tensor,
    device: torch.device,
    ) -> float:
    """Perform a single training step.

    Args:
        encoder: The sentence encoder.
        classifier: The classifier.
        optimizer: The optimizer.
        criterion: The loss function.
        batch: The batch of data.
        device: The device to use.

    Returns:
        The loss value for the batch.
    """
    # Unpack the batch
    premise, hypothesis, label = batch

    # Move the batch to the device
    premise = premise.to(device)
    hypothesis = hypothesis.to(device)
    label = label.to(device)

    # Zero out the gradients
    optimizer.zero_grad()

    # Compute the sentence representations
    premise_rep = encoder(premise)
    hypothesis_rep = encoder(hypothesis)

    # Classify the premise-hypothesis pairs
    logits = classifier(premise_rep, hypothesis_rep)

    # Compute the loss
    loss = criterion(logits, label)

    # Backpropagate the gradients
    loss.backward()

    # Update the parameters
    optimizer.step()

    return loss.item()

def validate(
    encoder: nn.Module,
    classifier: nn.Module,
    criterion: nn.Module,
    valid_data: DataLoader,
    device: torch.device,
    ) -> float:
    """Evaluate the model on the development data.

    Args:
        encoder: The sentence encoder.
        classifier: The classifier.
        criterion: The loss function.
        valid_data: The development data.
        device: The device to use.

    Returns:
        The average loss on the development data.
    """
    # Set the model to evaluation mode
    encoder.eval()
    classifier.eval()

    # Keep track of the validation loss
    valid_loss = 0.0

    for batch in valid_data:
        # Unpack the batch
        premise, hypothesis, label = batch

        # Move the batch to the device
        premise = premise.to(device)
        hypothesis = hypothesis.to(device)
        label = label.to(device)

        # Compute the sentence representations
        premise_rep = encoder(premise)
        hypothesis_rep = encoder(hypothesis)

        # Classify the premise-hypothesis pairs
        logits = classifier(premise_rep, hypothesis_rep)

        # Compute the loss
        loss = criterion(logits, label)

        valid_loss += loss.item()

    # Compute the average validation loss
    valid_loss = valid_loss / len(valid_data)

    return valid_loss

def train(
    encoder: nn.Module,
    classifier: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    train_data: DataLoader,
    valid_data: DataLoader,
    epochs: int,
    device: torch.device
) -> None:
    """Train a model on the training data and evaluate on the development data.

    Args:
        encoder: The sentence encoder.
        classifier: The classifier.
        optimizer: The optimizer.
        criterion: The loss function.
        train_data: The training data.
        valid_data: The development data.
        epochs: The number of epochs to train for.
        device: The device to use.
    """
    encoder.to(device)
    classifier.to(device)

    best_valid_loss = float("inf")
    writer = SummaryWriter()

    for epoch in range(epochs):
        logging.info(f"Epoch {epoch + 1} / {epochs}")
        logging.info("-" * 10)

        # Set the model to training mode
        encoder.train()
        classifier.train()

        # Keep track of the training loss
        train_loss = 0.0

        for batch in tqdm(train_data, desc=f"Epoch {epoch + 1}"):
            loss = train_step(encoder, classifier, optimizer, criterion, batch, device)
            train_loss += loss

        # Compute the average training loss
        train_loss = train_loss / len(train_data)

        # Write the train loss to TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)

        # Evaluate the model on the development data
        valid_loss = validate(encoder, classifier, criterion, valid_data, device)

        # Write the validation loss to TensorBoard
        writer.add_scalar("Loss/valid", valid_loss, epoch)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            logging.info("Saving the best model")

            # Save the model
            model_path = Path("models")
            model_path.mkdir(exist_ok=True)
            torch.save(encoder.state_dict(), model_path / "encoder_best.pt")
            torch.save(classifier.state_dict(), model_path / "classifier_best.pt")

    writer.close()

def get_encoder(args) -> nn.Module:
    """Get the sentence encoder."""
    if args.encoder == "baseline":
        encoder = BaselineEncoder(args.embeddings)
    elif args.encoder == "uni_lstm":
        encoder = UniLSTMEncoder(args.embeddings, args.hidden_size)
    else:
        error = f"Unknown encoder: {args.encoder}"
        raise ValueError(error)

    return encoder

def main(args):
    """Main function for training and evaluating the model."""
    # Set the random seed
    torch.manual_seed(args.seed)

    # Set the device
    device = torch.device(args.device)

    # Load the training data
    train_data = SNLI(args.data, split="train", glove_embedding_size=args.glove_embedding_size)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    # Load the validation data
    valid_data = SNLI(args.data, split="valid", vocab=train_data.vocab)
    valid_dataloader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)

    # Load the sentence encoder
    encoder = get_encoder(args)
    classifier = Classifier(args.hidden_size)

    # Move the model to the device
    encoder.to(device)
    classifier.to(device)

    # Define the optimizer
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=args.learning_rate,
    )

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Train the model
    train(encoder, classifier, optimizer, criterion, train_dataloader, valid_dataloader, args.epochs, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--glove_embedding_size", type=str, default="840B")
    parser.add_argument("--encoder", type=str, default="baseline")
    parser.add_argument("--hidden_size", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    main(args)
