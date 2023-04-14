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

from .data import SNLIDataset
from .data.utils import snli_collate_fn
from .models import NLIModel
from .models.classifiers import Classifier
from .models.encoders import BaselineEncoder, LSTMEncoder


def train_step(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    batch: torch.Tensor,
    device: torch.device,
    ) -> float:
    """Perform a single training step.

    Args:
        model: The model (encoder + classifier).
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

    # Compute the logits
    logits = model(premise, hypothesis)

    # Compute the loss
    loss = criterion(logits, label)

    # Backpropagate the gradients
    loss.backward()

    # Update the parameters
    optimizer.step()

    return loss.item()

def validate(
    model: nn.Module,
    criterion: nn.Module,
    valid_data: DataLoader,
    device: torch.device,
    ) -> float:
    """Evaluate the model on the development data.

    Args:
        model: The model (encoder + classifier).
        criterion: The loss function.
        valid_data: The development data.
        device: The device to use.

    Returns:
        The average loss on the development data.
    """
    # Set the model to evaluation mode
    model.eval()

    # Keep track of the validation loss
    valid_loss = 0.0

    for batch in tqdm(valid_data, desc="Validating"):
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

        valid_loss += loss.item()

    # Compute the average validation loss
    valid_loss = valid_loss / len(valid_data)

    return valid_loss

def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    epochs: int,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    """Train a model on the training data and evaluate on the development data.

    Args:
        model (nn.Module): The model (encoder + classifier).
        optimizer (torch.optim.Optimizer): The optimizer.
        criterion (nn.Module): The loss function.
        train_loader (DataLoader): The training data.
        valid_loader (DataLoader): The development data.
        epochs (int): The number of epochs to train for.
        device (torch.device): The device to use.
        args (argparse.Namespace): The command line arguments.
    """
    model.to(device)

    best_valid_loss = float("inf")
    writer = SummaryWriter()

    for epoch in range(epochs):
        logging.info(f"Epoch {epoch + 1} / {epochs}")

        # Set the model to training mode
        model.train()

        # Keep track of the training loss
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            loss = train_step(model, optimizer, criterion, batch, device)
            train_loss += loss

        # Compute the average training loss
        train_loss = train_loss / len(train_loader)

        logging.info(f"Train loss: {train_loss:.3f}")

        # Write the train loss to TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)

        # Evaluate the model on the development data
        valid_loss = validate(model, criterion, valid_loader, device)

        logging.info(f"Valid loss: {valid_loss:.3f}")

        # Write the validation loss to TensorBoard
        writer.add_scalar("Loss/valid", valid_loss, epoch)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            logging.info("Saving the best model")

            # Save the model
            model_path = Path("models").joinpath(args.encoder)
            model_path.mkdir(exist_ok=True)
            torch.save(model.state_dict(), model_path / "best_model.pt")

    writer.close()

def get_encoder(args) -> nn.Module:
    """Get the sentence encoder."""
    match args.encoder:
        case "baseline":
            encoder = BaselineEncoder()
        case "uni_lstm":
            encoder = LSTMEncoder(args.embeddings, args.hidden_size)
        case _:
            error = f"Unknown encoder: {args.encoder}"
            raise ValueError(error)

    return encoder

def main(args):
    """Main function for training and evaluating the model."""
    logging.info(f"Args: {args}")

    logging.info("Loading the data...")

    # Set the random seed
    torch.manual_seed(args.seed)

    # Set the device
    device = torch.device(args.device)

    # Load the training data
    train_data = SNLIDataset(
        args.data,
        split="train",
        glove_embedding_size=args.glove_embedding_size,
        subset=args.subset)
    train_dataloader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, collate_fn=snli_collate_fn)

    # Load the validation data
    valid_data = SNLIDataset(
        args.data,
        split="valid",
        vocab=train_data.vocab,
        glove_embedding_size=args.glove_embedding_size,
        subset=args.subset)
    valid_dataloader = DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, collate_fn=snli_collate_fn)

    logging.info("Building the model...")

    # Load the sentence encoder
    encoder = LSTMEncoder(args.embeddings_dim, args.hidden_size)
    classifier = Classifier(args.hidden_size)

    # Define the model
    model = NLIModel(encoder, classifier)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    logging.info("Training the model...")

    # Train the model
    train(model, optimizer, criterion, train_dataloader, valid_dataloader, args.epochs, device, args)

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
