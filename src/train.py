"""The train module contains the train function."""

import argparse
import logging
from pathlib import Path

import torch
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from src.data import SNLIDataset
from src.data.utils import snli_collate_fn
from src.eval import evaluate
from src.models import NLIModel
from src.models.classifiers import Classifier
from src.models.encoders import BaselineEncoder, LSTMEncoder


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

def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
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
        scheduler (ReduceLROnPlateau): The learning rate scheduler.
        criterion (nn.Module): The loss function.
        train_loader (DataLoader): The training data.
        valid_loader (DataLoader): The development data.
        epochs (int): The number of epochs to train for.
        device (torch.device): The device to use.
        args (argparse.Namespace): The command line arguments.
    """
    model.to(device)

    best_valid_loss = float("inf")
    best_valid_accuracy = 0.0
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
        train_loss = train_loss / len(train_loader.dataset)

        logging.info(f"Train loss: {train_loss:.3f}")

        # Write the train loss to TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)

        # Evaluate the model on the development data
        valid_loss, valid_accuracy = evaluate(model, criterion, valid_loader, device)

        logging.info(f"Valid loss: {valid_loss:.3f}")
        logging.info(f"Accuracy: {valid_accuracy:.3f}")

        # Update the learning rate
        scheduler.step(valid_accuracy)

        # Write the validation loss and accuracy to TensorBoard
        writer.add_scalar("Loss/valid", valid_loss, epoch)
        writer.add_scalar("Accuracy/valid", valid_accuracy, epoch)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_valid_accuracy = valid_accuracy
            logging.info("Saving the best model")

            # Save the model
            model_path = Path("models")
            model_path.mkdir(exist_ok=True)

            encoder_path = model_path / args.encoder
            encoder_path.mkdir(exist_ok=True)

            torch.save(model.state_dict(), encoder_path / "best_model.pt")

    logging.info(f"Best valid loss: {best_valid_loss:.3f}")
    logging.info(f"Best valid accuracy: {best_valid_accuracy:.3f}")

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
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4)

    # Define the learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=1/5, patience=0, min_lr=1e-5, verbose=True)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    logging.info("Training the model...")

    # Train the model
    train(model, optimizer, scheduler, criterion, train_dataloader, valid_dataloader, args.epochs, device, args)

    # Load the best model
    model.load_state_dict(torch.load(f"models/{args.encoder}/best_model.pt"))

    # Evaluate the model on the test data
    test_data = SNLIDataset(
        args.data,
        split="test",
        vocab=train_data.vocab,
        glove_embedding_size=args.glove_embedding_size,
        subset=args.subset)
    test_dataloader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, collate_fn=snli_collate_fn)
    test_loss, test_accuracy = evaluate(model, criterion, test_dataloader, device)

    logging.info(f"Test loss: {test_loss:.3f}")
    logging.info(f"Test accuracy: {test_accuracy:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--embeddings_dim", type=int, default=300)
    parser.add_argument("--glove_embedding_size", type=str, default="840B")
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--encoder", type=str, default="baseline", choices=["baseline", "lstm"])
    parser.add_argument("--hidden_size", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
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
