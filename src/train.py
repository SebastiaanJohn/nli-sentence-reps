"""The train module contains the train function."""

import argparse
import logging
from pathlib import Path

import torch
import torch.optim as optim
from eval import evaluate
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from data.dataset import SNLIDataset
from data.utils import snli_collate_fn
from models.classifiers import Classifier
from models.net import NLIModel
from models.utils import get_encoder


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
    premise, hypothesis, premise_lengths, hypothesis_lengths, label = batch

    # Move the batch to the device
    premise = premise.to(device)
    premise_lengths = premise_lengths.to(device)
    hypothesis = hypothesis.to(device)
    hypothesis_lengths = hypothesis_lengths.to(device)
    label = label.to(device)

    # Zero out the gradients
    optimizer.zero_grad()

    # Compute the logits
    logits = model(premise, premise_lengths, hypothesis, hypothesis_lengths)

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
        train_loss = train_loss / len(train_loader)

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

    writer.flush()
    writer.close()

def main(args):
    """Main function for training and evaluating the model."""
    logging.info(f"Args: {args}")
    logging.info("Starting training...")

    # Set the random seed
    torch.manual_seed(args.seed)

    # Set the device
    device = torch.device(args.device)

    logging.info("Loading the data...")

    # Load the training data
    train_data = SNLIDataset(
        args.data_path,
        split="train",
        glove_version=args.glove_version,
        subset=args.subset)
    train_dataloader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, collate_fn=snli_collate_fn)

    # Get the vocabulary
    vocab = train_data.vocab

    # Load the validation data
    valid_data = SNLIDataset(
        args.data_path,
        split="dev",
        vocab=vocab,
        subset=args.subset)
    valid_dataloader = DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, collate_fn=snli_collate_fn)

    logging.info("Building the model...")

    # Load the sentence encoder and the classifier
    encoder = get_encoder(vocab.word_embedding, args)
    if args.encoder == "bilstm" or args.encoder == "bilstm-max":
        args.hidden_size = 2 * args.hidden_size
    classifier = Classifier(args.hidden_size)

    # Define the model
    model = NLIModel(encoder, classifier).to(device)

    # Load the model from a checkpoint if one is provided
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))

    # Define the optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    # Define the learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=1/5, patience=0, min_lr=1e-5, verbose=True)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    logging.info("Training the model...")

    # Train the model
    train(model, optimizer, scheduler, criterion, train_dataloader, valid_dataloader, args.epochs, device, args)

    # Load the best model
    model.load_state_dict(torch.load(f"models/{args.encoder}/best_model.pt"))

    # Load the test data
    test_data = SNLIDataset(
        args.data_path,
        split="test",
        vocab=vocab,
        subset=args.subset)
    test_dataloader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, collate_fn=snli_collate_fn)

    # Evaluate the model on the test data
    test_loss, test_accuracy = evaluate(model, criterion, test_dataloader, device)

    logging.info(f"Test loss: {test_loss:.3f}")
    logging.info(f"Test accuracy: {test_accuracy:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data", help="Path to the data directory")
    parser.add_argument("--embeddings_dim", type=int, default=300, help="Embeddings dimension")
    parser.add_argument("--glove_version", type=str, default="840B", choices=["6B", "42B", "840B"], help="GloVe version")
    parser.add_argument("--subset", type=int, default=None, help="Subset of the data to use for training")
    parser.add_argument("--encoder", type=str, default="baseline", choices=["baseline", "lstm", "bilstm", "bilstm-max"], help="Sentence encoder")
    parser.add_argument("--hidden_size", type=int, default=300, help="Hidden size of the LSTM")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a checkpoint to load the model from")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    main(args)
