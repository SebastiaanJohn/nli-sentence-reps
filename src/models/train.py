"""The train module contains the train function."""

import logging
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm


def train_step(
    encoder: nn.Module,
    classifier: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    batch: torch.Tensor,
    device: str,
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
    device: str,
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
    device: str,
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
            Path("models").mkdir(exist_ok=True)
            torch.save(encoder.state_dict(), "models/encoder_best.pt")
            torch.save(classifier.state_dict(), "models/classifier_best.pt")

    writer.close()
