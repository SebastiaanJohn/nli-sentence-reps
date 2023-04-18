"""Evaluation and prediction functions."""

import argparse
import logging

import numpy as np
import senteval
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import build_vocabulary, get_dataloader, tokenize, tokens_to_indices
from models.classifiers import Classifier
from models.net import NLIModel
from models.utils import get_encoder


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
            (premise, hypothesis,
             premise_lengths, hypothesis_lengths,
             label) = batch

            # Move the batch to the device
            premise = premise.to(device)
            premise_lengths = premise_lengths.to(device)
            hypothesis = hypothesis.to(device)
            hypothesis_lengths = hypothesis_lengths.to(device)
            label = label.to(device)

            # Compute the logits
            logits = model(premise, premise_lengths, hypothesis, hypothesis_lengths)

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

def batcher(params, batch) -> np.ndarray:
    """Evaluate the model on the given batch of sentences (SentEval).

    Args:
        params (dict): SentEval parameters.
        batch (list): Numpy array of text sentences (of size params.batch_size)

    Returns:
        Numpy array of sentence embeddings (of size params.batch_size)
    """
    # Ensure non-empty sentences by replacing empty ones with a dot
    batch = [sent if sent != [] else ['.'] for sent in batch]

    sentence_encoder = params.model.encoder
    token_to_idx = params.token_to_idx

    # Tokenize the sentences
    tokenized_sentences = [tokenize(' '.join(sent)) for sent in batch]

    # Get the indices of the tokens
    indices = [tokens_to_indices(tokens, token_to_idx) for tokens in tokenized_sentences]

    # Pad sequences and compute lengths
    padded_sentences = pad_sequence(indices, batch_first=True, padding_value=1)
    lengths = torch.tensor([len(index) for index in indices])

    # Move the batch to the device
    padded_sentences = padded_sentences.to(params.device)
    lengths = lengths.to(params.device)

    # Compute the sentence embeddings
    sentence_embeddings = sentence_encoder(padded_sentences, lengths)

    # Convert the sentence embeddings to a numpy array
    return sentence_embeddings.detach().cpu().numpy()

def main(args):
    """Evaluate the model."""
    logging.info(f"Args: {args}")
    logging.info("Starting evaluation...")

    # Set the random seed
    torch.manual_seed(args.seed)

    # Create the vocabulary
    logging.info("Loading the vocabulary...")
    token_to_idx, word_embeddings = build_vocabulary("train", args.glove_version, args.input_dim)

    # Load the model
    logging.info("Building the model...")
    encoder = get_encoder(word_embeddings, args.encoder)
    if args.encoder in {"bilstm", "bilstm-max"}:
        classifier_input_dim = 4096
    elif args.encoder == "lstm":
        classifier_input_dim = 2048
    else:
        classifier_input_dim = 300
    classifier = Classifier(classifier_input_dim)
    model = NLIModel(encoder, classifier)

    # Load the checkpoint
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model = model.to(args.device)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    logging.info("Evaluating the model...")

    # Evaluate the model
    if args.eval:
        logging.info("Evaluating on the validation and test sets...")
        valid_loader = get_dataloader("validation", token_to_idx, args)
        test_loader = get_dataloader("test", token_to_idx, args)

        valid_loss, valid_accuracy = evaluate(model, criterion, valid_loader, args.device)
        test_loss, test_accuracy = evaluate(model, criterion, test_loader, args.device)

        logging.info(f"Valid loss: {valid_loss:.3f}")
        logging.info(f"Valid accuracy: {valid_accuracy:.3f}")
        logging.info(f"Test loss: {test_loss:.3f}")
        logging.info(f"Test accuracy: {test_accuracy:.3f}")

    # Evaluate the model using SentEval
    if args.senteval:
        logging.info("Evaluating on SentEval...")
        params = {
            "args": args,
            "task_path": args.senteval_data_path,
            "usepytorch": args.use_pytorch,
            "kfold": args.kfold,
            "device": args.device,
            "token_to_idx": token_to_idx,
            "model": model,
        }
        if args.senteval_fast:
            params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128, 'tenacity': 3, 'epoch_size': 2}

        se = senteval.engine.SE(params, batcher, None)
        transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC', 'SICKEntailment']
        results = se.eval(transfer_tasks)

        logging.info(f"Results: {results}")

        # Compute the macro and micro scores
        tasks_with_devacc = [task for task in transfer_tasks if 'devacc' in results[task]]
        macro_score = np.mean([results[task]['devacc'] for task in tasks_with_devacc])
        micro_score = np.sum(
            [results[task]['ndev'] * results[task]['devacc'] for task in tasks_with_devacc]) / \
                np.sum([results[task]['ndev'] for task in tasks_with_devacc])

        logging.info(f"Macro score: {macro_score:.3f}")
        logging.info(f"Micro score: {micro_score:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint to evaluate")

    # Other parameters
    parser.add_argument("--eval", action="store_true", help="Evaluate the model on the validation and test sets")
    parser.add_argument("--encoder", type=str, default="baseline", choices=["baseline", "lstm", "bilstm", "bilstm-max"],
                        help="Sentence encoder")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")

    # Model parameters
    parser.add_argument("--input_dim", type=int, default=300, help="Input dimension of the word embeddings")

    # Data parameters
    parser.add_argument("--glove_version", type=str, default="840B", choices=["6B", "42B", "840B"],
                        help="GloVe version to use")

    # Training parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for the data loaders")

    # SentEval parameters
    parser.add_argument("--senteval", action="store_true",  help="Use the SentEval evaluation metric")
    parser.add_argument("--senteval_fast", action="store_true",  help="Use the fast version of SentEval")
    parser.add_argument("--senteval_data_path", type=str, default="data", help="Path to the SentEval data directory")
    parser.add_argument("--kfold", type=int, default=5,  help="Number of folds for cross-validation")
    parser.add_argument("--no_pytorch", action="store_false", dest="use_pytorch",
                        help="Use PyTorch for SentEval")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    main(args)

