# Learning Sentence Representations From Natural Language Inference Data

This repository contains the code for the paper [Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](https://arxiv.org/abs/1705.02364) for the course Advanced Topics in Computational Semantics at the University of Amsterdam.

## Structure

The repository is structured as follows:

* `data/` contains the scripts to download the data for the experiments and SentEval. After training the vocabulary and embeddings are stored here as well.
* `logs/` contains the Lisa logs from the training.
* `models/` contains the pre-trained models.
* `runs/` contains the Tensorboard logs. The logs are stored in a directory with the name of the model.
* `src/` contains the source code of the project.
* `results.ipynb` contains the prediction code, results of the experiments, and discussion.
* `requirements.txt` contains the requirements for the project.
* `README.md` contains the instructions for the project.
* `pyproject.toml` contains the project configuration.

## Requirements

The code is written in Python 3.10. The requirements can be installed using `pip install -r requirements.txt` or with the conda environment file `conda env create -f environment.yml`.

## Datasets

The [The Stanford Natural Language Inference (SNLI)](https://nlp.stanford.edu/projects/snli/) will be downloaded automatically when running the training script. The [SentEval](https://github.com/facebookresearch/SentEval) datasets can be downloaded using the following command from the `data/downstream` directory:

```bash
bash ./get_transfer_data.bash
```

## Usage

You can train a model using the following command:

```bash
python src/train.py --encoder <encoder>
```

You can evaluate a model using the following command:

```bash
python src/eval.py --checkpoint <checkpoint> --encoder <encoder> --eval --senteval
```

The `--eval` flag will evaluate the model on the SNLI dataset. The `--senteval` flag will evaluate the model on the SentEval datasets. See `-h` for more options such as changing the batch size or the number of epochs.

## Models

The pre-trained models can be downloaded from [here](https://drive.google.com/drive/folders/1xkhM0sZonRz2Nh0RIuQRTQ2JO6OoAAcJ?usp=share_link). The models should be placed in the `models/` directory.
