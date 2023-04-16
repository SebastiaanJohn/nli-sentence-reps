# Learning Sentence Representations From Natural Language Inference Data

This repository contains the code for the paper [Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](https://arxiv.org/abs/1705.02364) for the course Advanced Topics in Computational Semantics at the University of Amsterdam.

## Structure

The repository is structured as follows:

* `data/` contains the scripts to download the data for the experiments and SentEval.
* `models/` contains the pre-trained models.
* `notebooks/` contains the notebooks with the results for the experiments.
* `runs/` contains the Tensorboard logs.
* `src/` contains the source code of the project.

## Requirements

The code is written in Python 3.10. The requirements can be installed using `pip install -r requirements.txt`.

## Datasets

The [The Stanford Natural Language Inference (SNLI)](https://nlp.stanford.edu/projects/snli/) can be downloaded using the following command:

```bash
sh /data/download_dataset.sh
```

The [SentEval](https://github.com/facebookresearch/SentEval) datasets can be downloaded using the following command:

```bash
bash data/downstream/get_transfer_data.bash
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