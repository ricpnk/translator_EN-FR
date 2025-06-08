# Homework 5 – Natural Language Processing

This project is intended for academic purposes as part of the KI430 module at Hochschule Landshut.

## Description
This project implements a neural machine translation system for English to French translation using a sequence-to-sequence model with attention mechanism. The implementation includes data preprocessing, vocabulary management, model training, and evaluation using BLEU score.

## Project Structure

- `src/`: Source code directory
  - `models.py`: Contains the model architecture (Encoder, Decoder, Attention, Seq2Seq)
  - `Vocab.py`: Vocabulary management class
  - `Translation_Data.py`: Dataset handling class
- `data/`: Directory containing the translation dataset
- `notebooks/`: Jupyter notebooks for data exploration and testing
- `saved_models/`: Directory for storing model checkpoints and attention visualizations
- `pyproject.toml`: Project metadata and dependencies
- `README.md`: Project description and instructions

## How to Run

This project uses [uv](https://github.com/astral-sh/uv) to manage dependencies and execute scripts in the virtual environment.

### Install Dependencies

If you haven't already, install `uv`:

```bash
pip install uv
```

Then install all project dependencies:

```bash
uv init
```

### Run the Scripts

To execute any of the scripts, use:

```bash
uv run <script.py>
```

For example:

```bash
uv run main.py
```

## File Descriptions

### `main.py`
The main script that handles:
- Data loading and preprocessing
- Model training and evaluation
- Attention visualization
- Model checkpointing
- BLEU score calculation

### `src/Vocab.py`
Implements the vocabulary management system:
- Word-to-index and index-to-word mappings
- Special token handling (`<pad>`, `<sos>`, `<eos>`, `<unk>`)
- Sentence encoding and decoding

### `src/Translation_Data.py`
Handles the dataset operations:
- Data loading and preprocessing
- Batch creation and padding
- Input/output sequence management

### `src/models.py`
Contains the neural network architecture:
- Encoder with GRU
- Decoder with GRU and attention mechanism
- Attention mechanism implementation
- Seq2Seq model combining encoder and decoder

### `notebooks/data_exploration.ipynb`
Jupyter notebook for:
- Data analysis and visualization
- Length distribution analysis
- Vocabulary statistics

### `notebooks/testing.ipynb`
Jupyter notebook for:
- Model architecture testing
- Shape verification
- Component testing
