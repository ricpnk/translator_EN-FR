import torch
import pandas as pd
from collections import Counter
from torch.utils.data import DataLoader

from src.Vocab import Vocab
from src.Translation_Data import Translation_Data, collate


# Hyperparameters
BATCH_SIZE = 32

def main():
    #todo check if apple gpu is available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device: ", device)

    # todo read in the data
    data = pd.read_csv("data/rec05_small_en_fr.csv")

    # todo initialize counters for the vocabs
    en_counter = Counter()
    fr_counter = Counter()
    for sentence in data["EN"]:
        en_counter.update(sentence.split())
    for sentence in data["FR"]:
        fr_counter.update(sentence.split())

    # todo create vocabs for input and output
    vocabulary_en = Vocab()
    vocabulary_fr = Vocab()
    vocabulary_en.build(en_counter)
    vocabulary_fr.build(fr_counter)

    # todo initiate translator dataset
    translator_data = Translation_Data(data, vocabulary_en, vocabulary_fr)

    train_loader = DataLoader(
        translator_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate(
            batch, special_idx=vocabulary_en.word2idx["<pad>"]
        ),
    )


if __name__ == "__main__":
    main()
