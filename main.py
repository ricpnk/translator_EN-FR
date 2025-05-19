import torch
import torch.nn as nn
import pandas as pd
from collections import Counter
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.Vocab import Vocab
from src.Translation_Data import Translation_Data, collate
from src.models import Encoder, Decoder, Seq2Seq


# Hyperparameters
BATCH_SIZE = 32
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
N_DIM = 1
MAX_LEN = 15
LEARNRATE = 0.001


def main():
    # todo check if apple gpu is available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device: ", device)

    # todo read in the data and train-test split
    data = pd.read_csv("data/rec05_small_en_fr.csv")
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

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
    train_translator_data = Translation_Data(train_data, vocabulary_en, vocabulary_fr)
    test_translator_data = Translation_Data(test_data, vocabulary_en, vocabulary_fr)

    train_loader = DataLoader(
        train_translator_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate(
            batch, special_idx=vocabulary_en.word2idx["<pad>"]
        ),
    )
    test_loader = DataLoader(
        test_translator_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: collate(
            batch, special_idx=vocabulary_en.word2idx["<pad>"]
        ),
    )

    # todo initiate the Objects for models
    encoder = Encoder(
        vocab_size=len(vocabulary_en),
        embedding_dim=EMBEDDING_DIM
        hidden_dim=HIDDEN_DIM,
        n_dim=N_DIM
        dropout=0.1
    ).to(device)

    decoder = Decoder(
        vocab_size=len(vocabulary_en),
        embedding_dim=EMBEDDING_DIM
        hidden_dim=HIDDEN_DIM,
        n_dim=N_DIM
        dropout=0.1
    ).to(device)

    model = Seq2Seq(encoder, decoder, device).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocabulary_en.word2idx["<pad>"])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNRATE)




if __name__ == "__main__":
    main()
