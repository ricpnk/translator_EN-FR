import torch
import pandas as pd
from src.Vocab import Vocab
from collections import Counter

def main():
    data = pd.read_csv("data/rec05_small_en_fr.csv")


    en_counter = Counter()
    for sentence in data["EN"]:
        en_counter.update(sentence.split())

    vocabulary = Vocab()
    vocabulary.build(en_counter)

    print(len(vocabulary))




if __name__ == "__main__":
    main()