import torch
import pandas as pd
from src.Vocab import Vocab
from collections import Counter

def main():
    #todo read in the data
    data = pd.read_csv("data/rec05_small_en_fr.csv")

    #todo initialize counters for the vocabs
    en_counter = Counter()
    fr_counter = Counter()
    for sentence in data["EN"]:
        en_counter.update(sentence.split())
    for sentence in data["FR"]:
        fr_counter.update(sentence.split())

    #todo create vocabs for input and output
    vocabulary_en = Vocab()
    vocabulary_fr = Vocab()
    vocabulary_en.build(en_counter)
    vocabulary_fr.build(fr_counter)

    


        




if __name__ == "__main__":
    main()