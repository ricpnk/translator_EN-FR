from collections import Counter


class Vocab:

    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.specials = ["<pad>", "<sos>", "<eos>", "<unk>"]

    def __len__(self):
        return len(self.idx2word)

    def build(self, counter: Counter):
        for spec in self.specials:
            self.add_token(spec)

        sorted_tokens = counter.most_common()
        for token, freq in sorted_tokens:
            if token not in self.word2idx:
                self.add_token(token)

    def add_token(self, token):
        self.idx2word.append(token)
        self.word2idx[token] = len(self.idx2word) - 1

    def sentence_to_idx(self, sentence):
        sentence_idx = []
        sentence_idx.append(self.word2idx["<sos>"])
        for token in sentence.split():
            sentence_idx.append(self.word2idx.get(token, self.word2idx["<unk>"]))
        sentence_idx.append(self.word2idx["<eos>"])
        return sentence_idx

    def idx_to_sentence(self, indices):
        words = []
        for idx in indices:
            word = self.idx2word[idx]
            if word in self.specials:
                continue
            words.append(word)

        return words
