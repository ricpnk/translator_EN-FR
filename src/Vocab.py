from collections import Counter

class Vocab():

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

