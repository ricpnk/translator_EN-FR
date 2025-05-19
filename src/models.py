import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_dim,
            bidirectional=False,
            batch_first=True,
            dropout=dropout if n_dim > 1 else 0.0
        )

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        outputs, hidden = self.gru(embedded)
        return outputs, hidden


class Decoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_dim,
            batch_first=True,
            dropout=dropout if n_dim > 1 else 0.0
        )
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input, hidden):
        embedded = self.dropout(self.embedding(input.unsqueeze(1)))
        output, hidden = self.gru(embedded, hidden)
        prediction = self.output_layer(output.squeeze(1))
        return prediction, hidden



class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device


    def forward(self, input, output, teacher_forcing_rate=0.5):
        batch_size, input_len = input.shape
        _, output_len = output.shape
        vocab_size = self.decoder.output_layer.out_features
        
        # get the hidden states per position and the last hidden layer
        encoder_outputs, hidden = self.encoder(input)

        # create the output tensor
        outputs = torch.zeros(batch_size, output_len, vocab_size, device=self.device)
        # input the <sos> token
        input_token = output[:, 0]

        for t in range(1, output_len):
            # get the prediction and safe hidden for next step
            prediction, hidden = self.decoder(input_token, hidden)
            # safe the prediction in outputs
            outputs[:, t, :] = prediction

            # randomly use next token or prediction max
            use_teacher = torch.rand(1).item() < teacher_forcing_rate
            top1 = prediction.argmax(1)
            input_token = output[:, t] if use_teacher else top1

        return outputs


        
