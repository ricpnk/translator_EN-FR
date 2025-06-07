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

    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_dim, dropout, pad_idx, attention_dim=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        
        # Add a layer to combine context and embedded input
        self.combine_context = nn.Linear(embedding_dim + hidden_dim, embedding_dim)
        
        self.gru = nn.GRU(
            input_size=embedding_dim,  # Changed back to embedding_dim
            hidden_size=hidden_dim,
            num_layers=n_dim,
            batch_first=True,
            dropout=dropout if n_dim > 1 else 0.0
        )
        self.attention = Attention(hidden_dim, attention_dim)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input, hidden, encoder_outputs, src_mask):
        # Get attention context
        dec_hidden_top = hidden[-1]
        context, attention_weights = self.attention(dec_hidden_top, encoder_outputs, mask=src_mask)
        
        # Embed the input
        embedded = self.dropout(self.embedding(input.unsqueeze(1)))
        
        # Combine context with embedded input
        combined = torch.cat((embedded, context.unsqueeze(1)), dim=2)
        combined = self.combine_context(combined)
        
        # GRU forward pass
        output, hidden = self.gru(combined, hidden)
        
        # Final prediction
        prediction = self.output_layer(output.squeeze(1))
        
        return prediction, hidden, attention_weights
    

class Attention(nn.Module):
    def __init__(self, hidden_dim, attention_dim=None):
        super().__init__()
        if attention_dim is None:
            self.attention_dim = hidden_dim
        else:
            self.attention_dim = attention_dim

        # Projection layers
        self.Weights_encoder = nn.Linear(hidden_dim, self.attention_dim)
        self.Weights_decoder = nn.Linear(hidden_dim, self.attention_dim)
        self.attention_score = nn.Linear(self.attention_dim, 1, bias=False)

    def forward(self, dec_hidden, enc_outputs, mask=None):
        # Project encoder outputs and decoder hidden state
        enc_projected = self.Weights_encoder(enc_outputs)  # [batch_size, src_len, attention_dim]
        dec_projected = self.Weights_decoder(dec_hidden).unsqueeze(1)  # [batch_size, 1, attention_dim]
        
        # Expand decoder projection
        dec_projected = dec_projected.expand(-1, enc_projected.size(1), -1)
        
        # Calculate energy scores with scaling
        energy = torch.tanh(enc_projected + dec_projected)
        energy = self.attention_score(energy).squeeze(-1)
        
        if mask is not None:
            energy = energy.masked_fill(~mask, float("-inf"))
        
        attention_weights = torch.softmax(energy, dim=1)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), enc_outputs).squeeze(1)
        
        return context_vector, attention_weights



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


        PAD_IDX = self.decoder.embedding.padding_idx
        input_mask = (input != PAD_IDX)
        

        # input the <sos> token
        input_token = output[:, 0]

        for t in range(1, output_len):
            # get the prediction and safe hidden for next step
            prediction, hidden, attention_weights = self.decoder(input_token, hidden, encoder_outputs, input_mask)

            # safe the prediction in outputs
            outputs[:, t, :] = prediction

            # randomly use next token or prediction max
            use_teacher = torch.rand(1).item() < teacher_forcing_rate
            top1 = prediction.argmax(1)
            input_token = output[:, t] if use_teacher else top1

        return outputs


