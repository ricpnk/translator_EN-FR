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
        # Embed input and apply dropout
        embedded = self.dropout(self.embedding(input))
        outputs, hidden = self.gru(embedded)
        return outputs, hidden


class Decoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_dim, dropout, pad_idx, attention_dim=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        
        # Layer to combine context vector with embedded input
        self.combine_context = nn.Linear(embedding_dim + hidden_dim, embedding_dim)
        
        self.gru = nn.GRU(
            input_size=embedding_dim,
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
        
        # Process input and combine with context
        embedded = self.dropout(self.embedding(input.unsqueeze(1)))
        combined = torch.cat((embedded, context.unsqueeze(1)), dim=2)
        combined = self.combine_context(combined)
        
        # Generate output
        output, hidden = self.gru(combined, hidden)
        prediction = self.output_layer(output.squeeze(1))
        
        return prediction, hidden, attention_weights


class Attention(nn.Module):
    
    def __init__(self, hidden_dim, attention_dim=None):
        super().__init__()
        self.attention_dim = attention_dim if attention_dim is not None else hidden_dim

        # Projection layers
        self.encoder_projection = nn.Linear(hidden_dim, self.attention_dim)
        self.decoder_projection = nn.Linear(hidden_dim, self.attention_dim)
        self.attention_score = nn.Linear(self.attention_dim, 1, bias=False)

    def forward(self, dec_hidden, enc_outputs, mask=None):
        # Project encoder and decoder states
        enc_projected = self.encoder_projection(enc_outputs)
        dec_projected = self.decoder_projection(dec_hidden).unsqueeze(1)
        dec_projected = dec_projected.expand(-1, enc_projected.size(1), -1)
        
        # Calculate attention scores
        energy = torch.tanh(enc_projected + dec_projected)
        energy = self.attention_score(energy).squeeze(-1)
        
        # Apply mask
        if mask is not None:
            energy = energy.masked_fill(~mask, float("-inf"))
        
        # Get attention weights and context
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
        
        # Encode input sequence
        encoder_outputs, hidden = self.encoder(input)
        outputs = torch.zeros(batch_size, output_len, vocab_size, device=self.device)

        # Create mask for padding
        pad_idx = self.decoder.embedding.padding_idx
        input_mask = (input != pad_idx)
        
        # Start with <sos> token
        input_token = output[:, 0]

        # Decode step by step
        for t in range(1, output_len):
            # Get prediction and update hidden state
            prediction, hidden, _ = self.decoder(input_token, hidden, encoder_outputs, input_mask)
            outputs[:, t, :] = prediction

            # Apply teacher forcing
            use_teacher = torch.rand(1).item() < teacher_forcing_rate
            top1 = prediction.argmax(1)
            input_token = output[:, t] if use_teacher else top1

        return outputs


