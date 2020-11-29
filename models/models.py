import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers=2, dropout=0.5):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(
            embedding_size,
            hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, source):
        # source: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(source))
        output, (hidden, cell) = self.lstm(embedded)

        # hidden: [num_layers * 2, batch_size, hidden_size]
        # cell: [num_layers * 2, batch_size, hidden_size]
        return output, hidden, cell


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

        self.attn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, hidden, encoder_outputs, mask):
        # hidden = [num_layers * 2, batch_size, hidden_size]
        # encoder_outputs = [batch_size, seq_len, hidden_size * 2]
        # mask = [batch_size, seq_len]
        batch_size, seq_len, _ = encoder_outputs.size()
        x = self.attn(hidden)
        attn_score = torch.bmm(encoder_outputs, x).squeeze(-1)
        print(attn_score.shape)  # [batch_size, seq_len] 이 되어야

        return F.softmax(attn_score, dim=-1)


class Decoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, num_layers=2, dropout=0.5):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(output_size, embedding_size)
        self.lstm = nn.LSTM(
            embedding_size,
            hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )
        self.attn = Attention(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, source, hidden, cell):
        source = source.unsqueeze(1)
        embedded = self.dropout(self.embedding(source))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        output = self.fc(output.squeeze(1))

        return output, hidden, cell


class Seq2seq(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, device='cpu'):
        super(Seq2seq, self).__init__()
        self.device = device

        self.SOS_TOKEN = 0
        self.EOS_TOKEN = 1
        self.PAD_TOKEN = 2

        self.encoder = Encoder(input_size, embedding_size, hidden_size)
        self.decoder = Decoder(embedding_size, hidden_size, output_size)

    def create_mask(self, source):
        return source != self.PAD_TOKEN

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        ner_output_size = self.ner_decoder.output_size
        intent_output_size = self.intent_decoder.output_size
        batch_size = source.size(0)
        seq_len = target.size(1)

        encoder_outputs, hidden, cell = self.encoder(source)

        initial_hidden = hidden
        initial_cell = cell

        # tensor to store decoder outputs
        ner_outputs = torch.zeros(
            seq_len, batch_size, ner_output_size).to(self.device)
        intent_outputs = torch.zeros(
            batch_size, seq_len, intent_output_size).to(self.device)

        # Create mask
        mask = self.create_mask(source)

        # Initialize
        decoder_input = target[:, 0]

        # Step through the length of the output sequence one token at a time
        # Teacher forcing is used to assist training
        for t in range(1, seq_len):
            output, hidden, cell = self.ner_decoder(
                decoder_input, hidden, cell)
            ner_outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            decoder_input = (target[:, t] if teacher_force else top1)

        # [batch_size, seq_len, output_size]
        return ner_outputs.permute(1, 0, 2)
