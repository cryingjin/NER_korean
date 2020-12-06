import torch
import torch.nn as nn
from torchcrf import CRF


class RNN_CRF(nn.Module):
    def __init__(self,
                 input_size, embedding_size,
                 hidden_size, output_size,
                 pretrained_weight=None, dropout=0.5):
        super(RNN_CRF, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(pretrained_weight)
        self.dropout = nn.Dropout(dropout)

        self.rnn = nn.GRU(
            embedding_size,
            hidden_size,
            batch_first=True,
            bidirectional=True
        )

        # CRF layer
        self.crf = CRF(output_size, batch_first=True)

        # (batch_size, seq_len, hidden_size * 2) -> (batch_size, seq_len, output_size)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, inputs, labels=None):
        # (batch_size, seq_len) -> (batch_size, seq_len, embedding_size)
        inputs = self.embedding(inputs)
        outputs, hidden = self.rnn(inputs)

        # (batch_size, seq_len, hidden_size * 2)
        outputs = self.dropout(outputs)

        # (batch_size, seq_len, hidden_size * 2) -> (batch_size, seq_len, output_size)
        logits = self.fc(outputs)

        if labels is not None:
            log_likelihood = self.crf(
                emissions=logits,
                tags=labels,
                reduction="mean"
            )
            loss = log_likelihood * -1.0
            return loss
        else:
            output = self.crf.decode(emissions=logits)
            return output
