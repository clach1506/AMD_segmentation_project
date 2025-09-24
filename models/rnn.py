# models/rnn.py
import torch
import torch.nn as nn

class FlexibleGRU(nn.Module):
    def __init__(
        self,
        input_size=1,          # 1 for pixel, 9 for patch 3x3
        hidden_size=64,
        num_layers=1,
        bidirectional=True,
        dropout=0.2
    ):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0  # dropout only if num_layers > 1
        )

        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(
            hidden_size * (2 if bidirectional else 1), 1
        )  # Output: 1 logit per timestep

    def forward(self, x):
        """
        x: tensor of shape (B, T, P) — P=1 for pixel, P=9 for 3x3 patch
        returns: logits of shape (B, T)
        """
        rnn_out, _ = self.rnn(x)             # (B, T, hidden)
        rnn_out = self.dropout(rnn_out)      # optional regularization
        logits = self.output_layer(rnn_out)  # (B, T, 1)
        return logits.squeeze(-1)            # (B, T)
  
