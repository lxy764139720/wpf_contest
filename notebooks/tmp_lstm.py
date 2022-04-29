import torch


class LSTMModel(torch.nn.Module):
    def __init__(self,
                 input_dim: int = 1,
                 hidden_dim: int = 16,
                 output_dim: int = 1,
                 dropout: int = 0.2,
                 lstm_layer: int = 2):
        super().__init__()
        self.output_dim = output_dim
        self.dropout = torch.nn.Dropout(dropout)
        self.lstm = torch.nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=lstm_layer, batch_first=True,
                                  bidirectional=True)
        self.projection = torch.nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        dec, _ = self.lstm(x)
        pred = self.projection(self.dropout(dec[:, -1, :]))
        return torch.clamp(pred, min=0, max=52)
