from torch import nn
import torch


class GRUNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 48,
        output_len: int = 288,
        output_dim: int = 1,
        dropout: int = 0.01,
        lstm_layer: int = 2,
    ):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.output_len = output_len
        self.output_dim = output_dim
        self.dropout = torch.nn.Dropout(dropout)
        self.lstm = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=lstm_layer, batch_first=True)
        self.projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        to_pred = torch.zeros([x.shape[0], self.output_len, x.shape[2]]).to(self.device)
        x = torch.concat((x, to_pred), 1)
        dec, _ = self.lstm(x)
        pred = self.projection(self.dropout(dec))
        pred = pred[:, -self.output_len:, -self.output_dim:]
        return pred

    def get_output_len(self):
        return self.output_len
