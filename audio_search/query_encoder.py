import torch
import torch.nn as nn


class QueryEncoder(nn.Module):

    def __init__(self):
        super(QueryEncoder, self).__init__()
        self.cnn1 = nn.Conv1d(13, 128, 6)
        self.cnn2 = nn.Conv1d(128, 128, 6)
        self.cnn3 = nn.Conv1d(128, 128, 6)
        self.cnn4 = nn.Conv1d(128, 128, 6)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.tanh(self.cnn1(x))
        x = torch.tanh(self.cnn2(x))
        x = torch.tanh(self.cnn3(x))
        x = self.cnn4(x)
        x = x.transpose(1, 2)
        return x
