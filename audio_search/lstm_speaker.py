import torch.nn as nn

class LSTMSpeakerIdentifier(nn.Module):

    def __init__(self):
        super(LSTMSpeakerIdentifier, self).__init__()
        self.hidden_dim = 256
        self.layer_dim = 2
        self.lstm = nn.LSTM(input_size=13, num_layers=1, hidden_size=256, batch_first=True, bidirectional=True)
        self.cnn = nn.Conv1d(512, 1, 3)

    def forward(self, x):
        out = self.lstm(x)[0].unsqueeze(-1)
        return self.cnn(out).transpose(1, 2).squeeze(1)
