# model.py
import torch
import torch.nn as nn

N_MELS = 128
T_FRAMES = 188
NUM_CLASSES = 6

class CNNBiLSTM(nn.Module):
    def __init__(self, lstm_hidden=128):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,1)),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2,1))
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, N_MELS, T_FRAMES)
            z = self.cnn(dummy)
            C, Fp, Tp = z.shape[1], z.shape[2], z.shape[3]
            lstm_in = C * Fp

        self.lstm = nn.LSTM(
            input_size=lstm_in,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2*lstm_hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, x):
        z = self.cnn(x)                 # (B, C, F, T)
        z = z.permute(0, 3, 1, 2)       # (B, T, C, F)
        z = z.flatten(2)                # (B, T, C*F)
        out, _ = self.lstm(z)           # (B, T, 2H)

        # Better than out[:, -1] for emotion:
        out = out.mean(dim=1)           # (B, 2H)

        return self.fc(out)
