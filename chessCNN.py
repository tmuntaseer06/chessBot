import torch.nn as nn
import torch

class ChessModel(nn.Module):
    def __init__(self, num_classes):
        super(ChessModel, self).__init__()
        #CNN
        self.conv1 = nn.Conv2d(13, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.board_fc = nn.Linear(8 * 8 * 128, 256)

        # MLP for evaluation scalar
        self.eval_fc1 = nn.Linear(1, 32)
        self.eval_fc2 = nn.Linear(32, 64)

        # Combined layers
        self.combined_fc1 = nn.Linear(256 + 64, 128)
        self.output = nn.Linear(128, num_classes)

    def forward(self, board_tensor, evaluation_scalar):
        # Board path
        x = self.relu(self.conv1(board_tensor))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.board_fc(x))

        # Evaluation path
        e = self.relu(self.eval_fc1(evaluation_scalar))
        e = self.relu(self.eval_fc2(e))

        # Combine
        combined = torch.cat((x, e), dim=1)
        combined = self.relu(self.combined_fc1(combined))
        out = self.output(combined)
        return out