from torch.utils.data import Dataset
import torch

class ChessMoveDataset(Dataset):
    def __init__(self, positions, evaluations, moves, move_to_idx):
        self.positions = torch.tensor(positions, dtype=torch.float32)  # (N, 13, 8, 8)
        self.evaluations = torch.tensor(evaluations, dtype=torch.float32).unsqueeze(1)  # (N, 1)
        self.moves = [move_to_idx.get(m, move_to_idx["<UNK>"]) for m in moves]
        self.moves = torch.tensor(self.moves, dtype=torch.long)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return self.positions[idx], self.evaluations[idx], self.moves[idx]
