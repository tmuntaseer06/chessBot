from collections import Counter

def build_move_vocab(moves):
    unique_moves = sorted(set(moves))
    move_to_idx = {move: idx for idx, move in enumerate(unique_moves)}
    move_to_idx["<UNK>"] = len(move_to_idx)  # Add unknown token
    return move_to_idx

import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

def train(model, dataset, epochs=10, batch_size=64, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for boards, evals, labels in loader:
            boards = boards.to(device)
            evals = evals.to(device)
            labels = torch.tensor(labels).to(device)

            optimizer.zero_grad()
            output = model(boards, evals)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")
    return model

def predict_best_move(model, evaluation,board_tensor, idx_to_move):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        sample_board = board_tensor.unsqueeze(0).to(device)
        sample_eval = torch.tensor([[evaluation]]).to(device)
        logits = model(sample_board, sample_eval)
        predicted_index = torch.argmax(logits, dim=1).item()
        predicted_move = idx_to_move[predicted_index]
    return predicted_move

from functions import load_processed_data
from dataset import ChessMoveDataset
from chessCNN import ChessModel
from torch.utils.data import DataLoader

# Load processed data
positions, evaluations, moves, game_ids = load_processed_data("processed_data/chess_data.pkl")

# Build vocabulary
move_to_idx = build_move_vocab(moves)

# Create dataset
dataset = ChessMoveDataset(positions, evaluations, moves, move_to_idx)

# Optional: create DataLoader
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize model
model = ChessModel(num_classes=len(move_to_idx))

# Train
model = train(model, dataset, epochs=10)

torch.save(model.state_dict(), 'model/sigma2.pth')