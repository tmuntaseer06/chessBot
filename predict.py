import torch
import chess
import numpy as np
from chessCNN import ChessModel  
import random

def load_model(model_path, num_classes):
    model = ChessModel(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

import random
import torch
import chess

import chess

import chess

def is_blunder(board: chess.Board, move: chess.Move, threshold: float = -1.5) -> bool:
    material_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9
    }

    def evaluate_material(b):
        return sum(
            material_values.get(piece.piece_type, 0) * (1 if piece.color == board.turn else -1)
            for piece in b.piece_map().values()
        )

    # Simulate the move
    temp_board = board.copy()
    temp_board.push(move)

    # Check for mate-in-1 by opponent
    for opp_move in temp_board.legal_moves:
        temp_board_next = temp_board.copy()
        temp_board_next.push(opp_move)
        if temp_board_next.is_checkmate():
            return True  # The move allows the opponent to checkmate

    # Check for significant material loss
    before_material = evaluate_material(board)
    after_material = evaluate_material(temp_board)
    material_diff = after_material - before_material

    return material_diff < threshold



def predict_best_move(model, evaluation, board_tensor, idx_to_move, board):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    with torch.no_grad():
        sample_board = torch.tensor(board_tensor, dtype=torch.float32).unsqueeze(0).to(device)
        sample_eval = torch.tensor([[evaluation]], dtype=torch.float32).to(device)

        logits = model(sample_board, sample_eval)
        probs = torch.softmax(logits, dim=1).squeeze()

        # Rank predicted moves by confidence
        ranked_indices = torch.argsort(probs, descending=True)

        for idx in ranked_indices:
            predicted_move_str = idx_to_move[idx.item()]
            try:
                move = chess.Move.from_uci(predicted_move_str)
                if move in board.legal_moves and not is_blunder(board, move):
                    return move
            except Exception:
                continue  # Skip malformed moves

        # return a legal non-blunder move if any
        fallback_moves = [m for m in board.legal_moves if not is_blunder(board, m)]
        if fallback_moves:
            print("Predicted moves were blunders. Choosing safe legal move.")
            return random.choice(fallback_moves)

        # Final fallback: any legal move
        print("All predicted moves blunder. Picking any legal move.")
        return random.choice(list(board.legal_moves))


    