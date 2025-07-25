
import chess.engine
import chess.pgn
import numpy as np
import chess
import pickle
import json
import os



engine = chess.engine.SimpleEngine.popen_uci('stockfish/stockfish-windows-x86-64-avx2.exe')
def board_to_tensor(board):
    """Convert chess board to 13-channel tensor"""
    tensor = np.zeros((13, 8, 8))
    piece_map = board.piece_map()

    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color = 0 if piece.color else 6
        tensor[piece_type + piece_color, row, col] = 1

    for move in board.legal_moves:
        to_square = move.to_square
        row, col = divmod(to_square, 8)
        tensor[12, row, col] = 1

    return tensor

def get_stockfish_evaluation(board, depth=12):
    """Get Stockfish evaluation for a position"""
    
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info["score"]
        
        if score.is_mate():
            # Convert mate in N to large number
            mate_value = score.white().mate()
            
            return 10000 if mate_value > 0 else -10000
        else:
            # Return centipawn evaluation
            
            return score.white().score(mate_score=10000) / 100.0  # Convert to pawn units
    except:
        return 0.0  # Return neutral if evaluation fails


def extract_all_games_data(pgn_path,stockfish_path,max_games=10,depth=10,save_every= 1000):
    """
    Extract all moves and evaluations from PGN file
    """
    
    # Storage for data
    positions = []      # Board tensors
    evaluations = []    # Stockfish evaluations
    moves = []         # Actual moves made (UCI format)
    game_ids = []      # Track which game each position comes from
    
    total_positions = 0
    games_processed = 0

    with open(pgn_path) as f:
        game_count = 0

        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            if max_games and game_count >= max_games:
                break

            board = game.board()
            move_count = 0

            print(f"Processing game {game_count + 1}...")

            for move in game.mainline_moves():
                position_tensor = board_to_tensor(board)

                evaluation = get_stockfish_evaluation(board,depth)

                positions.append(position_tensor)
                evaluations.append(evaluation)
                moves.append(move.uci())
                game_ids.append(game_count)

                board.push(move)
                move_count += 1
                total_positions += 1

            games_processed += 1


            # if games_processed % save_every == 0:
            #         print(f"Saving intermediate data after {games_processed} games...")
            #         save_intermediate_data(positions, evaluations, moves, game_ids, 
            #                              f"chess_data_intermediate_{games_processed}.pkl")
                
            game_count += 1
                
            if game_count % 100 == 0:
                print(f"Processed {game_count} games, {total_positions} positions")

    
    print(f"Extraction complete!")
    print(f"Total games processed: {games_processed}")
    print(f"Total positions extracted: {total_positions}")
    print(f"Average positions per game: {total_positions/games_processed:.1f}")
    
    return np.array(positions), np.array(evaluations), moves, game_ids



def save_intermediate_data(positions, evaluations, moves, game_ids, filename):
    """Save intermediate data to avoid losing progress"""
    data = {
        'positions': np.array(positions),
        'evaluations': np.array(evaluations),
        'moves': moves,
        'game_ids': game_ids,
        'count': len(positions)
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved {len(positions)} positions to {filename}")

def save_final_data(positions, evaluations, moves, game_ids, base_filename="chess_data"):
    """Save final processed data in multiple formats"""
    
    print("Saving final data...")
    
    # Create data directory
    os.makedirs("processed_data", exist_ok=True)
    
    # Save as pickle (recommended for training)
    pickle_path = f"processed_data/{base_filename}.pkl"
    data = {
        'positions': positions,
        'evaluations': evaluations,
        'moves': moves,
        'game_ids': game_ids,
        'shape_info': {
            'num_positions': len(positions),
            'position_shape': positions[0].shape,
            'eval_range': (evaluations.min(), evaluations.max())
        }
    }
    
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved complete dataset to {pickle_path}")
    
    # Save numpy arrays separately (faster loading)
    np.save(f"processed_data/{base_filename}_positions.npy", positions)
    np.save(f"processed_data/{base_filename}_evaluations.npy", evaluations)
    
    # Save metadata as JSON
    metadata = {
        'total_positions': len(positions),
        'total_games': len(set(game_ids)),
        'position_shape': positions[0].shape,
        'evaluation_stats': {
            'min': float(evaluations.min()),
            'max': float(evaluations.max()),
            'mean': float(evaluations.mean()),
            'std': float(evaluations.std())
        }
    }
    
    with open(f"processed_data/{base_filename}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Data extraction complete!")
    print(f"Files saved in processed_data/ directory")
    return pickle_path

def load_processed_data(pickle_path):
    """Load the processed data for training"""
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    engine.quit()
    return data['positions'], data['evaluations'], data['moves'], data['game_ids']

def analyze_extracted_data(pickle_path):
    """Analyze the extracted data"""
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    positions = data['positions']
    evaluations = data['evaluations']
    
    print(f"Dataset Analysis:")
    print(f"Total positions: {len(positions):,}")
    print(f"Position tensor shape: {positions[0].shape}")
    print(f"Evaluation range: {evaluations.min():.2f} to {evaluations.max():.2f}")
    print(f"Evaluation mean: {evaluations.mean():.2f}")
    print(f"Evaluation std: {evaluations.std():.2f}")
    print(f"Total games: {len(set(data['game_ids']))}")

    
from collections import Counter

def build_move_vocab(moves):
    unique_moves = sorted(set(moves))
    move_to_idx = {move: idx for idx, move in enumerate(unique_moves)}
    move_to_idx["<UNK>"] = len(move_to_idx)  # Add unknown token
    return move_to_idx


if __name__ == "__main__":
    # Configuration
    PGN_PATH = "data/lichess_db_standard_rated_2015-08.pgn" # be sure to change this
    STOCKFISH_PATH = "stockfish/stockfish-windows-x86-64-avx2.exe"  # Update this path
    MAX_GAMES = 10000  # Set to None for all games
    DEPTH = 10  # Stockfish analysis depth
    
    print("Starting chess data extraction...")
    print("This may take a long time depending on the number of games!")
    
    # Extract data
    positions, evaluations, moves, game_ids = extract_all_games_data(
        pgn_path=PGN_PATH,
        stockfish_path=STOCKFISH_PATH,
        max_games=MAX_GAMES,
        depth=DEPTH,
        save_every=500  # Save intermediate results every 500 games
    )
    
    # Save final data
    data_path = save_final_data(positions, evaluations, moves, game_ids)
    
    # Analyze the results
    analyze_extracted_data(data_path)
    
    print('engine quitted')
