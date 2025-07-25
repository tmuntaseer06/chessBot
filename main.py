import pygame
import chess
import sys
import chess.engine as cg
from predict import predict_best_move, load_model




WIDTH, HEIGHT = 640,640
SQ_SIZE = WIDTH//8


# engine = cg.SimpleEngine.popen_uci("stockfish\stockfish-windows-x86-64-avx2.exe")
# Load model and move map
model_path = "model/sigma2.pth"  # Update if needed
from functions import load_processed_data,build_move_vocab,get_stockfish_evaluation,board_to_tensor

positions, evaluations, moves, game_ids = load_processed_data("processed_data/chess_data.pkl")

move_to_int = build_move_vocab(moves)
model = load_model(model_path, num_classes=len(move_to_int))

pygame.init()
screen = pygame.display.set_mode((WIDTH,HEIGHT))

def get_row(square): #gets row for black and white
    rank = chess.square_rank(square)
    return 7 - rank if player_color == 'white' else rank


pieces_images = {}
def load_images(): #loads images from file path
    pieces = ['P', 'R', 'N', 'B', 'Q', 'K']
    colors = ['w', 'b']
    for color in colors:
        for piece in pieces:
            name = color +'-'+ piece
            img = pygame.image.load(f"pieces/{name}.png")
            pieces_images[name] = pygame.transform.scale(img, (SQ_SIZE, SQ_SIZE))
    print('images loaded!')

def draw_board(screen):#loads screen
    colors = [pygame.Color("white"), pygame.Color((118,150,86))]
    #font = pygame.font.SysFont("Arial", 16)
    rows = range(8) 
    cols = range(8) 

    for r in rows:
        for c in cols:
            color = colors[(r + c) % 2]
            rect = pygame.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE) 
            pygame.draw.rect(screen, color, rect)


    


def draw_pieces(screen, board):# draw the pieces on the board
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            col = chess.square_file(square)
            row = get_row(square)

            if player_color == 'black':
                row = 7 - row
            color_char = 'w' if piece.color == chess.WHITE else 'b'
            piece_img = pieces_images[color_char +'-'+ piece.symbol().upper()]
            if player_color=='black':
                piece_img = pygame.transform.rotate(piece_img, 180)
            screen.blit(piece_img, pygame.Rect(col*SQ_SIZE,row*SQ_SIZE, SQ_SIZE, SQ_SIZE))

def highlight_squares(screen, board, selected_square):
    if selected_square is None:
        return

    piece = board.piece_at(selected_square)
    # Get row and col from square index 
    col = chess.square_file(selected_square)
    row = get_row(selected_square)
    if player_color == 'black':
        row = 7 - row
    

    if (player_color == 'white' and board.turn != chess.WHITE) or \
       (player_color == 'black' and board.turn != chess.BLACK) or \
       (player_color == 'white' and piece.color != chess.WHITE) or \
       (player_color == 'black' and piece.color != chess.BLACK):
        return

    # Highlight selected square
    highlight_color = pygame.Color(100, 52, 235, 100)  # purple overlay
    surface = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
    surface.fill(highlight_color)
    screen.blit(surface, (col * SQ_SIZE, row * SQ_SIZE))

    # Highlight legal moves
    for move in board.legal_moves:
        if move.from_square == selected_square:
            dest_col = chess.square_file(move.to_square)
            dest_row = get_row(move.to_square) if player_color=='white' else 7 - get_row(move.to_square)
            surface.fill(pygame.Color(255, 255, 0, 100))  # Yellow overlay
            screen.blit(surface, (dest_col * SQ_SIZE, dest_row * SQ_SIZE))

def draw_start_menu(screen):
    screen.fill(pygame.Color("darkslategray"))
    
    title_font = pygame.font.SysFont("Arial", 36)
    button_font = pygame.font.SysFont("Arial", 24)

    title = title_font.render("Choose Your Side", True, pygame.Color("white"))
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 100))

    # White Button
    white_btn = pygame.Rect(WIDTH // 4 - 75, 250, 150, 50)
    pygame.draw.rect(screen, pygame.Color("white"), white_btn)
    white_label = button_font.render("Play as White", True, pygame.Color("black"))
    screen.blit(white_label, (white_btn.x + 15, white_btn.y + 10))

    # Black Button
    black_btn = pygame.Rect(WIDTH * 3 // 4 - 75, 250, 150, 50)
    pygame.draw.rect(screen, pygame.Color("black"), black_btn)
    black_label = button_font.render("Play as Black", True, pygame.Color("white"))
    screen.blit(black_label, (black_btn.x + 15, black_btn.y + 10))

    pygame.display.flip()
    return white_btn, black_btn



def main():
    global player_color
    board = chess.Board()
    selected_square = None
    load_images()
    

    player_color = None  # 'white' or 'black'
    running = True
    font_big = pygame.font.SysFont("Arial", 36)
    font_small = pygame.font.SysFont("Arial", 24)
    
    while running:
        screen.fill(pygame.Color("darkslategray"))

        if player_color is None:
            # Draw side selection buttons
            title = font_big.render("Choose Your Side", True, pygame.Color("white"))
            screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 100))

            white_btn = pygame.Rect(WIDTH // 4 - 75, 250, 150, 50)
            pygame.draw.rect(screen, pygame.Color("white"), white_btn)
            screen.blit(font_small.render("Play as White", True, pygame.Color("black")),
                        (white_btn.x + 15, white_btn.y + 10))

            black_btn = pygame.Rect(WIDTH * 3 // 4 - 75, 250, 150, 50)
            pygame.draw.rect(screen, pygame.Color("black"), black_btn)
            screen.blit(font_small.render("Play as Black", True, pygame.Color("white")),
                        (black_btn.x + 15, black_btn.y + 10))
            
            pygame.display.flip()
            


            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    print("Menu exited")
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    if white_btn.collidepoint(x, y):
                        player_color = 'white'
                        print(f"You picked {player_color}")
                    elif black_btn.collidepoint(x, y):
                        player_color = 'black'
                        print(f"You picked {player_color}")
                            # Let Stockfish play the first move
                        # if board.turn == chess.WHITE:
                        #     result = engine.play(board, chess.engine.Limit(time=0.1))
                        evaluation = get_stockfish_evaluation(board)
                        board_tensor = board_to_tensor(board)
                        idx_to_move = {idx: move for move, idx in move_to_int.items()}
                        result = predict_best_move( model,evaluation,board_tensor, idx_to_move,board)
                        board.push(result)

        else:
            board_surface = pygame.Surface((WIDTH, HEIGHT))
            draw_board(board_surface)
            highlight_squares(board_surface, board, selected_square)
            draw_pieces(board_surface, board)

            # Rotate for black player
            if player_color == 'black':
                board_surface = pygame.transform.rotate(board_surface, 180)

            screen.blit(board_surface, (0, 0))

           
            pygame.display.flip()
        
        for event in pygame.event.get():

            if event.type == pygame.MOUSEBUTTONDOWN:
                if (player_color == 'white' and board.turn == chess.WHITE) or \
                (player_color == 'black' and board.turn == chess.BLACK):

                    x, y = pygame.mouse.get_pos()
                    col = (7 - (x // SQ_SIZE)) if player_color == 'black' else (x // SQ_SIZE)
                    row = (7 - (y // SQ_SIZE)) if player_color == 'white' else (y // SQ_SIZE)
                    square = chess.square(col, row)
                    print(f"clicked {square}")

                    piece = board.piece_at(square)

                    if selected_square is None:
                        # Select only your own pieces
                        if piece and ((player_color == 'white' and piece.color == chess.WHITE) or
                                    (player_color == 'black' and piece.color == chess.BLACK)):
                            selected_square = square
                    else:
                        if square == selected_square:
                            selected_square = None
                        else:
                            move = chess.Move(selected_square, square)
                            if move in board.legal_moves:
                                board.push(move)
                                selected_square = None
                                # Let model respond
                                if not board.is_game_over():
                                    if (player_color == 'white' and board.turn == chess.BLACK) or \
                                    (player_color == 'black' and board.turn == chess.WHITE):
                                        # result = engine.play(board, chess.engine.Limit(time=0.1))
                                        evaluation = get_stockfish_evaluation(board)
                                        board_tensor = board_to_tensor(board)
                                        idx_to_move = {idx: move for move, idx in move_to_int.items()}
                                        result = predict_best_move( model,evaluation,board_tensor, idx_to_move,board)
                                        if result in board.legal_moves:
                                            print(result)
                                            board.push(result)
                                        else:
                                            print(result)
                                            print("Move is illegal")
                                
                    if board.is_game_over() or board.is_checkmate():
                        print("Game over:", board.result())
                        running = False
                        break

            if event.type == pygame.QUIT:
                print("Game exited")
                running =False
                pygame.quit()
                sys.exit()
             
    pygame.quit()
    sys.exit()     
                

main()

