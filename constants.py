# constants.py
from __future__ import annotations

BOARD_H, BOARD_W = 10, 9

# 统一用英文代号（你后面可映射到 '车马炮相士兵将/帅'）
# R: rook, N: knight, C: cannon, B: bishop/elephant, A: advisor, P: pawn, K: king
PIECE_KINDS = ["R", "N", "C", "B", "A", "P", "K"]

# 每方：15暗子 + 1明摆K
START_COUNTS = {
    "R": 2,
    "N": 2,
    "C": 2,
    "B": 2,
    "A": 2,
    "P": 5,
    "K": 1,
}

RED, BLACK = 0, 1
MAX_PLIES = 300  # 防止无限对弈，达到就判和
NO_CAPTURE_DRAW_PLIES = 30      # 连续无吃子判和

def other(player: int) -> int:
    return BLACK if player == RED else RED
