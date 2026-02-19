# actions.py (move-only, 2550 actions)
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

BOARD_H, BOARD_W = 10, 9

# ---- move label space (2550) ----
HORSE_DELTAS = [(-2,-1), (-2, 1), (-1,-2), (-1, 2),
                ( 1,-2), ( 1, 2), ( 2,-1), ( 2, 1)]
DIAG1_DELTAS = [(-1,-1), (-1, 1), ( 1,-1), ( 1, 1)]
DIAG2_DELTAS = [(-2,-2), (-2, 2), ( 2,-2), ( 2, 2)]

def in_bounds(y: int, x: int) -> bool:
    return 0 <= y < BOARD_H and 0 <= x < BOARD_W

def move_str(y1: int, x1: int, y2: int, x2: int) -> str:
    return f"{y1}{x1}{y2}{x2}"

def build_move_labels_2550() -> List[str]:
    moves: List[str] = []

    # A) straight lines: 1530
    for y in range(BOARD_H):
        for x in range(BOARD_W):
            for ty in range(BOARD_H):
                if ty != y:
                    moves.append(move_str(y, x, ty, x))
            for tx in range(BOARD_W):
                if tx != x:
                    moves.append(move_str(y, x, y, tx))

    # B) knight: 508
    for y in range(BOARD_H):
        for x in range(BOARD_W):
            for dy, dx in HORSE_DELTAS:
                ty, tx = y + dy, x + dx
                if in_bounds(ty, tx):
                    moves.append(move_str(y, x, ty, tx))

    # C) diag 1-step anywhere: 288
    for y in range(BOARD_H):
        for x in range(BOARD_W):
            for dy, dx in DIAG1_DELTAS:
                ty, tx = y + dy, x + dx
                if in_bounds(ty, tx):
                    moves.append(move_str(y, x, ty, tx))

    # D) diag 2-step anywhere: 224
    for y in range(BOARD_H):
        for x in range(BOARD_W):
            for dy, dx in DIAG2_DELTAS:
                ty, tx = y + dy, x + dx
                if in_bounds(ty, tx):
                    moves.append(move_str(y, x, ty, tx))

    uniq = list(dict.fromkeys(moves))
    if len(uniq) != 2550:
        raise RuntimeError(f"expected 2550 move labels, got {len(uniq)}")
    return uniq

MOVE_LABELS: List[str] = build_move_labels_2550()
MOVE2ID: Dict[str, int] = {m: i for i, m in enumerate(MOVE_LABELS)}
ID2MOVE: List[str] = MOVE_LABELS

# ---- action space = move-only ----
ACTION_SIZE = len(MOVE_LABELS)  # 2550

@dataclass(frozen=True)
class DecodedAction:
    kind: str  # always "move"
    move: Tuple[int, int, int, int]

def encode_move(y1: int, x1: int, y2: int, x2: int) -> Optional[int]:
    return MOVE2ID.get(move_str(y1, x1, y2, x2))  # 0..2549

def decode_action(aid: int) -> DecodedAction:
    s = ID2MOVE[aid]
    y1, x1, y2, x2 = int(s[0]), int(s[1]), int(s[2]), int(s[3])
    return DecodedAction(kind="move", move=(y1, x1, y2, x2))

# 可选：数据增强用的左右翻转映射（move-only）
def flip_action_id(aid: int) -> int:
    s = ID2MOVE[aid]
    y1, x1, y2, x2 = int(s[0]), int(s[1]), int(s[2]), int(s[3])
    mid = encode_move(y1, 8 - x1, y2, 8 - x2)
    if mid is None:
        raise RuntimeError("flip mapping failed (should not happen)")
    return mid
