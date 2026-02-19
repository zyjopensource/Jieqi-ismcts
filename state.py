# state.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import copy
import random
import numpy as np

from constants import BOARD_H, BOARD_W, RED, BLACK, PIECE_KINDS, START_COUNTS, MAX_PLIES, NO_CAPTURE_DRAW_PLIES, other
from actions import ACTION_SIZE, decode_action, encode_move, in_bounds

KINDS_NO_K = ["R", "N", "C", "B", "A", "P"]  # 暗子池不含K

# --- Jieqi fixed starting squares (excluding K) in 10x9 coords ---
# y: 0(top black side) .. 9(bottom red side)
# x: 0..8

# Black side (traditional Xiangqi positions), excluding black K at (0,4)
BLACK_START_SQUARES = [
    (0,0), (0,1), (0,2), (0,3),        (0,5), (0,6), (0,7), (0,8),  # 8 pieces
    (2,1), (2,7),  # cannons
    (3,0), (3,2), (3,4), (3,6), (3,8),  # pawns
]

# Red side, excluding red K at (9,4)
RED_START_SQUARES = [
    (9,0), (9,1), (9,2), (9,3),        (9,5), (9,6), (9,7), (9,8),
    (7,1), (7,7),
    (6,0), (6,2), (6,4), (6,6), (6,8),
]

# cover_type meaning:
# D=暗车, E=暗马, F=暗相(象), G=暗士, H=暗炮, I=暗兵
# This is determined by the STARTING SQUARE (position-based rule).
def start_square_to_cover_type(color: int, y: int, x: int) -> str:
    """
    Return cover_type for an initial square (excluding K).
    This encodes "dark piece moves like the Xiangqi piece that normally starts here".
    """
    if color == BLACK:
        # back rank y=0
        if y == 0:
            if x in (0, 8): return "D"  # rook
            if x in (1, 7): return "E"  # knight
            if x in (2, 6): return "F"  # bishop/elephant
            if x in (3, 5): return "G"  # advisor/guard
            raise ValueError("black back rank invalid (excluding K)")
        # cannons y=2
        if (y, x) in ((2,1), (2,7)): return "H"
        # pawns y=3
        if y == 3 and x in (0,2,4,6,8): return "I"
        raise ValueError("black start square invalid")

    # RED
    if y == 9:
        if x in (0, 8): return "D"
        if x in (1, 7): return "E"
        if x in (2, 6): return "F"
        if x in (3, 5): return "G"
        raise ValueError("red back rank invalid (excluding K)")
    if (y, x) in ((7,1), (7,7)): return "H"
    if y == 6 and x in (0,2,4,6,8): return "I"
    raise ValueError("red start square invalid")

@dataclass
class Piece:
    color: int              # RED/BLACK
    kind: str               # 当前身份：未翻开时可以先写成 "?"；翻开后为 'R','N','B','A','C','P','K'
    covered: bool           # True=暗子
    cover_type: Optional[str] = None   # 暗子走法类型：'D','E','F','G','H','I'（翻开后可为 None）
    true_kind: Optional[str] = None    # 暗子的真实身份（翻开后 kind=true_kind，true_kind 可留着或清空）


def _empty_counts() -> Dict[str, int]:
    return {k: 0 for k in PIECE_KINDS}

def _copy_counts(c: Dict[str, int]) -> Dict[str, int]:
    return {k: int(v) for k, v in c.items()}

def _multiset_to_list(pool: Dict[str, int]) -> List[str]:
    out = []
    for k, n in pool.items():
        out.extend([k] * n)
    return out

def _list_to_multiset(lst: List[str]) -> Dict[str, int]:
    d = {k: 0 for k in PIECE_KINDS}
    for x in lst:
        d[x] += 1
    return d

def _sub_count(pool: Dict[str, int], kind: str, n: int = 1) -> None:
    pool[kind] -= n
    if pool[kind] < 0:
        raise ValueError(f"Pool underflow for {kind}")

def _sample_without_replacement(rng: random.Random, pool: Dict[str, int], k: int) -> List[str]:
    flat = _multiset_to_list(pool)
    if k > len(flat):
        raise ValueError("Not enough items in pool to sample")
    rng.shuffle(flat)
    chosen = flat[:k]
    # remove chosen from pool
    for t in chosen:
        _sub_count(pool, t, 1)
    return chosen

class JieqiState:
    """
    这是“裁判真状态” +（每方的可见性记账）。
    MCTS/网络只能通过 obs_planes(player) 获取信息态输入。
    """

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.board: List[List[Optional[Piece]]] = [[None for _ in range(BOARD_W)] for _ in range(BOARD_H)]
        self.current_player: int = RED
        self.last_move: Optional[Tuple[int, int, int, int]] = None
        self.plies: int = 0
        self.no_capture_plies = 0

        # ---------- 可见性相关：每方自己的视角信息 ----------
        # 我吃到对方的“已知身份”统计（对我可见）
        self.captured_by: List[Dict[str, int]] = [_empty_counts(), _empty_counts()]  # captured_by[player][kind]

        # 我方被吃“明子”的已知身份统计（对我可见）
        self.lost_known: List[Dict[str, int]] = [_empty_counts(), _empty_counts()]   # lost_known[player][kind]

        # 我方被吃“暗子”的未知身份数量（对我可见：只知道数量）
        self.lost_unknown_count: List[int] = [0, 0]

        # 我方被吃暗子的“真身份事件列表”（裁判知道；我不知道）
        # 每个元素是 kind 字符串；在 determinize_for(victim) 中会被重采样（信息集采样）
        self.lost_unknown_events: List[List[str]] = [[], []]

        self._init_random_position()

    def _init_random_position(self):
        # 1) K 固定（明摆）
        self.board[9][4] = Piece(RED, "K", covered=False, cover_type=None, true_kind="K")
        self.board[0][4] = Piece(BLACK, "K", covered=False, cover_type=None, true_kind="K")
    
        # 2) 为每方准备 15 个真实身份 true_kind（不含K），可随机
        def sample_true_kinds_for_side():
            kinds = []
            for k in KINDS_NO_K:
                kinds.extend([k] * START_COUNTS[k])  # 这应该是每方各自的数量：R2 N2 C2 B2 A2 P5
            self.rng.shuffle(kinds)
            return kinds[:15]
    
        red_true = sample_true_kinds_for_side()
        black_true = sample_true_kinds_for_side()
    
        # 3) 把 15 个暗子放到固定起始格，并绑定 cover_type（由格子决定）
        self.rng.shuffle(RED_START_SQUARES)
        self.rng.shuffle(BLACK_START_SQUARES)
    
        for (y, x), tk in zip(RED_START_SQUARES, red_true):
            ct = start_square_to_cover_type(RED, y, x)
            # 暗子：kind 可以先用 "?" 或 tk 都行；关键是 covered=True + true_kind=tk + cover_type=ct
            self.board[y][x] = Piece(RED, kind="?", covered=True, cover_type=ct, true_kind=tk)
    
        for (y, x), tk in zip(BLACK_START_SQUARES, black_true):
            ct = start_square_to_cover_type(BLACK, y, x)
            self.board[y][x] = Piece(BLACK, kind="?", covered=True, cover_type=ct, true_kind=tk)
    
        # 4) 清理可见性记账（如果你复用旧 state 对象时需要；新建一般不用）
        self.captured_by = [_empty_counts(), _empty_counts()]
        self.lost_known = [_empty_counts(), _empty_counts()]
        self.lost_unknown_count = [0, 0]
        self.lost_unknown_events = [[], []]

    # ---------------------- 观测：构造网络输入平面（含吃子信息） ----------------------
    def obs_planes(self, player: int) -> np.ndarray:
        """
        返回 shape = [C, 10, 9] 的信息态输入。
        C 设计（最小可跑且包含你要求的吃子信息）：
          14: 明子按 (color,type) one-hot（R,N,C,B,A,P,K 共7类 * 2色）
           2: 暗子位置（红暗/黑暗）
           1: last move（from+to 标记在同一平面）
           1: side-to-move（全1或全0）
           7: 我吃到对方各类数量（captured_by[player]）
           7: 我方已知损失各类数量（lost_known[player]）
           1: 我方暗子被吃但未知身份的数量（lost_unknown_count[player]）
        """
        # 平面索引
        # 0..13 明子
        # 14..15 暗子位置
        # 16 last move
        # 17 side to move
        # 18..24 captured_by
        # 25..31 lost_known
        # 32 lost_unknown_count
        C = 33
        planes = np.zeros((C, BOARD_H, BOARD_W), dtype=np.float32)

        kind_to_idx = {k: i for i, k in enumerate(PIECE_KINDS)}  # 0..6

        # board planes
        for y in range(BOARD_H):
            for x in range(BOARD_W):
                p = self.board[y][x]
                if p is None:
                    continue
                if p.covered:
                    # 暗子：只标位置，不标身份
                    if p.color == RED:
                        planes[14, y, x] = 1.0
                    else:
                        planes[15, y, x] = 1.0
                else:
                    # 明子：身份可见（无论谁的明子）
                    base = 0 if p.color == RED else 7
                    planes[base + kind_to_idx[p.kind], y, x] = 1.0

        # last move
        if self.last_move is not None:
            y1, x1, y2, x2 = self.last_move
            planes[16, y1, x1] = 1.0
            planes[16, y2, x2] = 1.0

        # side-to-move
        planes[17, :, :] = 1.0 if self.current_player == player else 0.0

        # captured_by[player] (normalize by start counts)
        for i, k in enumerate(PIECE_KINDS):
            denom = float(START_COUNTS[k])
            planes[18 + i, :, :] = (self.captured_by[player][k] / denom) if denom > 0 else 0.0

        # lost_known[player]
        for i, k in enumerate(PIECE_KINDS):
            denom = float(START_COUNTS[k])
            planes[25 + i, :, :] = (self.lost_known[player][k] / denom) if denom > 0 else 0.0

        # lost_unknown_count[player] (最多15)
        planes[32, :, :] = float(self.lost_unknown_count[player]) / 15.0

        return planes

    # ---- 走法方向表（等价于你原来的 directions + 特判） ----
    # 我们用“红方视角”为基准：红往上走（dy=-1）。
    # 黑方通过坐标变换映射到红方视角生成，再映射回去 —— 这样能最像你原代码里用 rotate 的效果。
    
    DIRS = {
        'P': [(-1, 0), (0, -1), (0, 1)],
        'I': [(-1, 0)],  # 暗兵
        'N': [(-2, 1), (-1, 2), (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1)],
        'E': [(-2, 1), (-1, 2), (-1, -2), (-2, -1)],  # 暗马（只保留“向前”的四个）
        'B': [(-2, 2), (2, 2), (2, -2), (-2, -2)],
        'F': [(-2, 2), (-2, -2)],  # 暗相（只向前）
        'R': [(-1, 0), (0, 1), (1, 0), (0, -1)],
        'D': [(-1, 0), (0, 1), (0, -1)],  # 暗车（无后退）
        'C': [(-1, 0), (0, 1), (1, 0), (0, -1)],
        'H': [(-1, 0), (0, 1), (1, 0), (0, -1)],  # 暗炮
        'A': [(-1, 1), (1, 1), (1, -1), (-1, -1)],
        'G': [(-1, 1), (-1, -1)],  # 暗士（只向前斜）
        'K': [(-1, 0), (0, 1), (1, 0), (0, -1)]
    }
    
    def _to_red_view(self, color, y, x):
        # 黑方坐标旋转到红方视角
        if color == 0:  # RED
            return y, x
        return 9 - y, 8 - x
    
    def _from_red_view(self, color, y, x):
        if color == 0:
            return y, x
        return 9 - y, 8 - x
    
    def _palace_ok_red_view(self, y, x):
        # 等价于你原代码对 K 的九宫限制：行 7..9，列 3..5（红方视角）
        return (7 <= y <= 9) and (3 <= x <= 5)
    
    def _gen_moves_for_piece_red_view(self, y, x, p, board_get):
        """
        y,x 是红方视角坐标
        p 是 Piece（但 y,x 已在红方视角）
        board_get(yy,xx) -> (Piece|None) 也是红方视角访问
        产出 (y2,x2) in 红方视角
        """
        sym = None
        if p.covered:
            sym = p.cover_type  # 'D','E','F','G','H','I'
        else:
            sym = p.kind        # 'R','N','B','A','C','P','K'
    
        if sym is None:
            return
    
        # 1) 将帅照面：红方视角下，当前方 K 往上扫，如果直线上遇到对方 K 且无遮挡，可吃
        if sym == 'K':
            for yy in range(y - 1, -1, -1):
                q = board_get(yy, x)
                if q is None:
                    continue
                if (not q.covered) and q.kind == 'K' and q.color != p.color:
                    yield (yy, x)
                break  # 遇到任何子就停
    
        # 2) 炮/暗炮：隔山打牛
        if sym in ('C', 'H'):
            for dy, dx in self.DIRS[sym]:
                cfoot = 0
                yy, xx = y + dy, x + dx
                while 0 <= yy < 10 and 0 <= xx < 9:
                    q = board_get(yy, xx)
                    if cfoot == 0:
                        if q is None:
                            yield (yy, xx)  # 空格可走
                        else:
                            cfoot = 1       # 遇到炮架
                    else:
                        if q is None:
                            pass
                        else:
                            if q.color != p.color:
                                yield (yy, xx)  # 隔一个子吃
                            break
                    yy += dy
                    xx += dx
            return
    
        # 3) 其余棋：按 directions 扫描/跳跃
        for dy, dx in self.DIRS[sym]:
            yy, xx = y + dy, x + dx
    
            # 出界直接跳过
            if not (0 <= yy < 10 and 0 <= xx < 9):
                continue
    
            q = board_get(yy, xx)
            if q is not None and q.color == p.color:
                continue
    
            # 兵过河后才能横走：红方视角下，y>4 表示未过河
            if sym == 'P' and dx != 0 and y > 4:
                continue
    
            # 九宫限制（只限制 K，与你原代码一致；A/B 不限制因此可过河）
            if sym == 'K' and (not self._palace_ok_red_view(yy, xx)):
                continue
    
            # 暗士 G：只能落到花心 (8,4) —— 等价于你原代码 j==183
            if sym == 'G' and not (yy == 8 and xx == 4):
                continue
    
            # 马/暗马：蹩马腿
            if sym in ('N', 'E'):
                if abs(dy) == 2:
                    leg_y, leg_x = y + dy // 2, x
                else:
                    leg_y, leg_x = y, x + dx // 2
                if board_get(leg_y, leg_x) is not None:
                    continue
    
            # 象/暗相：塞象眼
            if sym in ('B', 'F'):
                eye_y, eye_x = y + dy // 2, x + dx // 2
                if board_get(eye_y, eye_x) is not None:
                    continue
    
            # 车/暗车：滑行（D 无后退已经通过 DIRS 限制）
            if sym in ('R', 'D'):
                # 对 R/D 用滑行：一直走到堵住
                ddy, ddx = dy, dx
                yy, xx = y + ddy, x + ddx
                while 0 <= yy < 10 and 0 <= xx < 9:
                    q = board_get(yy, xx)
                    if q is None:
                        yield (yy, xx)
                    else:
                        if q.color != p.color:
                            yield (yy, xx)
                        break
                    yy += ddy
                    xx += ddx
                continue
    
            # 其它（兵、士、相、马、暗兵、暗马、暗相、暗士）：一步到位
            yield (yy, xx)

    def legal_actions(self):
        # debug: ensure all covered pieces have cover_type
        # for y in range(10):
        #     for x in range(9):
        #         p = self.board[y][x]
        #         if p is not None and p.covered and p.cover_type is None:
        #             raise ValueError(f"covered piece at {(y,x)} missing cover_type")
        
        acts = []
        cp = self.current_player
    
        # moves：按红方视角生成，再映射回绝对坐标
        def board_get_red_view(yy, xx):
            ay, ax = self._from_red_view(cp, yy, xx)
            return self.board[ay][ax]
    
        for ay in range(10):
            for ax in range(9):
                p = self.board[ay][ax]
                if p is None or p.color != cp:
                    continue
    
                ry, rx = self._to_red_view(cp, ay, ax)
                # 用红方视角访问 piece：注意 piece 本身还是同一个对象，不影响
                for (ry2, rx2) in self._gen_moves_for_piece_red_view(ry, rx, p, board_get_red_view):
                    by2, bx2 = self._from_red_view(cp, ry2, rx2)
                    # encode move
                    aid = encode_move(ay, ax, by2, bx2)
                    if aid is not None:
                        acts.append(aid)
    
        return acts

    def apply_action(self, action_id: int):
        a = decode_action(action_id)
        cp = self.current_player
        op = other(cp)
        
        # move
        y1, x1, y2, x2 = a.move
        if not in_bounds(y2, x2):
            raise ValueError("Illegal move out of bounds")
    
        mover = self.board[y1][x1]
        if mover is None or mover.color != cp:
            raise ValueError("Illegal move: no piece / wrong color")
    
        # （可选）强校验：目标必须在 legal_actions 里
        # if action_id not in set(self.legal_actions()): raise ValueError("Illegal move")
    
        target = self.board[y2][x2]
        if target is not None and target.color == cp:
            raise ValueError("Illegal capture own piece")
    
        # --- capture bookkeeping（保持你之前那套可见性规则） ---
        if target is not None:
            captured = (target is not None)
            if captured:
                self.no_capture_plies = 0
            else:
                self.no_capture_plies += 1

            # 吃子方：永远知道真实身份
            # 如果目标是暗子：真实身份在 target.true_kind
            captured_kind = target.kind
            if target.covered:
                if target.true_kind is None:
                    raise ValueError("captured covered piece missing true_kind")
                captured_kind = target.true_kind
    
            self.captured_by[cp][captured_kind] += 1
    
            if target.covered:
                # 被吃方：不知道自己暗子被吃的真实身份，只知道数量
                self.lost_unknown_count[op] += 1
                # 裁判真值事件列表（ISMCTS 会重采样 victim 的这个列表）
                self.lost_unknown_events[op].append(captured_kind)
            else:
                # 被吃方：明子被吃，身份可见
                self.lost_known[op][captured_kind] += 1
    
            # 终局：吃掉 K 直接结束（你原代码也是吃 'k' 判 checkmate）
            if (not target.covered) and target.kind == "K":
                # 直接落子并结束
                self.board[y1][x1] = None
                self.board[y2][x2] = mover
                self.last_move = (y1, x1, y2, x2)
                self.plies += 1
                self.current_player = op
                return
    
        # --- move / reveal-on-move ---
        self.board[y1][x1] = None
    
        # 如果起点是暗子：走完自动翻开（等价于你 mymove_check 里 mapping[i] 那段）
        if mover.covered:
            mover.covered = False
            if mover.true_kind is None:
                raise ValueError("moved covered piece missing true_kind")
            mover.kind = mover.true_kind
            mover.cover_type = None  # 可选
    
        self.board[y2][x2] = mover
        self.last_move = (y1, x1, y2, x2)
    
        self.current_player = op
        self.plies += 1


    # ---------------------- 终局判定（最小可跑） ----------------------
    def game_end(self) -> Tuple[bool, Optional[int]]:
        """
        返回 (ended, winner)
        winner: RED/BLACK or None(和棋)
        """
        # king existence
        red_k, black_k = False, False
        for y in range(BOARD_H):
            for x in range(BOARD_W):
                p = self.board[y][x]
                if p is None:
                    continue
                if p.kind == "K":
                    if p.color == RED:
                        red_k = True
                    else:
                        black_k = True
        if not red_k:
            return True, BLACK
        if not black_k:
            return True, RED

        if self.no_capture_plies >= 30:
            return True, None
        if self.plies >= MAX_PLIES:
            return True, None

        # no legal moves -> draw (placeholder)
        if len(self.legal_actions()) == 0:
            return True, None

        return False, None

    def clone(self) -> "JieqiState":
        return copy.deepcopy(self)

    # ---------------------- ISMCTS: 信息集确定化（关键） ----------------------
    def determinize_for(self, player: int, seed: Optional[int] = None) -> "JieqiState":
        """
        生成一个“对 player 来说一致”的确定化世界：
          - 采样棋盘上所有 covered piece 的 kind（双方都一样未知）
          - 对 player 自己“暗子被吃但未知身份”的那批事件，重采样其 kind；
            并据此让对手在该确定化世界里“知道自己吃到了哪些”（captured_by[opponent]会对应变化）

        注意：这是骨架实现，主要保证“不会把暗子真身份喂给 player 的 obs_planes”。
        """
        d = self.clone()
        rng = random.Random(seed if seed is not None else d.rng.randint(0, 10**9))

        # helper: build remaining pool for a given color
        def build_pool(color: int, victim_is_player: bool) -> Dict[str, int]:
            # 暗子池：不含K
            pool = {k: START_COUNTS[k] for k in KINDS_NO_K}

            # subtract all uncovered pieces on board (excluding K)
            for y in range(BOARD_H):
                for x in range(BOARD_W):
                    p = d.board[y][x]
                    if p is None or p.color != color:
                        continue
                    if (not p.covered) and p.kind != "K":
                        _sub_count(pool, p.kind, 1)

            # subtract known losses for this color
            for k in KINDS_NO_K:
                n = d.lost_known[color][k]
                if n:
                    _sub_count(pool, k, n)

            # subtract unknown-loss events of this color IF victim is NOT 'player'
            # 因为对 player 自己的 unknown-loss events，我们要重采样，不用真值扣。
            if not victim_is_player:
                for t in d.lost_unknown_events[color]:
                    if t != "K":  # 理论上不会出现
                        _sub_count(pool, t, 1)

            # also subtract pieces captured by opponent that are known to victim? 已包含在 lost_known/unknown
            return pool

        # 1) 重采样 player 的 lost_unknown_events[player]
        victim = player
        capturer = other(player)
        unk_k = len(d.lost_unknown_events[victim])

        if unk_k > 0:
            pool_v = build_pool(victim, victim_is_player=True)

            # 我方暗子被吃（未知） + 我方棋盘暗子（未知），一起采样更一致
            victim_covered_positions = [(y, x) for y in range(BOARD_H) for x in range(BOARD_W)
                                        if (d.board[y][x] is not None and d.board[y][x].color == victim and d.board[y][x].covered)]
            total_slots = unk_k + len(victim_covered_positions)

            sampled = _sample_without_replacement(rng, pool_v, total_slots)

            # 前 unk_k 个给“被吃暗子事件”
            new_unk = sampled[:unk_k]
            # 后面给棋盘暗子
            assign_onboard = sampled[unk_k:]

            d.lost_unknown_events[victim] = new_unk  # 裁判真值在确定化世界里替换为采样值

            # 同步：capturer 在该确定化世界里知道自己吃到的类型
            # captured_by[capturer] 应该等于 victim 的全部损失（已知 + 采样未知）
            d.captured_by[capturer] = _empty_counts()
            for k in PIECE_KINDS:
                d.captured_by[capturer][k] += d.lost_known[victim][k]
            for t in new_unk:
                d.captured_by[capturer][t] += 1

            # 赋值给 victim 的棋盘暗子 true_kind（保持 kind="?" 不泄露）
            rng.shuffle(victim_covered_positions)
            for (y, x), tk in zip(victim_covered_positions, assign_onboard):
                piece = d.board[y][x]
                if piece is None:
                    continue
                piece.true_kind = tk
                piece.kind = "?"
        else:
            # unk_k == 0：也必须给 victim 的棋盘暗子采样 true_kind
            pool_v = build_pool(victim, victim_is_player=True)
            victim_covered_positions = [(y, x) for y in range(BOARD_H) for x in range(BOARD_W)
                                        if (d.board[y][x] is not None and d.board[y][x].color == victim and d.board[y][x].covered)]
            if victim_covered_positions:
                assign_onboard = _sample_without_replacement(rng, pool_v, len(victim_covered_positions))
                rng.shuffle(victim_covered_positions)
                for (y, x), tk in zip(victim_covered_positions, assign_onboard):
                    piece = d.board[y][x]
                    piece.true_kind = tk
                    piece.kind = "?"

        # 2) 采样 opponent 的棋盘暗子 kind（对 player 不可见）
        opp = other(player)
        opp_pool = build_pool(opp, victim_is_player=False)

        opp_covered_positions = [(y, x) for y in range(BOARD_H) for x in range(BOARD_W)
                                 if (d.board[y][x] is not None and d.board[y][x].color == opp and d.board[y][x].covered)]
        if len(opp_covered_positions) > 0:
            sampled_opp = _sample_without_replacement(rng, opp_pool, len(opp_covered_positions))
            rng.shuffle(opp_covered_positions)
            for (y, x), tk in zip(opp_covered_positions, sampled_opp):
                piece = d.board[y][x]
                piece.true_kind = tk
                piece.kind = "?"

        return d
