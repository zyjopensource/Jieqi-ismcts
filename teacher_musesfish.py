# teacher_musesfish.py
from __future__ import annotations
import random
import numpy as np
import musesfish_pvs_20260219 as mf

from actions import encode_move


def idx_to_yx(idx: int):
    y = (idx // 16) - 3
    x = (idx % 16) - 3
    return y, x


def state_to_mf_board_abs(state) -> str:
    # 256 chars: 16x16; rows 0..14 end with '\n' at col 15
    arr = [' '] * 256
    for r in range(15):
        arr[r * 16 + 15] = '\n'

    # fill playable squares with '.'
    for y in range(10):
        for x in range(9):
            idx = (3 + y) * 16 + (3 + x)
            arr[idx] = '.'

    # place pieces
    for y in range(10):
        for x in range(9):
            p = state.board[y][x]
            if p is None:
                continue
            idx = (3 + y) * 16 + (3 + x)

            if p.covered:
                # 暗子用 cover_type：红方大写 D..I，黑方小写 d..i
                ch = p.cover_type
                if ch is None:
                    # 理论上不应发生；防御一下
                    ch = "I"
                ch = ch.lower() if p.color == 1 else ch
            else:
                # 明子用 kind：红方大写，黑方小写
                ch = p.kind
                ch = ch.lower() if p.color == 1 else ch

            arr[idx] = ch

    return ''.join(arr)


def state_to_mf_position(state):
    board_abs = state_to_mf_board_abs(state)
    pos = mf.Position(board_abs, 0, True, 0).set()  # 先当作“红走”的绝对局面
    if state.current_player == 1:                   # BLACK
        pos = pos.rotate()                          # 黑走时旋转到“走子方为大写”
    return pos


class MusefishTeacher:
    """
    Teacher wrapper with musesfish forbidden-move mechanism enabled.

    Notes:
    - musesfish 的 generate_forbiddenmoves(pos, ...) 依赖全局 mf.cache 与 mf.forbidden_moves
    - 我们在每步选招前：
        1) mf.cache[pos.board] += 1  (记录历史局面)
        2) mf.generate_forbiddenmoves(pos, step=state.plies)  (生成禁着集合)
      然后再搜索，并对搜索结果做一次禁着兜底过滤。
    """

    def __init__(
        self,
        think_time: float = 0.05,
        eps: float = 0.02,
        seed: int = 0,
        check_bozi: bool = True,
    ):
        self.searcher = mf.Searcher()
        # 这里 calc_average() 可能较重，但只在 init 做一次
        self.searcher.calc_average()

        self.think_time = float(think_time)
        self.eps = float(eps)
        self.rng = random.Random(seed)
        self.check_bozi = bool(check_bozi)

    # ---------- forbidden-move support ----------
    def reset_game(self):
        """每局开始调用：清空引擎侧的重复局面缓存与禁着集合，防止跨局污染。"""
        try:
            mf.cache.clear()
        except Exception:
            mf.cache = {}

        try:
            mf.forbidden_moves.clear()
        except Exception:
            mf.forbidden_moves = set()

    def _update_forbidden(self, pos, step: int):
        """更新 mf.cache 并生成 mf.forbidden_moves。pos 必须是当前走子方视角。"""
        # 记录当前局面出现次数
        mf.cache[pos.board] = mf.cache.get(pos.board, 0) + 1

        # 生成禁着（函数内部会写 global forbidden_moves）
        try:
            mf.generate_forbiddenmoves(pos, check_bozi=self.check_bozi, step=int(step))
        except TypeError:
            # 兼容老版本签名：generate_forbiddenmoves(pos, check_bozi=True, step=0)
            mf.generate_forbiddenmoves(pos, self.check_bozi, int(step))

    def _engine_move_to_action_id(self, pos, mv):
        """
        将引擎 move (i,j) 转换为动作 id。
        需要把 pos.turn=False 时的坐标做反转（与你原实现一致）。
        """
        if mv is None:
            return None
        i, j = mv

        # musesfish 内部坐标在某些情况下需要反转回绝对坐标
        if pos.turn is False:
            i, j = 254 - i, 254 - j

        y1, x1 = idx_to_yx(i)
        y2, x2 = idx_to_yx(j)

        return encode_move(y1, x1, y2, x2)

    def _pick_non_forbidden_fallback(self, pos, legal_set):
        """
        当搜索返回禁着/无效着时，从 pos.gen_moves() 中挑一个：
        - 不在 mf.forbidden_moves
        - 能映射到 action_id
        - action_id 在 legal_set
        这里用随机挑选（简单但能立刻解决长将/重复局面）。
        """
        try:
            forb = mf.forbidden_moves
        except Exception:
            forb = set()

        cand = []
        for mv in pos.gen_moves():
            if mv in forb:
                continue
            aid = self._engine_move_to_action_id(pos, mv)
            if aid is None:
                continue
            if aid in legal_set:
                cand.append(aid)

        if not cand:
            return None
        return self.rng.choice(cand)

    # ---------- main policy ----------
    def pick_action(self, state, legal, legal_set):
        pos = state_to_mf_position(state)
        pos.set()

        # 关键：启用禁着（重复局面/博子/重复将军等）
        self._update_forbidden(pos, step=getattr(state, "plies", 0))

        last_move = None
        for d, mv, sc in self.searcher.search(pos, history=(), time_limit=self.think_time):
            last_move = mv

        if last_move is None:
            return None

        # 如果搜索返回的是禁着，做兜底 fallback
        try:
            if last_move in mf.forbidden_moves:
                aid_fb = self._pick_non_forbidden_fallback(pos, legal_set)
                return aid_fb
        except Exception:
            pass

        aid = self._engine_move_to_action_id(pos, last_move)
        if aid is None:
            return None
        if aid not in legal_set:
            # 搜索结果不在本项目合法动作集（可能映射失败或规则差异），fallback
            return self._pick_non_forbidden_fallback(pos, legal_set)

        return aid

    def get_action_probs(self, state, temp: float = 1.0, legal=None):
        if legal is None:
            legal = state.legal_actions()
        if not legal:
            return [], []

        legal_set = set(legal)
        aid = self.pick_action(state, legal, legal_set)
        if aid is None:
            # 如果引擎也给不出，就随机合法着
            aid = self.rng.choice(legal)

        acts = list(legal)

        # 生成 one-hot + eps 平滑
        idx_map = {a: i for i, a in enumerate(acts)}
        probs = np.full((len(acts),), self.eps / len(acts), dtype=np.float32)
        probs[idx_map[aid]] += 1.0 - self.eps
        return acts, probs.tolist()

    def update_with_move(self, last_action):
        # teacher 不维护搜索树，空实现即可
        return
