# ui_viewer.py
import sys
import time
import pygame
import pygame.freetype

from actions import decode_action
from constants import RED, BLACK

# 颜色/字体
RED_CHESS_COLOR = [255, 255, 255]
BLACK_CHESS_COLOR = [0, 0, 0]
SCREEN_COLOR = [238, 154, 73]
LINE_COLOR = [0, 0, 0]

if sys.platform == "linux":
    FONT_PATH = "/usr/share/fonts/truetype/arphic/ukai.ttc"
else:
    FONT_PATH = "C:/Windows/Fonts/simkai.ttf"


def piece_to_cn(kind: str, color: int) -> str:
    """把 kind 映射成你 UI 里用的中文棋子字"""
    # 红：车马炮象士兵帅
    # 黑：俥傌炮相仕卒将
    if color == RED:
        return {
            "R": "车",
            "N": "马",
            "C": "炮",
            "B": "象",
            "A": "士",
            "P": "兵",
            "K": "帅",
        }.get(kind, "?")
    else:
        return {
            "R": "俥",
            "N": "傌",
            "C": "炮",
            "B": "相",
            "A": "仕",
            "P": "卒",
            "K": "将",
        }.get(kind, "?")

class JieqiPygameViewer:
    """
    从 JieqiState 直接渲染棋盘：
      - 不读 stdout
      - 不需要点击
      - 每步 apply_action 后调用 render(state, last_action_id, info)
    """

    def __init__(self, width=670, height=670, board_width=500):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Jieqi self-play viewer")

        self.font = pygame.freetype.Font(FONT_PATH, 30)
        self.font.strong = True
        self.font.antialiased = True

        self.footer_font = pygame.freetype.Font(FONT_PATH, 22)
        self.footer_font.strong = True
        self.footer_font.antialiased = True

        self.width = board_width
        self.row_spacing = board_width / 10
        self.col_spacing = board_width / 8
        self.start_point = (60, 60)
        self.start_point_x = self.start_point[0]
        self.start_point_y = self.start_point[1]

        self.footer_height = 34

        # 用于显示“上一手起点”
        self.last_from = None  # (y,x) in 0..9,0..8
        self.last_to = None

        # 吃子显示：这里直接显示“红方吃子 / 黑方吃子”
        # 每项：(name_cn, is_dark)
        self.captured_by_red = []
        self.captured_by_black = []

        self._should_quit = False

    def should_quit(self) -> bool:
        return self._should_quit

    def pump_events(self):
        """保持窗口响应；ESC 或关闭窗口会置 should_quit"""
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self._should_quit = True
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    self._should_quit = True

    def _get_chess_pos(self, row_1to10, col_1to9):
        center = [
            self.start_point_x + self.col_spacing * (col_1to9 - 1),
            self.start_point_y + self.row_spacing * (row_1to10 - 1),
        ]
        radius = self.row_spacing / 2
        return center, radius

    def _draw_board_grid(self):
        x, y = self.start_point
        # 横线 10
        for _ in range(10):
            pygame.draw.line(self.screen, LINE_COLOR, [x, y], [x + self.width, y])
            y += self.row_spacing

        x, y = self.start_point
        # 竖线 9
        for _ in range(9):
            pygame.draw.line(self.screen, LINE_COLOR, [x, y], [x, y + self.row_spacing * 9])
            x += self.col_spacing

        # 中间“楚河汉界”留白（按你原 UI 方式抹掉一段）
        x, y = self.start_point_x + self.col_spacing, self.start_point_y + self.row_spacing * 4
        for _ in range(7):
            pygame.draw.line(self.screen, SCREEN_COLOR, [x, y], [x, y + self.row_spacing])
            x += self.col_spacing

        # 九宫斜线（上、下各一组）
        def p(row, col):
            c, _ = self._get_chess_pos(row, col)
            return c

        # 上九宫（行 1..3，列 4..6）
        pygame.draw.line(self.screen, LINE_COLOR, p(1, 4), p(3, 6))
        pygame.draw.line(self.screen, LINE_COLOR, p(1, 6), p(3, 4))
        # 下九宫（行 8..10，列 4..6）
        pygame.draw.line(self.screen, LINE_COLOR, p(8, 4), p(10, 6))
        pygame.draw.line(self.screen, LINE_COLOR, p(8, 6), p(10, 4))

    def _draw_a_chess(self, row, col, chess_color, chess_name):
        center, radius = self._get_chess_pos(row, col)

        border_w = 1
        border_color = [255, 255, 255] if chess_color == BLACK_CHESS_COLOR else [0, 0, 0]

        pygame.draw.circle(self.screen, border_color, center, int(radius))
        inner_r = max(1, int(radius - border_w))
        pygame.draw.circle(self.screen, chess_color, center, inner_r)

        # 花纹同心圆
        pattern_r = max(1, int(inner_r * 0.92))
        pygame.draw.circle(self.screen, border_color, center, pattern_r, width=1)

        # 暗子只画圆不写字
        if chess_name == "暗":
            return

        # 文字颜色
        if chess_color == BLACK_CHESS_COLOR:
            font_color = [255, 255, 255]
        else:
            font_color = [255, 0, 0]

        text_rect = self.font.get_rect(chess_name)
        text_rect.center = (center[0], center[1])
        self.font.render_to(self.screen, text_rect.topleft, chess_name, font_color)

    def _draw_last_move_marker(self):
        """在上一手 from 画一个空心蓝圈"""
        if self.last_from is None:
            return
        y, x = self.last_from
        row = y + 1
        col = x + 1
        center, _ = self._get_chess_pos(row, col)
        pygame.draw.circle(self.screen, [0, 0, 255], center, 10, width=2)

    def _draw_footer(self, info_text: str):
        w, h = self.screen.get_size()
        y0 = h - self.footer_height
        pygame.draw.rect(self.screen, [245, 245, 245], pygame.Rect(0, y0, w, self.footer_height))

        base = "ESC 退出"
        x = 10
        y = y0 + self.footer_height // 2

        rect_base = self.footer_font.get_rect(base)
        rect_base.midleft = (x, y)
        self.footer_font.render_to(self.screen, rect_base.topleft, base, [0, 0, 0])
        x += rect_base.width + 20

        if info_text:
            rect_info = self.footer_font.get_rect(info_text)
            rect_info.midleft = (x, y)
            self.footer_font.render_to(self.screen, rect_info.topleft, info_text, [0, 0, 0])

    def _draw_captured_area(self):
        """棋盘下方两行显示红/黑吃子"""
        w, h = self.screen.get_size()
        board_bottom = self.start_point_y + self.row_spacing * 9
        margin_x = 12
        y1 = int(board_bottom + 52)  # 第一行
        y2 = int(y1 + 40)            # 第二行

        r = 16
        step = 2 * r + 6

        area_h = 40 * 2 + 12
        area_rect = pygame.Rect(0, y1 - 8, w, area_h)
        pygame.draw.rect(self.screen, [245, 245, 245], area_rect)

        self.footer_font.render_to(self.screen, (margin_x, y1 - 6), "红方吃子:", [0, 0, 0])
        self.footer_font.render_to(self.screen, (margin_x, y2 - 6), "黑方吃子:", [0, 0, 0])

        pad = 30
        w1 = self.footer_font.get_rect("红方吃子:").width
        w2 = self.footer_font.get_rect("黑方吃子:").width
        x0 = margin_x + max(w1, w2) + pad

        def _draw_cap_piece(cx, cy, chess_color, name, is_dark):
            border_w = 1
            border_color = [255, 255, 255] if chess_color == BLACK_CHESS_COLOR else [0, 0, 0]
            pygame.draw.circle(self.screen, border_color, (cx, cy), r)
            pygame.draw.circle(self.screen, chess_color, (cx, cy), r - border_w)

            if is_dark:
                pygame.draw.circle(self.screen, [0, 0, 255], (cx + r - 5, cy - r + 5), 6, 0)

            if not name:
                return

            if chess_color == BLACK_CHESS_COLOR:
                font_color = [255, 255, 255]
            else:
                font_color = [255, 0, 0]

            text_rect = self.footer_font.get_rect(name)
            text_rect.center = (cx, cy)
            self.footer_font.render_to(self.screen, text_rect.topleft, name, font_color)

        # 红方吃子（吃到黑子）
        x = x0
        for name, is_dark in self.captured_by_red[-20:]:
            if x + r > w - margin_x:
                break
            _draw_cap_piece(x, y1 + 18, BLACK_CHESS_COLOR, name, is_dark)
            x += step

        # 黑方吃子（吃到红子）
        x = x0
        for name, is_dark in self.captured_by_black[-20:]:
            if x + r > w - margin_x:
                break
            _draw_cap_piece(x, y2 + 18, RED_CHESS_COLOR, name, is_dark)
            x += step

    def on_action(self, state_before, action_id: int):
        """用于更新 last_move 与吃子列表（从 state_before 读取被吃者信息）"""
        a = decode_action(action_id)
        if a.kind != "move":
            return

        y1, x1, y2, x2 = a.move
        self.last_from = (y1, x1)
        self.last_to = (y2, x2)

        target = state_before.board[y2][x2]
        if target is None:
            return

        # 被吃的是暗子：显示真实身份并标记暗
        is_dark = bool(target.covered)
        if target.covered:
            # true_kind 若没填，就退化用 target.kind（你的 determinize/init 已保证 true_kind）
            tk = target.true_kind if target.true_kind is not None else target.kind
            name = piece_to_cn(tk, target.color)
        else:
            name = piece_to_cn(target.kind, target.color)

        mover = state_before.board[y1][x1]
        if mover is None:
            return

        if mover.color == RED:
            self.captured_by_red.append((name, is_dark))
        else:
            self.captured_by_black.append((name, is_dark))

    def render(self, state, info_text: str = ""):
        self.pump_events()
        if self._should_quit:
            return

        self.screen.fill(SCREEN_COLOR)
        self._draw_board_grid()

        # 画棋子：state.board[y][x] -> row=y+1 col=x+1
        for y in range(10):
            for x in range(9):
                p = state.board[y][x]
                if p is None:
                    continue
                row, col = y + 1, x + 1
                if p.covered:
                    chess_name = "暗"
                else:
                    chess_name = piece_to_cn(p.kind, p.color)

                chess_color = RED_CHESS_COLOR if p.color == RED else BLACK_CHESS_COLOR
                self._draw_a_chess(row, col, chess_color, chess_name)

        self._draw_captured_area()
        self._draw_last_move_marker()
        self._draw_footer(info_text)

        pygame.display.flip()
