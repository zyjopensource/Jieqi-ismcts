# collect_teacher_data.py  (NO TORCH)
import os
import numpy as np
import random
import time
from state import JieqiState
from actions import ACTION_SIZE
from teacher_musesfish import MusefishTeacher
from threading import Thread
from queue import Queue, Empty
from multiprocessing import get_context

SHOW_UI = False   # True: 显示pygame走子回放; False: 纯数据生成
UI_FPS = 0.3      # 控制走子播放速度（越大越快）

def normalize_probs(acts, probs):
    """强制把 probs 清洗并归一化到 sum=1，必要时退化为均匀分布。"""
    if not acts:
        return [], []

    p = np.asarray(probs, dtype=np.float64)

    # 长度不一致直接退化
    if p.shape[0] != len(acts):
        p = np.ones((len(acts),), dtype=np.float64)

    # 清 NaN/Inf/负数
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p[p < 0] = 0.0

    s = float(p.sum())
    if not np.isfinite(s) or s <= 0.0:
        # 全0或异常，退化为均匀
        p[:] = 1.0 / len(p)
    else:
        p /= s

    return acts, p

def get_action_probs_async(teacher, state, temp, legal=None, viewer=None):
    """
    在后台线程跑 teacher.get_action_probs（可传 legal 复用），
    主线程持续 pump pygame 事件，保持窗口响应。
    返回：(acts, probs, status, err)
      - status == "quit" 表示用户关闭
      - err 非 None 表示 teacher 内部异常（外层应 fallback，不要当成无棋可走）
    """
    q = Queue(maxsize=1)

    def _worker():
        try:
            acts, probs = teacher.get_action_probs(state, temp=temp, legal=legal)
            q.put((acts, probs, None, None))
        except Exception as e:
            q.put((None, None, None, e))

    th = Thread(target=_worker, daemon=True)
    th.start()

    while True:
        if viewer is not None:
            viewer.pump_events()
            if viewer.should_quit():
                return [], [], "quit", None
        try:
            acts, probs, status, err = q.get(timeout=0.01)
            return acts, probs, status, err
        except Empty:
            continue

def self_play_teacher_one_game(seed: int, teacher: MusefishTeacher, viewer=None, temp: float = 1.0):
    """
    用 teacher 进行一局自我对弈，生成 (S_list, P_list, Z_list) 训练样本。

    关键改动：
    1) “无棋可走”只由环境 state.legal_actions() 判定（唯一真值来源）。
       - 若 legal 为空：当前行动方直接判负。
       - 绝不使用 teacher 返回空 acts 来判“无棋可走”，避免 teacher 内部出错导致误判。
    2) legal 只计算一次，并传给 teacher.get_action_probs(..., legal=legal) 复用，避免重复开销。
    3) teacher 出错/返回空 acts：不误判，退化为在 legal 上均匀随机（并生成对应 pi）。
    4) SHOW_UI 时通过 get_action_probs_async 保持窗口响应。
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    state = JieqiState(seed=seed)
    S_list, P_list, players = [], [], []

    while True:
        # ------------------------------------------------------------
        # 1) 终局检测：返回每步视角的 Z_list（+1/-1/0）
        # ------------------------------------------------------------
        ended, winner = state.game_end()
        if ended:
            Z_list = []
            for pl in players:
                if winner is None:
                    Z_list.append(0.0)
                elif winner == pl:
                    Z_list.append(1.0)
                else:
                    Z_list.append(-1.0)

            # UI：画最后一帧并允许用户关闭窗口
            if viewer is not None:
                info = f"GAME OVER winner={winner}  (close window / ESC)"
                viewer.render(state, info_text=info)
                while not viewer.should_quit():
                    viewer.pump_events()
                    time.sleep(0.01)

            return S_list, P_list, Z_list, winner

        # 当前行动方
        cur = state.current_player

        # ------------------------------------------------------------
        # 2) 计算观测（存 S）
        # ------------------------------------------------------------
        obs = state.obs_planes(cur)  # [C, 10, 9]

        # ------------------------------------------------------------
        # 3) 计算合法着法：这是“是否无棋可走”的唯一判定来源
        # ------------------------------------------------------------
        legal = state.legal_actions()
        if len(legal) == 0:
            # 无合法着：当前行动方负
            loser = cur
            winner = 1 - loser

            Z_list = []
            for pl in players:
                Z_list.append(1.0 if pl == winner else -1.0)

            # UI：画最后一帧并允许用户关闭窗口
            if viewer is not None:
                info = f"NO LEGAL MOVE -> winner={winner}  (close window / ESC)"
                viewer.render(state, info_text=info)
                while not viewer.should_quit():
                    viewer.pump_events()
                    time.sleep(0.01)

            return S_list, P_list, Z_list, winner

        # ------------------------------------------------------------
        # 4) 调 teacher 得到 (acts, probs)，只在 legal 上分布（减少一次 legal 计算）
        #    - teacher 异常/返回空：fallback 为 legal 上均匀分布（不误判为无棋可走）
        # ------------------------------------------------------------
        err = None
        if viewer is not None:
            # 先渲染“思考中”，让窗口立即刷新并处理事件
            viewer.render(state, info_text=f"seed={seed} ply={state.plies} thinking...")

            acts, probs, status, err = get_action_probs_async(
                teacher, state, temp=temp, legal=legal, viewer=viewer
            )
            if status == "quit":
                # 用户关闭窗口：立刻返回（外层会保存已有数据）
                Z_list = [0.0] * len(players)
                return S_list, P_list, Z_list, None
        else:
            try:
                acts, probs = teacher.get_action_probs(state, temp=temp, legal=legal)
            except Exception as e:
                acts, probs, err = [], [], e

        # teacher 出错/返回空：退化为在 legal 上均匀随机（不要误判为无棋可走）
        if err is not None or len(acts) == 0:
            acts = list(legal)
            probs = np.ones((len(acts),), dtype=np.float64) / float(len(acts))

        # ------------------------------------------------------------
        # 5) 清洗并构造 pi（存 P），并把当前行动方记录到 players
        # ------------------------------------------------------------
        acts, p = normalize_probs(acts, probs)
        a_arr = np.array(acts, dtype=np.int64)

        pi = np.zeros((ACTION_SIZE,), dtype=np.float32)
        pi[a_arr] = p.astype(np.float32)

        S_list.append(obs.astype(np.float32))
        P_list.append(pi)
        players.append(cur)

        # ------------------------------------------------------------
        # 6) 采样一个动作并落子（状态推进）
        # ------------------------------------------------------------
        aid = int(np_rng.choice(a_arr, p=p))

        if viewer is not None:
            # 记录吃子/上一手（用落子前的 state 读取被吃子信息）
            viewer.on_action(state, aid)

        # 这里如果你未来要强校验，可改为 state.apply_action(aid, legal_cache=legal)
        state.apply_action(aid)

        # ------------------------------------------------------------
        # 7) UI 渲染与可响应延迟
        # ------------------------------------------------------------
        if viewer is not None:
            info = f"seed={seed} ply={state.plies} turn={'红' if state.current_player==0 else '黑'}"
            viewer.render(state, info_text=info)

            if UI_FPS > 0:
                delay = 1.0 / UI_FPS
                t_end = time.perf_counter() + delay
                while time.perf_counter() < t_end and not viewer.should_quit():
                    viewer.pump_events()
                    time.sleep(0.01)

            if viewer.should_quit():
                # 用户关闭窗口：返回已有数据
                Z_list = [0.0] * len(players)
                return S_list, P_list, Z_list, None


def save_chunk(out_dir: str, chunk_name: str, samples):
    """
    chunk_name: e.g. "w00_teacher_chunk_000012"
    """
    os.makedirs(out_dir, exist_ok=True)
    S = np.stack([s for (s, p, z) in samples], axis=0).astype(np.float32)
    P = np.stack([p for (s, p, z) in samples], axis=0).astype(np.float32)
    Z = np.array([z for (s, p, z) in samples], dtype=np.float32)

    path = os.path.join(out_dir, f"{chunk_name}.npz")
    np.savez_compressed(path, S=S, P=P, Z=Z)
    print(f"[saved] {path}  samples={len(samples)}  S={S.shape}  P={P.shape}", flush=True)
    return path

def _fmt_hms(seconds: float) -> str:
    s = int(seconds)
    hh = s // 3600
    mm = (s % 3600) // 60
    ss = s % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"

def debug_print_chunk(npz_path: str, k: int = 3):
    """
    读取刚保存的 npz，打印前 k 条样本的关键信息，验证内容是否合理。
    """
    data = np.load(npz_path)
    S = data["S"]
    P = data["P"]
    Z = data["Z"]

    print(f"[debug_chunk] file={npz_path}")
    print(f"  S: shape={S.shape} dtype={S.dtype}  min={S.min():.3g} max={S.max():.3g} mean={S.mean():.3g}")
    print(f"  P: shape={P.shape} dtype={P.dtype}  min={P.min():.3g} max={P.max():.3g} mean={P.mean():.3g}")
    print(f"  Z: shape={Z.shape} dtype={Z.dtype}  values(head)={Z[:min(k, len(Z))].tolist()}")

    n = min(k, S.shape[0])
    for i in range(n):
        pi = P[i]
        nz = int(np.count_nonzero(pi))
        ssum = float(pi.sum())
        top_idx = int(np.argmax(pi)) if pi.size > 0 else -1
        top_val = float(pi[top_idx]) if top_idx >= 0 else 0.0
        print(f"  sample[{i}]: Z={float(Z[i]):+.1f}  pi_nonzero={nz}  pi_sum={ssum:.6f}  pi_top=({top_idx},{top_val:.6f})")

def worker_run(worker_id: int,
               out_dir: str,
               global_seed: int,
               game_start: int,
               game_end: int,
               chunk_size: int,
               think_time: float,
               eps: float,
               temp: float,
               debug_chunk: bool):
    """
    每个进程独立跑 [game_start, game_end) 的局，并独立写 chunk 文件（带 worker_id 前缀避免冲突）
    """
    t0 = time.perf_counter()

    teacher = MusefishTeacher(think_time=think_time, eps=eps, seed=worker_id)
    # warmup
    _ = teacher.get_action_probs(JieqiState(seed=12345 + worker_id), temp=1.0)

    win_stat = {0: 0, 1: 0, None: 0}
    total_steps = 0
    buffer = []
    local_chunk_id = 0

    # 可选：每 worker 单独目录也行（更整洁）
    # worker_dir = os.path.join(out_dir, f"w{worker_id:02d}")
    # os.makedirs(worker_dir, exist_ok=True)
    # write_dir = worker_dir
    write_dir = out_dir

    for game_id in range(game_start, game_end):
        seed = global_seed + game_id   # 全局唯一 seed
        teacher.reset_game()

        S_list, P_list, Z_list, winner = self_play_teacher_one_game(
            seed=seed, teacher=teacher, viewer=None, temp=temp
        )
        win_stat[winner] += 1

        for s, p, z in zip(S_list, P_list, Z_list):
            buffer.append((s, p, float(z)))
        total_steps += len(S_list)

        # 分块写盘（worker 内部）
        while len(buffer) >= chunk_size:
            chunk_name = f"w{worker_id:02d}_teacher_chunk_{local_chunk_id:06d}"
            path = save_chunk(write_dir, chunk_name, buffer[:chunk_size])
            if debug_chunk:
                try:
                    debug_print_chunk(path, k=3)
                except Exception as e:
                    print(f"[debug_chunk] failed {path}: {e}", flush=True)
            buffer = buffer[chunk_size:]
            local_chunk_id += 1

        # 少量日志（避免太刷屏：比如每 100 局打印一次）
        if (game_id - game_start + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            avg = elapsed / max(1, total_steps)
            print(f"[worker {worker_id:02d}] games={game_id-game_start+1} "
                  f"steps={total_steps} time={_fmt_hms(elapsed)} avg={avg:.4f}s/step",
                  flush=True)

    # 尾巴
    if buffer:
        chunk_name = f"w{worker_id:02d}_teacher_chunk_{local_chunk_id:06d}"
        path = save_chunk(write_dir, chunk_name, buffer)
        if debug_chunk:
            try:
                debug_print_chunk(path, k=3)
            except Exception as e:
                print(f"[debug_chunk] failed {path}: {e}", flush=True)

    elapsed = time.perf_counter() - t0
    avg = elapsed / max(1, total_steps)
    print(f"[worker {worker_id:02d} done] games={game_end-game_start} win_stat={win_stat} "
          f"steps={total_steps} time={_fmt_hms(elapsed)} avg={avg:.4f}s/step",
          flush=True)

    # 返回统计给主进程汇总
    return win_stat, total_steps

def main():
    t0 = time.perf_counter()

    # -------- config --------
    out_dir = "data_teacher"
    n_games = 1
    chunk_size = 262144
    temp = 1.0
    debug_chunk = False

    think_time = 0.1
    eps = 0.01
    global_seed = 1234

    N_WORKERS = 1  # 当它==1 时允许 SHOW_UI

    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Case A: single worker -> allow SHOW_UI
    # ------------------------------------------------------------
    if N_WORKERS == 1:
        teacher = MusefishTeacher(think_time=think_time, eps=eps, seed=0)
        _ = teacher.get_action_probs(JieqiState(seed=12345), temp=1.0)  # warmup

        viewer = None
        if SHOW_UI:
            from ui_viewer import JieqiPygameViewer
            viewer = JieqiPygameViewer()

        chunk_id = 0
        buffer = []
        total_steps = 0
        win_stat = {0: 0, 1: 0, None: 0}

        for g in range(n_games):
            seed = global_seed + g
            teacher.reset_game()

            S_list, P_list, Z_list, winner = self_play_teacher_one_game(
                seed=seed, teacher=teacher, viewer=viewer, temp=temp
            )
            win_stat[winner] += 1

            for s, p, z in zip(S_list, P_list, Z_list):
                buffer.append((s, p, float(z)))
            total_steps += len(S_list)

            elapsed = time.perf_counter() - t0
            avg = elapsed / max(1, total_steps)
            print(f"[game {g+1}/{n_games}] plies={len(S_list)} winner={winner} "
                  f"total_time={_fmt_hms(elapsed)} total_steps={total_steps} avg={avg:.4f}s/step",
                  flush=True)

            while len(buffer) >= chunk_size:
                save_chunk(out_dir, chunk_id, buffer[:chunk_size])
                if debug_chunk:
                    npz_path = os.path.join(out_dir, f"teacher_chunk_{chunk_id:04d}.npz")
                    try:
                        debug_print_chunk(npz_path, k=3)
                    except Exception as e:
                        print(f"[debug_chunk] failed {npz_path}: {e}", flush=True)

                buffer = buffer[chunk_size:]
                chunk_id += 1

        if buffer:
            save_chunk(out_dir, chunk_id, buffer)
            if debug_chunk:
                npz_path = os.path.join(out_dir, f"teacher_chunk_{chunk_id:04d}.npz")
                try:
                    debug_print_chunk(npz_path, k=3)
                except Exception as e:
                    print(f"[debug_chunk] failed {npz_path}: {e}", flush=True)

        elapsed = time.perf_counter() - t0
        print("[done] win_stat =", win_stat)
        print(f"[done] total_steps={total_steps} total_time={_fmt_hms(elapsed)} avg={elapsed/max(1,total_steps):.4f}s/step")

        if viewer is not None:
            try:
                import pygame
                pygame.quit()
            except Exception:
                pass

        return

    # ------------------------------------------------------------
    # Case B: multi-worker -> force disable UI, run multiprocessing
    # ------------------------------------------------------------
    if SHOW_UI:
        print("[warn] SHOW_UI=True is not supported when N_WORKERS > 1. Forcing SHOW_UI=False.", flush=True)

    # 注意：多进程时不要创建 viewer，也不要用 get_action_probs_async
    # 下面调用你之前实现的 multiprocessing worker_run 方案即可

    from multiprocessing import get_context
    ctx = get_context("spawn")

    base = n_games // N_WORKERS
    rem = n_games % N_WORKERS
    ranges = []
    start = 0
    for wid in range(N_WORKERS):
        cnt = base + (1 if wid < rem else 0)
        end = start + cnt
        ranges.append((start, end))
        start = end

    with ctx.Pool(processes=N_WORKERS) as pool:
        jobs = []
        for wid, (gs, ge) in enumerate(ranges):
            jobs.append(pool.apply_async(
                worker_run,
                (wid, out_dir, global_seed, gs, ge, chunk_size, think_time, eps, temp, debug_chunk)
            ))

        win_stat = {0: 0, 1: 0, None: 0}
        total_steps = 0
        for j in jobs:
            ws, steps = j.get()
            for k in win_stat:
                win_stat[k] += ws.get(k, 0)
            total_steps += steps

    elapsed = time.perf_counter() - t0
    avg = elapsed / max(1, total_steps)
    print("[done] win_stat =", win_stat)
    print(f"[done] total_steps={total_steps} total_time={_fmt_hms(elapsed)} avg={avg:.4f}s/step")


if __name__ == "__main__":
    main()
