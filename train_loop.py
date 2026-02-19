# train_loop.py
import os
import random
import numpy as np
import torch

from state import JieqiState
from actions import ACTION_SIZE
from replay_buffer import ReplayBuffer

from mcts import ISMCTS
from net import PolicyValueNet, train_one_step

from teacher_musesfish import MusefishTeacher
from agents import NnMctsAgent  # 你刚刚新建的 agents.py

# ----------------------------
# schedule: teacher -> mix -> nn
# ----------------------------
def teacher_ratio(it: int, warm_iters: int = 30, mix_iters: int = 30) -> float:
    """
    返回 teacher 的使用概率
    it: 1..n_iterations
    """
    if it <= warm_iters:
        return 1.0
    if it <= warm_iters + mix_iters:
        # linearly decay 1 -> 0
        return 1.0 - (it - warm_iters) / float(mix_iters)
    return 0.0

# ----------------------------
# policy_value_fn wrapper (for ISMCTS)
# ----------------------------
class PolicyValueWrapper:
    def __init__(self, model: PolicyValueNet, device: str = "cpu"):
        self.model = model
        self.device = device

    @torch.no_grad()
    def __call__(self, obs: np.ndarray, legal: list[int]):
        """
        obs: [C,10,9]
        legal: list of action_id
        return: (action_priors, leaf_value)
          action_priors: List[(aid, prob)] only over legal
          leaf_value: float in [-1,1] from current player perspective
        """
        x = torch.tensor(obs[None, ...], dtype=torch.float32, device=self.device)  # [1,C,10,9]
        logits, v = self.model(x)  # logits [1,A], v [1] or [1,1]
        logits = logits[0]
        v = v.view(-1)[0].item()

        # softmax over legal only
        legal_t = torch.tensor(legal, dtype=torch.long, device=self.device)
        legal_logits = logits.index_select(0, legal_t)
        probs = torch.softmax(legal_logits, dim=0).detach().cpu().numpy()

        action_priors = list(zip(legal, probs.tolist()))
        leaf_value = float(v)
        return action_priors, leaf_value

# ----------------------------
# self-play
# ----------------------------
def self_play_one_game(
    it: int,
    seed: int,
    teacher: MusefishTeacher,
    nn_agent: NnMctsAgent,
    temp: float = 1.0,
    warm_iters: int = 30,
    mix_iters: int = 30,
):
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    state = JieqiState(seed=seed)

    # 每局开始重置 NN 的 MCTS 树（非常重要）
    nn_agent.update_with_move(None)

    S_list, P_list, players = [], [], []

    while True:
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
            return S_list, P_list, Z_list, winner

        cur = state.current_player
        obs = state.obs_planes(cur)

        r = teacher_ratio(it, warm_iters=warm_iters, mix_iters=mix_iters)
        use_teacher = (rng.random() < r)

        if use_teacher:
            acts, probs = teacher.get_action_probs(state, temp=temp)
        else:
            acts, probs = nn_agent.get_action_probs(state, temp=temp)

        if len(acts) == 0:
            # 无合法着：判和（或判负都行，你可自行调整）
            Z_list = [0.0] * len(players)
            return S_list, P_list, Z_list, None

        # full pi over ACTION_SIZE
        pi = np.zeros((ACTION_SIZE,), dtype=np.float32)
        pi[np.array(acts, dtype=np.int64)] = np.array(probs, dtype=np.float32)

        S_list.append(obs)
        P_list.append(pi)
        players.append(cur)

        # sample action
        aid = int(np_rng.choice(np.array(acts, dtype=np.int64), p=np.array(probs, dtype=np.float64)))
        state.apply_action(aid)

        # 更新/重置 MCTS tree
        if use_teacher:
            # teacher 走了一步，NN 的树和真实走法不一致，直接重置
            nn_agent.update_with_move(None)
        else:
            nn_agent.update_with_move(aid)

# ----------------------------
# main train loop
# ----------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- hyperparams ----
    n_iterations = 200
    games_per_iter = 5

    n_playout = 200          # ISMCTS playouts per move (NN阶段用)
    c_puct = 5.0
    temp = 1.0               # 训练期一般 temp=1; 后期可降低

    buffer_size = 50000
    batch_size = 256
    train_steps_per_iter = 50
    lr = 1e-3

    warm_iters = 30
    mix_iters = 30

    teacher_think_time = 0.05
    teacher_eps = 0.02

    max_plies = 400

    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- model / optim ----
    model = PolicyValueNet(action_size=ACTION_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # ---- teacher / nn mcts ----
    teacher = MusefishTeacher(think_time=teacher_think_time, eps=teacher_eps, seed=0)

    pv = PolicyValueWrapper(model, device=device)
    ismcts = ISMCTS(policy_value_fn=pv, c_puct=c_puct, n_playout=n_playout, seed=0)
    nn_agent = NnMctsAgent(ismcts)

    # ---- buffer ----
    buffer = ReplayBuffer(capacity=buffer_size, seed=0)

    global_seed = 1234

    for it in range(1, n_iterations + 1):
        # 1) self-play collect
        for g in range(games_per_iter):
            seed = global_seed + it * 1000 + g
            S, P, Z, winner = self_play_one_game(
                it=it,
                seed=seed,
                teacher=teacher,
                nn_agent=nn_agent,
                temp=temp,
                max_plies=max_plies,
                warm_iters=warm_iters,
                mix_iters=mix_iters,
            )
            buffer.add_game(S, P, Z)

        # 2) train
        if len(buffer) >= batch_size:
            losses = []
            for _ in range(train_steps_per_iter):
                bS, bP, bZ = buffer.sample(batch_size)
                loss, pl, vl = train_one_step(model, optimizer, bS, bP, bZ, device=device)
                losses.append((loss, pl, vl))
            m = np.mean(np.array(losses, dtype=np.float32), axis=0)
            tr = teacher_ratio(it, warm_iters=warm_iters, mix_iters=mix_iters)
            print(f"[it {it}] teacher_ratio={tr:.2f} buffer={len(buffer)} loss={m[0]:.4f} policy={m[1]:.4f} value={m[2]:.4f}")
        else:
            print(f"[it {it}] buffer={len(buffer)} (warming up)")

        # 3) save checkpoint
        if it % 10 == 0:
            path = os.path.join(ckpt_dir, f"model_it{it}.pth")
            torch.save({"it": it, "model": model.state_dict(), "opt": optimizer.state_dict()}, path)

if __name__ == "__main__":
    main()
