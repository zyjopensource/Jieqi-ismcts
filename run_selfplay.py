# run_selfplay.py
from __future__ import annotations
import numpy as np
import random

from state import JieqiState
from mcts import ISMCTS
from net import RandomPolicyValueNet, TorchPolicyValueWrapper, TORCH_OK
from actions import ACTION_SIZE

def self_play_one_game(seed: int = 0, n_playout: int = 200, temp: float = 1.0):
    rng = random.Random(seed)
    state = JieqiState(seed=seed)

    # choose net
    if TORCH_OK:
        net = TorchPolicyValueWrapper(in_ch=33, device="cpu")
        policy_value_fn = net.policy_value
    else:
        net = RandomPolicyValueNet()
        policy_value_fn = net.policy_value

    mcts = ISMCTS(policy_value_fn, c_puct=5.0, n_playout=n_playout, seed=seed)

    states, mcts_probs, players = [], [], []

    while True:
        ended, winner = state.game_end()
        if ended:
            # z from each stored player's perspective
            zs = []
            for p in players:
                if winner is None:
                    zs.append(0.0)
                else:
                    zs.append(1.0 if winner == p else -1.0)
            return states, mcts_probs, zs, winner

        current_player = state.current_player
        obs = state.obs_planes(current_player)

        acts, probs = mcts.get_action_probs(state, temp=temp)

        # build full π over ACTION_SIZE
        pi = np.zeros((ACTION_SIZE,), dtype=np.float32)
        for a, pr in zip(acts, probs):
            pi[a] = pr

        states.append(obs)
        mcts_probs.append(pi)
        players.append(current_player)

        # sample action by probs (temp>0)
        a = rng.choices(acts, weights=probs, k=1)[0]
        state.apply_action(a)
        mcts.update_with_move(a)

def main():
    all_samples = 0
    for g in range(2):
        S, P, Z, winner = self_play_one_game(seed=1234 + g, n_playout=100, temp=1.0)
        print(f"game {g}: steps={len(S)}, winner={winner}")
        all_samples += len(S)

    print("total samples:", all_samples)
    # 可选：保存为 npz
    # np.savez("selfplay_data.npz", states=np.array(S), probs=np.array(P), zs=np.array(Z))

if __name__ == "__main__":
    main()
