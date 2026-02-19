# mcts.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import math
import numpy as np
import random

from constants import other
from state import JieqiState

@dataclass
class TreeNode:
    parent: Optional["TreeNode"]
    prior_p: float
    n_visits: int = 0
    q: float = 0.0
    children: Dict[int, "TreeNode"] = None  # action_id -> TreeNode

    def __post_init__(self):
        if self.children is None:
            self.children = {}

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def expand(self, action_priors: List[Tuple[int, float]]):
        for aid, p in action_priors:
            if aid not in self.children:
                self.children[aid] = TreeNode(parent=self, prior_p=float(p))

    def select(self, c_puct: float) -> Tuple[int, "TreeNode"]:
        best_a, best_node, best_val = None, None, -1e18
        for aid, node in self.children.items():
            u = c_puct * node.prior_p * math.sqrt(self.n_visits + 1e-8) / (1 + node.n_visits)
            val = node.q + u
            if val > best_val:
                best_val = val
                best_a, best_node = aid, node
        return best_a, best_node

    def update(self, leaf_value: float):
        self.n_visits += 1
        # incremental mean
        self.q += (leaf_value - self.q) / self.n_visits

    def update_recursive(self, leaf_value: float):
        if self.parent is not None:
            # value is always from current player's perspective at node; switching player flips sign
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)

class ISMCTS:
    def __init__(self, policy_value_fn, c_puct: float = 5.0, n_playout: int = 200, seed: int = 0):
        self.root = TreeNode(parent=None, prior_p=1.0)
        self.policy_value_fn = policy_value_fn
        self.c_puct = c_puct
        self.n_playout = n_playout
        self.rng = random.Random(seed)

    def _playout(self, root_state: JieqiState, root_player: int):
        # 关键：每次 playout 先 determinize_for(root_player)
        state = root_state.determinize_for(root_player, seed=self.rng.randint(0, 10**9))
        node = self.root

        while True:
            ended, winner = state.game_end()
            if ended:
                if winner is None:
                    leaf_value = 0.0
                else:
                    leaf_value = 1.0 if winner == state.current_player else -1.0
                node.update_recursive(leaf_value)
                return

            if node.is_leaf():
                legal = state.legal_actions()
                obs = state.obs_planes(state.current_player)
                action_priors, leaf_value = self.policy_value_fn(obs, legal)
                node.expand(action_priors)
                node.update_recursive(leaf_value)
                return

            # select
            aid, node = node.select(self.c_puct)
            state.apply_action(aid)

    def get_action_probs(self, state: JieqiState, temp: float = 1e-3) -> Tuple[List[int], List[float]]:
        root_player = state.current_player
        for _ in range(self.n_playout):
            self._playout(state, root_player)

        # collect visits
        acts = list(self.root.children.keys())
        visits = np.array([self.root.children[a].n_visits for a in acts], dtype=np.float32)

        if temp <= 1e-6:
            best = int(np.argmax(visits))
            probs = np.zeros_like(visits)
            probs[best] = 1.0
        else:
            x = visits ** (1.0 / temp)
            probs = x / (x.sum() + 1e-12)

        return acts, probs.tolist()

    def update_with_move(self, last_action: Optional[int]):
        if last_action is not None and last_action in self.root.children:
            self.root = self.root.children[last_action]
            self.root.parent = None
        else:
            self.root = TreeNode(parent=None, prior_p=1.0)
