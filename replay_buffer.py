# replay_buffer.py
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        self.buf = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def add_game(self, S_list, P_list, Z_list):
        for s, p, z in zip(S_list, P_list, Z_list):
            self.buf.append((s, p, z))

    def __len__(self):
        return len(self.buf)

    def sample(self, batch_size: int):
        batch = self.rng.sample(self.buf, batch_size)
        S, P, Z = zip(*batch)
        return list(S), list(P), list(Z)
