# net.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_OK = True
except Exception:
    TORCH_OK = False

from actions import ACTION_SIZE

class RandomPolicyValueNet:
    def policy_value(self, obs_planes: np.ndarray, legal_actions: List[int]) -> Tuple[List[Tuple[int, float]], float]:
        # uniform over legal actions, value=0
        if not legal_actions:
            return [], 0.0
        p = 1.0 / len(legal_actions)
        return [(a, p) for a in legal_actions], 0.0

if TORCH_OK:
    class TinyPolicyValueNet(nn.Module):
        def __init__(self, in_ch: int, board_h: int = 10, board_w: int = 9, n_actions: int = ACTION_SIZE):
            super().__init__()
            self.board_h, self.board_w = board_h, board_w
            self.conv1 = nn.Conv2d(in_ch, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 64, 3, padding=1)

            self.policy_head = nn.Conv2d(64, 2, 1)
            self.policy_fc = nn.Linear(2 * board_h * board_w, n_actions)

            self.value_head = nn.Conv2d(64, 1, 1)
            self.value_fc1 = nn.Linear(board_h * board_w, 64)
            self.value_fc2 = nn.Linear(64, 1)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))

            # policy
            p = F.relu(self.policy_head(x))
            p = p.view(p.size(0), -1)
            logits = self.policy_fc(p)

            # value
            v = F.relu(self.value_head(x))
            v = v.view(v.size(0), -1)
            v = F.relu(self.value_fc1(v))
            v = torch.tanh(self.value_fc2(v))
            return logits, v

    class TorchPolicyValueWrapper:
        def __init__(self, in_ch: int = 33, device: str = "cpu"):
            self.device = device
            self.net = PolicyValueNet(action_size=ACTION_SIZE, in_ch=in_ch).to(device)
            self.net.eval()

        @torch.no_grad()
        def policy_value(self, obs_planes: np.ndarray, legal_actions: List[int]):
            if not legal_actions:
                return [], 0.0
            x = torch.from_numpy(obs_planes[None, ...]).float().to(self.device)  # [1,C,H,W]
            logits, v = self.net(x)
            logits = logits[0].cpu().numpy()
            v = float(v[0].cpu().numpy())

            # softmax over all actions then filter legal
            # 为简化：只在 legal actions 上做 softmax
            legal_logits = np.array([logits[a] for a in legal_actions], dtype=np.float32)
            legal_logits -= legal_logits.max()
            probs = np.exp(legal_logits)
            probs /= probs.sum() + 1e-12

            return list(zip(legal_actions, probs.tolist())), v
        
    def train_one_step(model, optimizer, batch_S, batch_P, batch_Z, device="cpu", value_weight=1.0):
        model.train()
        S = torch.tensor(batch_S, dtype=torch.float32, device=device)   # [B,C,10,9]
        P = torch.tensor(batch_P, dtype=torch.float32, device=device)   # [B,A]
        Z = torch.tensor(batch_Z, dtype=torch.float32, device=device)   # [B]
    
        logits, v = model(S)  # logits [B,A], v [B] or [B,1]
        if v.dim() == 2:
            v = v.squeeze(1)
    
        logp = F.log_softmax(logits, dim=1)
        policy_loss = -(P * logp).sum(dim=1).mean()
    
        value_loss = F.mse_loss(v, Z)
    
        loss = policy_loss + value_weight * value_loss
    
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
        return float(loss.item()), float(policy_loss.item()), float(value_loss.item())

    class PolicyValueNet(TinyPolicyValueNet):
        """
        Backward-compatible wrapper so training scripts can do:
            PolicyValueNet(action_size=ACTION_SIZE, in_ch=33)
        """
        def __init__(self, action_size: int = ACTION_SIZE, in_ch: int = 33,
                     board_h: int = 10, board_w: int = 9):
            super().__init__(in_ch=in_ch, board_h=board_h, board_w=board_w, n_actions=action_size)


else:
    TorchPolicyValueWrapper = None
