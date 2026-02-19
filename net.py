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
    # ----------------------------
    # ResNet building blocks
    # ----------------------------
    class ResidualBlock(nn.Module):
        """
        Standard AlphaZero-style residual block:
          x -> Conv-BN-ReLU -> Conv-BN -> +x -> ReLU
        """
        def __init__(self, channels: int):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(channels)

        def forward(self, x):
            out = self.conv1(x)
            out = self.bn1(out)
            out = F.relu(out, inplace=True)

            out = self.conv2(out)
            out = self.bn2(out)

            out = out + x
            out = F.relu(out, inplace=True)
            return out

    class PolicyValueNet(nn.Module):
        """
        Lightweight AlphaZero-style Policy/Value network.

        Input:  x [B, C=33, H=10, W=9]
        Output: logits [B, ACTION_SIZE], value [B, 1] (tanh range in [-1,1])

        Keep interface compatible with your existing scripts:
            PolicyValueNet(action_size=ACTION_SIZE, in_ch=33)
        """
        def __init__(
            self,
            action_size: int = ACTION_SIZE,
            in_ch: int = 33,
            board_h: int = 10,
            board_w: int = 9,
            channels: int = 64,
            n_blocks: int = 6,          # 6~8 usually enough for 10x9 board
            policy_channels: int = 32,  # policy head width
            value_channels: int = 32,   # value head width
            value_hidden: int = 128,    # value head MLP hidden
        ):
            super().__init__()
            self.action_size = int(action_size)
            self.board_h = int(board_h)
            self.board_w = int(board_w)

            # trunk stem
            self.conv_in = nn.Conv2d(in_ch, channels, kernel_size=3, padding=1, bias=False)
            self.bn_in = nn.BatchNorm2d(channels)

            # residual tower
            self.blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(int(n_blocks))])

            # ----------------------------
            # policy head
            # ----------------------------
            self.policy_conv = nn.Conv2d(channels, policy_channels, kernel_size=1, bias=False)
            self.policy_bn = nn.BatchNorm2d(policy_channels)
            self.policy_fc = nn.Linear(policy_channels * board_h * board_w, self.action_size)

            # ----------------------------
            # value head
            # ----------------------------
            self.value_conv = nn.Conv2d(channels, value_channels, kernel_size=1, bias=False)
            self.value_bn = nn.BatchNorm2d(value_channels)
            self.value_fc1 = nn.Linear(value_channels * board_h * board_w, value_hidden)
            self.value_fc2 = nn.Linear(value_hidden, 1)

            # init
            self._init_weights()

        def _init_weights(self):
            # Kaiming init for conv, xavier for linear
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, x):
            # trunk
            x = self.conv_in(x)
            x = self.bn_in(x)
            x = F.relu(x, inplace=True)

            for blk in self.blocks:
                x = blk(x)

            # policy head
            p = self.policy_conv(x)
            p = self.policy_bn(p)
            p = F.relu(p, inplace=True)
            p = p.view(p.size(0), -1)
            logits = self.policy_fc(p)

            # value head
            v = self.value_conv(x)
            v = self.value_bn(v)
            v = F.relu(v, inplace=True)
            v = v.view(v.size(0), -1)
            v = F.relu(self.value_fc1(v), inplace=True)
            v = torch.tanh(self.value_fc2(v))  # [-1,1]
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

            # softmax only over legal actions
            legal_logits = np.array([logits[a] for a in legal_actions], dtype=np.float32)
            legal_logits -= legal_logits.max()
            probs = np.exp(legal_logits)
            probs /= probs.sum() + 1e-12

            return list(zip(legal_actions, probs.tolist())), v

    def train_one_step(model, optimizer, batch_S, batch_P, batch_Z, device="cpu", value_weight=1.0):
        """
        Kept for backward compatibility with your older training loop.
        """
        model.train()
        S = torch.tensor(batch_S, dtype=torch.float32, device=device)   # [B,C,10,9]
        P = torch.tensor(batch_P, dtype=torch.float32, device=device)   # [B,A]
        Z = torch.tensor(batch_Z, dtype=torch.float32, device=device)   # [B]

        logits, v = model(S)  # logits [B,A], v [B,1]
        v = v.view(-1)

        # normalize P defensively
        s = P.sum(dim=1, keepdim=True)
        P = torch.where(s > 0, P / (s + 1e-12), torch.full_like(P, 1.0 / P.shape[1]))

        logp = F.log_softmax(logits, dim=1)
        policy_loss = -(P * logp).sum(dim=1).mean()

        # v already tanh in forward, Z in {-1,0,1}
        value_loss = F.mse_loss(v, Z)

        loss = policy_loss + value_weight * value_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        return float(loss.item()), float(policy_loss.item()), float(value_loss.item())

else:
    PolicyValueNet = None
    TorchPolicyValueWrapper = None
