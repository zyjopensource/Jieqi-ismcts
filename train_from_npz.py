# train_from_npz.py
import os
import glob
import time
import argparse
from typing import Iterable, List, Tuple, Optional
import numpy as np

import torch
import torch.nn.functional as F

from actions import ACTION_SIZE
from net import PolicyValueNet


# ----------------------------
# utils
# ----------------------------
def unpack_model_output(out):
    if isinstance(out, (tuple, list)) and len(out) == 2:
        logits, v = out
        return logits, v
    if isinstance(out, dict) and "logits" in out and "value" in out:
        return out["logits"], out["value"]
    raise RuntimeError("PolicyValueNet forward output must be (logits, value) or dict with keys logits/value")


def list_npz_files(data_dir: str, pattern: str = "*.npz") -> List[str]:
    files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No .npz found in {data_dir} with pattern={pattern}")
    return files


def policy_loss_soft_targets(logits: torch.Tensor, target_probs: torch.Tensor):
    logp = F.log_softmax(logits, dim=1)
    return -(target_probs * logp).sum(dim=1).mean()


def normalize_target_pi(target_pi: torch.Tensor) -> torch.Tensor:
    # 防御：确保 target_pi 每行和为1；若全0则退化为均匀
    s = target_pi.sum(dim=1, keepdim=True)
    uniform = torch.full_like(target_pi, 1.0 / target_pi.shape[1])
    return torch.where(s > 0, target_pi / (s + 1e-12), uniform)


# ----------------------------
# data iterator with simple prefetch
# ----------------------------
def iter_minibatches_from_files(
    files: List[str],
    batch_size: int,
    shuffle_files: bool,
    shuffle_in_file: bool,
    seed: int,
    prefetch_files: int = 0,
):
    """
    逐文件读取，逐 batch yield，避免一次性把所有数据载入内存。
    prefetch_files>0 时，会提前在内存中预取后续若干文件（减少IO/解压间隙）。
    """
    rng = np.random.default_rng(seed)
    file_list = list(files)
    if shuffle_files:
        rng.shuffle(file_list)

    # 简单预取：用一个队列缓存若干已 np.load 的文件内容
    cache = []
    ptr = 0

    def _load_one(path):
        data = np.load(path)
        S = data["S"]  # [N,C,10,9]
        P = data["P"]  # [N,A]
        Z = data["Z"]  # [N]
        return S, P, Z

    while ptr < len(file_list) or cache:
        # fill cache
        while prefetch_files > 0 and ptr < len(file_list) and len(cache) < prefetch_files:
            path = file_list[ptr]
            cache.append((path, _load_one(path)))
            ptr += 1

        # if no prefetch, load directly
        if prefetch_files <= 0:
            path = file_list[ptr]
            S, P, Z = _load_one(path)
            ptr += 1
        else:
            path, (S, P, Z) = cache.pop(0)

        n = S.shape[0]
        idx = np.arange(n)
        if shuffle_in_file:
            rng.shuffle(idx)

        for i in range(0, n, batch_size):
            j = idx[i:i + batch_size]
            yield S[j], P[j], Z[j]


# ----------------------------
# main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data_teacher")
    ap.add_argument("--pattern", type=str, default="*.npz", help="glob pattern inside data_dir")
    ap.add_argument("--ckpt_dir", type=str, default="checkpoints_teacher_pretrain")
    ap.add_argument("--resume", type=str, default="", help="path to checkpoint to resume")

    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--grad_accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--value_weight", type=float, default=1.0)
    ap.add_argument("--clip_grad", type=float, default=5.0)

    ap.add_argument("--shuffle_files", action="store_true")
    ap.add_argument("--no_shuffle_in_file", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--save_every", type=int, default=1)
    ap.add_argument("--log_every", type=int, default=50)

    ap.add_argument("--amp", action="store_true", help="use torch.cuda.amp mixed precision")
    ap.add_argument("--prefetch_files", type=int, default=0, help="prefetch N files into memory")
    args = ap.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[device]", device)
    if device == "cuda":
        print("[cuda] name =", torch.cuda.get_device_name(0))

    files = list_npz_files(args.data_dir, args.pattern)
    print(f"[data] {len(files)} files from {args.data_dir} pattern={args.pattern}")

    # model
    model = PolicyValueNet(action_size=ACTION_SIZE).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device == "cuda"))

    start_ep = 1
    global_step = 0

    # resume
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        start_ep = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        print(f"[resume] from {args.resume} start_ep={start_ep} global_step={global_step}")

    shuffle_in_file = not args.no_shuffle_in_file

    effective_bs = args.batch_size * max(1, args.grad_accum)
    print(f"[train] batch_size={args.batch_size} grad_accum={args.grad_accum} effective_batch={effective_bs}")

    for ep in range(start_ep, args.epochs + 1):
        t_ep0 = time.time()
        model.train()

        losses, plosses, vlosses = [], [], []
        n_samples = 0
        t_data = 0.0
        t_iter0 = time.time()

        opt.zero_grad(set_to_none=True)

        it = iter_minibatches_from_files(
            files,
            batch_size=args.batch_size,
            shuffle_files=args.shuffle_files,
            shuffle_in_file=shuffle_in_file,
            seed=args.seed + ep,
            prefetch_files=args.prefetch_files,
        )

        for b_idx, (bS, bP, bZ) in enumerate(it, start=1):
            t_load_done = time.time()

            x = torch.tensor(bS, dtype=torch.float32, device=device)         # [B,C,10,9]
            target_pi = torch.tensor(bP, dtype=torch.float32, device=device) # [B,A]
            target_z = torch.tensor(bZ, dtype=torch.float32, device=device)  # [B]

            target_pi = normalize_target_pi(target_pi)

            with torch.cuda.amp.autocast(enabled=(args.amp and device == "cuda")):
                logits, v = unpack_model_output(model(x))
                v = v.view(-1)

                pl = policy_loss_soft_targets(logits, target_pi)
                # value head：你现在用 tanh 把输出夹到 [-1,1] 是合理的
                vl = F.mse_loss(v.view(-1), target_z)
                loss = pl + args.value_weight * vl

                # grad accum: 让每个 micro-step 的 loss 缩小
                loss = loss / max(1, args.grad_accum)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # step when reached grad_accum
            if b_idx % max(1, args.grad_accum) == 0:
                if args.clip_grad is not None and args.clip_grad > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

                if scaler.is_enabled():
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)

            bs = x.shape[0]
            n_samples += bs
            global_step += 1

            losses.append(loss.item() * max(1, args.grad_accum))
            plosses.append(pl.item())
            vlosses.append(vl.item())

            t_now = time.time()
            # 估算 IO/解压耗时（粗略）：上一轮到本轮 tensor 化之前的时间
            t_data += (t_load_done - t_iter0)
            t_iter0 = t_now

            if global_step % args.log_every == 0:
                dt = t_now - t_ep0
                sps = n_samples / max(1e-9, dt)  # samples per sec
                print(f"[ep {ep}] step={global_step} samples={n_samples} "
                      f"loss={np.mean(losses):.4f} policy={np.mean(plosses):.4f} value={np.mean(vlosses):.4f} "
                      f"samples/s={sps:.1f}",
                      flush=True)

        dt_ep = time.time() - t_ep0
        print(f"[epoch {ep}/{args.epochs}] time={dt_ep:.1f}s "
              f"loss={np.mean(losses):.4f} policy={np.mean(plosses):.4f} value={np.mean(vlosses):.4f}",
              flush=True)

        if ep % args.save_every == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"pretrain_ep{ep}.pth")
            torch.save({
                "epoch": ep,
                "global_step": global_step,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "action_size": ACTION_SIZE,
                "args": vars(args),
            }, ckpt_path)
            print("[saved]", ckpt_path, flush=True)

    final_path = os.path.join(args.ckpt_dir, "pretrain_final.pth")
    torch.save({
        "epoch": args.epochs,
        "global_step": global_step,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "action_size": ACTION_SIZE,
        "args": vars(args),
    }, final_path)
    print("[saved]", final_path, flush=True)


if __name__ == "__main__":
    main()
