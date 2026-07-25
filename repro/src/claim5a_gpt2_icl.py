#!/usr/bin/env python3
"""CLAIM 5a (Section 5.1) — GPT-2-architecture ICL: non-monotone per-task error.

Exact claim: decoder-style GPT-2 transformers trained on linear-regression ICL
exhibit NON-MONOTONIC per-task prediction error as a function of context length M
in the multi-task (in-context continual learning) setting — later tasks show a
peak at intermediate M (e.g. M=3) before recovering — reproducing the theoretical
prediction of Theorem 4.3 / Claim 3 in a real softmax-attention model.

Implementation (clean-room, CPU-only):
  - A GPT-2-style decoder (tiny config: 3 layers, 2 heads, d_model=64, ~0.2M params;
    a larger 6-layer/4-head/128 config is also supported) with scalar read-in/out,
    trained on single-task linear-regression ICL (Garg et al. 2022 setup: w~N(0,I_d),
    x~N(0,I_d), y=w^T x, identity covariance), AdamW, curriculum on the training
    prompt length.
  - Evaluation: multi-task continual prompts (T=5 tasks), per-task query MSE vs M,
    M in {1,2,3,5,8,12,16,20}. Task 1 should decrease monotonically; later tasks
    should be non-monotone (interior peak).
Verdict logic: VERIFIED if Task-1 is monotone-decreasing AND at least one later task
shows a clear interior peak; else reports the observed curve honestly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)


# ---------------------------------------------------------------------------------------------
# Minimal GPT-2-style decoder for scalar ICL regression.
# ----------------------------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.head = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]           # (B, nh, T, h)
        att = (q @ k.transpose(-2, -1)) / (self.head ** 0.5)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(mask, float("-inf"))
        att = self.drop(torch.softmax(att, dim=-1))
        y = att @ v                                   # (B, nh, T, h)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, mlp_ratio * d_model),
                                 nn.GELU(), nn.Linear(mlp_ratio * d_model, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class GPT2ICL(nn.Module):
    def __init__(self, d_model=64, depth=3, n_heads=2, max_len=2048):
        super().__init__()
        self.readin = nn.Linear(1, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(depth)])
        self.ln_f = nn.LayerNorm(d_model)
        self.readout = nn.Linear(d_model, 1)
        nn.init.normal_(self.pos.weight, std=0.02)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, seq, positions):
        # seq: (B, L) scalars ; positions: (B, P) indices to read predictions from
        x = self.readin(seq.unsqueeze(-1)) + self.pos(torch.arange(seq.size(1), device=seq.device))[None]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.readout(x).squeeze(-1)         # (B, L)
        return logits.gather(1, positions)           # (B, P)


# ---------------------------------------------------------------------------------------------
# Data: build single-task and multi-task scalar prompts.
# ----------------------------------------------------------------------------------------------

def build_single(w, x, y, xq):
    """w: (d,), x: (M,d), y: (M,), xq: (d,) -> (seq, pred_positions, targets)."""
    M, d = x.shape
    blocks = [np.concatenate([x[i], [y[i]]]) for i in range(M)]
    seq = np.concatenate(blocks + [xq])              # len M*(d+1)+d
    ex_pos = [i * (d + 1) + (d - 1) for i in range(M)]
    q_pos = M * (d + 1) + (d - 1)
    pos = ex_pos + [q_pos]
    targets = list(y) + [float(w @ xq)]
    return seq.astype(np.float32), np.array(pos, dtype=np.int64), np.array(targets, dtype=np.float32)


def sample_single_batch(rng, M, d, B):
    W = rng.standard_normal((B, d))
    X = rng.standard_normal((B, M, d))
    Y = np.einsum("bd,bmd->bm", W, X)
    Xq = rng.standard_normal((B, d))
    seqs, pos, tgt = [], [], []
    for b in range(B):
        s, p, t = build_single(W[b], X[b], Y[b], Xq[b])
        seqs.append(s); pos.append(p); tgt.append(t)
    L = len(s)
    seqs = np.stack([np.pad(s, (0, L - len(s))) for s in seqs]) if False else np.stack(seqs)
    return (torch.tensor(np.stack(seqs)), torch.tensor(np.stack(pos)),
            torch.tensor(np.stack(tgt)), W, Y, Xq)


def sample_multitask_batch(rng, M, d, T, B):
    """Multi-task continual prompt: T tasks each with M examples + own query."""
    seqs, pos_list, tgt_list, task_of_q = [], [], [], []
    for b in range(B):
        full_seq, full_pos, full_tgt = [], [], []
        qpos_by_task = []
        cursor = 0
        for t in range(T):
            w = rng.standard_normal(d)
            x = rng.standard_normal((M, d))
            y = x @ w
            xq = rng.standard_normal(d)
            s, p, tg = build_single(w, x, y, xq)
            full_seq.append(s)
            full_pos.append(p + cursor)
            full_tgt.append(tg)
            qpos_by_task.append(cursor + p[-1])     # query position for task t
            cursor += len(s)
        seqs.append(np.concatenate(full_seq))
        pos_list.append(np.concatenate(full_pos))
        tgt_list.append(np.concatenate(full_tgt))
        task_of_q.append(qpos_by_task)
    Lmax = max(len(s) for s in seqs)
    seq = torch.tensor(np.stack([np.pad(s, (0, Lmax - len(s))) for s in seqs]))
    pos = torch.tensor(np.stack(pos_list))
    tgt = torch.tensor(np.stack(tgt_list))
    return seq, pos, tgt, task_of_q, Lmax


# ---------------------------------------------------------------------------------------------
# Train + evaluate.
# ----------------------------------------------------------------------------------------------

def train(args):
    set_seed(args.seed)
    device = "cpu"
    torch.set_num_threads(args.threads)
    model = GPT2ICL(d_model=args.d_model, depth=args.depth, n_heads=args.heads)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GPT-2 ICL model: {args.depth}L/{args.heads}H/d_model={args.d_model}, params={n_params/1e6:.3f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    rng = np.random.default_rng(args.seed)
    d = args.dim
    t0 = time.time()
    losses = []
    warmup = min(500, args.steps // 20)
    for step in range(1, args.steps + 1):
        # linear warmup then cosine decay
        if step <= warmup:
            lr = args.lr * step / warmup
        else:
            prog = (step - warmup) / max(1, args.steps - warmup)
            lr = args.lr * 0.5 * (1 + np.cos(np.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr
        Mtr = int(rng.integers(args.m_train_min, args.m_train_max + 1))
        seq, pos, tgt, *_ = sample_single_batch(rng, Mtr, d, args.batch)
        opt.zero_grad()
        pred = model(seq, pos)
        # loss on example y-positions only (positions 0..M-1), not the query slot
        loss = F.mse_loss(pred[:, :-1], tgt[:, :-1])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0 or step == 1:
            losses.append({"step": step, "loss": float(loss.detach()), "M_train": Mtr})
            dt = time.time() - t0
            print(f"step {step:>6}/{args.steps}  loss={float(loss.detach()):.4f}  M_train={Mtr}  "
                  f"({dt/step*1000:.1f}ms/step, {dt:.0f}s elapsed)", flush=True)
        # time-budget stop (adaptive to CPU speed): leave room for eval
        if step >= args.min_steps and (time.time() - t0) > args.time_budget_sec:
            print(f"time budget {args.time_budget_sec}s reached at step {step}; stopping training", flush=True)
            break
    return model, losses, n_params


def evaluate(model, args):
    device = "cpu"
    model.eval()
    rng = np.random.default_rng(args.seed + 777)
    T = args.T_eval
    d = args.dim
    results = {}
    with torch.no_grad():
        for M in args.M_eval:
            per_task = [[] for _ in range(T)]
            n_batches = args.eval_batches
            for _ in range(n_batches):
                seq, pos, tgt, task_of_q, Lmax = sample_multitask_batch(rng, M, d, T, args.eval_batch)
                pred = model(seq, pos)              # (B, T*(M+1)); query slot of task t is at index t*(M+1)+M
                q_idx = [t * (M + 1) + M for t in range(T)]
                qpred = pred[:, q_idx]              # (B, T)
                qtgt = tgt[:, q_idx]                # (B, T)
                # normalized MSE per task: divide by d = E[y_q^2] (w,x ~ N(0,I))
                for t in range(T):
                    sq = ((qpred[:, t] - qtgt[:, t]) ** 2).numpy()
                    per_task[t].append(sq)
            mses = [float(np.concatenate(pt).mean()) for pt in per_task]
            # normalized MSE (divide by d = E[y_q^2] since w,x ~ N(0,I))
            norm = [m / d for m in mses]
            results[M] = {"per_task_mse": mses, "per_task_nmse": norm}
            print(f"  M={M:>2}  per-task NMSE: " + "  ".join(f"t{i+1}={n:.3f}" for i, n in enumerate(norm)), flush=True)
    return results


def classify(results):
    """Task1 monotone decreasing? any later task interior peak?"""
    Ms = sorted(results.keys())
    t1 = [results[M]["per_task_nmse"][0] for M in Ms]
    t1_mono = all(t1[i] >= t1[i + 1] - 1e-6 for i in range(len(t1) - 1))
    later_peaks = []
    for t in range(1, len(results[Ms[0]]["per_task_nmse"])):
        curve = [results[M]["per_task_nmse"][t] for M in Ms]
        am = int(np.argmax(curve))
        peaked = (0 < am < len(curve) - 1) and curve[am] > curve[0] + 1e-4 and curve[am] > curve[-1] + 1e-4
        later_peaks.append({"task": t + 1, "argmax_M": Ms[am], "peaked": bool(peaked),
                            "start": curve[0], "peak": curve[am], "end": curve[-1]})
    any_peak = any(p["peaked"] for p in later_peaks)
    return {"task1_monotone_decreasing": bool(t1_mono), "later_tasks": later_peaks,
            "any_later_task_interior_peak": bool(any_peak)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--min-steps", dest="min_steps", type=int, default=3000)
    ap.add_argument("--time-budget-sec", dest="time_budget_sec", type=int, default=9000)
    ap.add_argument("--dim", type=int, default=5)
    ap.add_argument("--d-model", dest="d_model", type=int, default=64)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--m-train-min", dest="m_train_min", type=int, default=4)
    ap.add_argument("--m-train-max", dest="m_train_max", type=int, default=20)
    ap.add_argument("--T-eval", dest="T_eval", type=int, default=5)
    ap.add_argument("--M-eval", dest="M_eval", type=int, nargs="+", default=[1, 2, 3, 5, 8, 12, 16, 20])
    ap.add_argument("--eval-batches", dest="eval_batches", type=int, default=30)
    ap.add_argument("--eval-batch", dest="eval_batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--log-every", dest="log_every", type=int, default=250)
    args = ap.parse_args()
    if args.threads == 0:
        # The model is tiny (~0.3M params): small matmuls thrash with many threads.
        # Cap at 8 — past that, thread overhead dominates and steps get slower.
        args.threads = min(8, os.cpu_count() or 1)
    torch.set_num_threads(args.threads)
    import os, platform
    print(f"CLAIM 5a — GPT-2 ICL training (<= {args.steps} steps, d={args.dim}, "
          f"{args.depth}L/{args.heads}H/{args.d_model}) on CPU")
    print(f"cpu: {platform.processor()}  cpus_available={os.cpu_count()}  torch_threads={args.threads}  "
          f"time_budget={args.time_budget_sec}s", flush=True)
    t0 = time.time()
    model, losses, n_params = train(args)
    train_time = time.time() - t0
    steps_trained = losses[-1]["step"] if losses else 0
    print(f"training done in {train_time:.0f}s; steps={steps_trained}; final loss={losses[-1]['loss']:.4f}", flush=True)
    ev_t0 = time.time()
    results = evaluate(model, args)
    eval_time = time.time() - ev_t0
    cls = classify(results)
    out = {
        "claim": "Claim 5a — GPT-2-architecture ICL: non-monotone per-task error vs context length M",
        "source": {"arxiv": "2605.28705", "section": "5.1, Appendix D (model: tiny GPT-2 3L/2H/64)"},
        "config": {"steps_budget": args.steps, "steps_trained": steps_trained, "dim": args.dim,
                   "d_model": args.d_model, "depth": args.depth, "heads": args.heads, "batch": args.batch,
                   "lr": args.lr, "seed": args.seed, "params_millions": round(n_params / 1e6, 4),
                   "T_eval": args.T_eval, "M_eval": args.M_eval, "time_budget_sec": args.time_budget_sec},
        "training": {"final_loss": losses[-1]["loss"], "loss_curve": losses[::max(1, len(losses)//20)],
                     "train_seconds": round(train_time, 1), "threads": args.threads,
                     "ms_per_step": round(train_time / max(steps_trained, 1) * 1000, 1)},
        "eval_seconds": round(eval_time, 1),
        "per_task_nmse_vs_M": {str(M): results[M]["per_task_nmse"] for M in sorted(results.keys())},
        "classification": cls,
        "verdict": "VERIFIED" if (cls["task1_monotone_decreasing"] and cls["any_later_task_interior_peak"]) else "PARTIAL",
    }
    payload = json.dumps(out, indent=2, sort_keys=True)
    odir = Path(__file__).resolve().parents[1] / "outputs"; odir.mkdir(parents=True, exist_ok=True)
    (odir / "claim5a_gpt2_icl.json").write_text(payload + "\n")
    print("\nCLASSIFICATION:")
    print(f"  Task-1 monotone decreasing: {cls['task1_monotone_decreasing']}")
    for lp in cls["later_tasks"]:
        print(f"  Task {lp['task']}: peak={lp['peaked']} (argmax M={lp['argmax_M']}, "
              f"start={lp['start']:.3f} peak={lp['peak']:.3f} end={lp['end']:.3f})")
    print(f"verdict: {out['verdict']}")
    print("RESULTS_SHA256=" + hashlib.sha256(payload.encode()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
