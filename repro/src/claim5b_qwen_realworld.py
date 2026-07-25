#!/usr/bin/env python3
"""CLAIM 5b (Section 5.5, Table 2) — Qwen2.5-1.5B-Instruct real-world ICCL.

Exact claim: an instruction-tuned LLM (Qwen2.5-1.5B-Instruct), with frozen weights,
processing a concatenated two-task prompt (M SST-2 examples then M AG News examples)
exhibits (i) negative transfer on Task B at small M recovering to baseline by M=19,
and (ii) catastrophic forgetting on Task A — accuracy dropping ~46% (0.934 -> 0.472 at
M=1) and remaining far below baseline even as M grows (persistent forgetting floor).

Implementation (clean-room, CPU-only):
  - Qwen2.5-1.5B-Instruct via `transformers`, greedy generation (max_new_tokens=16),
    label parsed by keyword matching (Appendix D protocol).
  - For each M in {1,3,5,19}: baseline accuracy of A alone and B alone (M own examples);
    ICCL accuracy of Task B after M Task-A examples (negative transfer); final accuracy
    of Task A queried at the end of the A-then-B prompt (forgetting).
  - N evaluation queries per condition, batched generation.
Verdict: VERIFIED if Task-A forgetting drop is large (>=25pp at M=1) and Task-B shows
small-M degradation recovering toward baseline by M=19; else reports the table honestly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SST2_LABELS = {0: "Negative", 1: "Positive"}
AGNEWS_LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Science/Tech"}
AGNEWS_WORDS = {0: ["world", "politic"], 1: ["sport"], 2: ["business", "econom", "market"],
                3: ["science", "tech", "computer"]}


def parse_label(text, task):
    t = text.lower().strip()
    if task == "sst2":
        if any(w in t for w in ["negative", "neg"]) and not any(w in t for w in ["positive"]):
            return 0
        if any(w in t for w in ["positive", "pos"]):
            return 1
        if "negative" in t:
            return 0
        return -1
    # ag_news
    for cls, words in AGNEWS_WORDS.items():
        if any(w in t.split("\n")[0] for w in words):
            return cls
    # fallback: first-line keyword
    first = t.split("\n")[0]
    for cls, words in AGNEWS_WORDS.items():
        if any(w in first for w in words):
            return cls
    return -1


HEADER = ("You are an expert text classifier. Read the labeled examples and classify "
          "the final text. Answer with only the label word.\n\n")


def build_prompt(a_examples, b_examples, query_text, query_task_name, query_label_words):
    parts = [HEADER]
    if a_examples:
        parts.append("--- Task 1: Sentiment Analysis ---\n")
        for txt, lab in a_examples:
            parts.append(f"Text: {txt}\nLabel: {lab}\n\n")
    if b_examples:
        parts.append("--- Task 2: News Topic Classification ---\n")
        for txt, lab in b_examples:
            parts.append(f"Text: {txt}\nLabel: {lab}\n\n")
    parts.append(f"--- Final Query: {query_task_name} ---\n")
    parts.append(f"Text: {query_text}\nLabel:")
    return "".join(parts)


def make_inputs(tok, prompts, device, max_len):
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    enc = {k: v.to(device) for k, v in enc.items()}
    return enc


@torch.no_grad()
def generate_labels(model, tok, prompts, device, max_len, max_new):
    enc = make_inputs(tok, prompts, device, max_len)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=1.0,
                         pad_token_id=tok.pad_token_id)
    new = out[:, enc["input_ids"].shape[1]:]
    return [tok.decode(o, skip_special_tokens=True) for o in new]


def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def accuracy_over(model, tok, device, max_len, max_new, query_items, build_fn, batch_size):
    prompts, gold = [], []
    for q in query_items:
        prompts.append(build_fn(q)); gold.append(q["label"])
    preds = []
    for bat in chunk(prompts, batch_size):
        outs = generate_labels(model, tok, bat, device, max_len, max_new)
        preds.extend(outs)
    task = "sst2" if query_items and isinstance(query_items[0]["label"], int) and query_items[0].get("task") == "sst2" else "agnews"
    correct = 0; parsed = 0
    for p, g in zip(preds, gold):
        pl = parse_label(p, task)
        if pl == -1:
            continue
        parsed += 1
        if pl == g:
            correct += 1
    return correct / max(len(prompts), 1), parsed / len(prompts), preds[:3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=128)
    ap.add_argument("--M-list", type=int, nargs="+", default=[1, 3, 5, 19])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model", type=str, default=MODEL)
    args = ap.parse_args()

    print(f"CLAIM 5b — Qwen2.5-1.5B-Instruct ICCL on SST-2 + AG News (CPU)")
    t0 = time.time()
    device = "cpu"
    torch.set_num_threads(max(1, torch.get_num_threads()))
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"loading {args.model} (float32, CPU) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model loaded: {n_params/1e9:.2f}B params, {time.time()-t0:.0f}s")

    rng = torch.Generator().manual_seed(args.seed)
    import numpy as np
    rngnp = np.random.default_rng(args.seed)
    print("loading datasets ...", flush=True)
    sst2 = load_dataset("stanfordnlp/sst2", split="validation")
    agnews = load_dataset("fancyzhx/ag_news", split="test")
    # subsample pools
    na = len(sst2); nb = len(agnews)
    a_idx = rngnp.choice(na, size=min(2000, na), replace=False)
    b_idx = rngnp.choice(nb, size=min(2000, nb), replace=False)
    a_pool = [{"text": sst2[int(i)]["sentence"], "label": int(sst2[int(i)]["label"]),
               "word": SST2_LABELS[int(sst2[int(i)]["label"])]} for i in a_idx]
    b_pool = [{"text": agnews[int(i)]["text"], "label": int(agnews[int(i)]["label"]),
               "word": AGNEWS_LABELS[int(agnews[int(i)]["label"])]} for i in b_idx]

    table = {}
    for M in args.M_list:
        row = {"M": M}
        # fixed demonstration pools for this M
        a_ex_pool = a_pool[M * 4:]            # exclude the first M*4 used as demos
        b_ex_pool = b_pool[M * 4:]
        demos_a = [(d["text"], d["word"]) for d in a_pool[:M]]
        demos_b = [(d["text"], d["word"]) for d in b_pool[:M]]
        queries_a = [dict(task="sst2", **q) for q in a_ex_pool[:args.n_queries]]
        queries_b = [dict(task="agnews", **q) for q in b_ex_pool[:args.n_queries]]

        def baseline_a(q):
            return build_prompt([(q["text"], q["word"])] if False else demos_a, [], q["text"],
                                "Sentiment Analysis", "Positive/Negative")
        # baseline A: A demos + A query
        def ba(q):
            return build_prompt(demos_a, [], q["text"], "Sentiment Analysis", "Positive/Negative")
        # baseline B: B demos + B query
        def bb(q):
            return build_prompt([], demos_b, q["text"], "News Topic Classification",
                                "World/Sports/Business/Science")
        # ICCL Task B (negative transfer): A demos, B demos, B query
        def iccl_b(q):
            return build_prompt(demos_a, demos_b, q["text"], "News Topic Classification",
                                "World/Sports/Business/Science")
        # ICCL Task A (forgetting): A demos, B demos, A query at the END
        def iccl_a(q):
            return build_prompt(demos_a, demos_b, q["text"], "Sentiment Analysis", "Positive/Negative")

        print(f"\n--- M={M} ---", flush=True)
        acc_a_base, pa, _ = accuracy_over(model, tok, device, args.max_len, args.max_new,
                                          queries_a, ba, args.batch_size)
        acc_b_base, pb, _ = accuracy_over(model, tok, device, args.max_len, args.max_new,
                                          queries_b, bb, args.batch_size)
        acc_b_iccl, pb2, _ = accuracy_over(model, tok, device, args.max_len, args.max_new,
                                           queries_b, iccl_b, args.batch_size)
        acc_a_final, pa2, _ = accuracy_over(model, tok, device, args.max_len, args.max_new,
                                            queries_a, iccl_a, args.batch_size)
        row.update({"B_baseline": acc_b_base, "B_iccl": acc_b_iccl, "B_delta": acc_b_iccl - acc_b_base,
                    "A_baseline": acc_a_base, "A_final": acc_a_final, "A_delta": acc_a_final - acc_a_base,
                    "parsed_frac_A": pa2, "parsed_frac_B": pb2})
        table[M] = row
        print(f"  Task B: base={acc_b_base:.3f} iccl={acc_b_iccl:.3f} delta={acc_b_iccl-acc_b_base:+.3f}", flush=True)
        print(f"  Task A: base={acc_a_base:.3f} final={acc_a_final:.3f} delta={acc_a_final-acc_a_base:+.3f}", flush=True)

    # verdict
    m1 = table.get(1, table[args.M_list[0]])
    a_drop = m1["A_baseline"] - m1["A_final"]
    # negative transfer at small M, recovery by large M
    small_delta = min(table[M]["B_delta"] for M in args.M_list[:2])
    big_m = max(args.M_list)
    big_delta = table[big_m]["B_delta"]
    forgetting_persists = all(table[M]["A_delta"] < -0.2 for M in args.M_list)
    checks = {"task_A_forgetting_drop_large": bool(a_drop >= 0.25),
              "task_B_small_M_negative_transfer": bool(small_delta < 0.0),
              "task_B_recovers_toward_baseline_at_large_M": bool(big_delta > small_delta),
              "task_A_forgetting_persists": bool(forgetting_persists)}
    verdict = "VERIFIED" if all(checks.values()) else "PARTIAL"

    out = {"claim": "Claim 5b — Qwen2.5-1.5B-Instruct ICCL: ~46% Task-A forgetting + Task-B negative transfer",
           "source": {"arxiv": "2605.28705", "section": "5.5, Table 2"},
           "config": {"model": args.model, "n_queries": args.n_queries, "M_list": args.M_list,
                      "batch_size": args.batch_size, "max_new_tokens": args.max_new, "seed": args.seed,
                      "params_billions": round(n_params / 1e9, 3)},
           "paper_reference": {"M1": {"A_base": 0.934, "A_final": 0.472, "A_delta": -0.462,
                                      "B_base": 0.736, "B_iccl": 0.580, "B_delta": -0.156}},
           "results_table": {str(M): table[M] for M in args.M_list},
           "checks": checks,
           "wall_seconds": round(time.time() - t0, 1),
           "verdict": verdict}
    payload = json.dumps(out, indent=2, sort_keys=True)
    odir = Path(__file__).resolve().parents[1] / "outputs"; odir.mkdir(parents=True, exist_ok=True)
    (odir / "claim5b_qwen_realworld.json").write_text(payload + "\n")
    print("\nCHECKS:")
    for k, v in checks.items():
        print(f"  {k}: {v}")
    print(f"verdict: {verdict}")
    print("RESULTS_SHA256=" + hashlib.sha256(payload.encode()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
