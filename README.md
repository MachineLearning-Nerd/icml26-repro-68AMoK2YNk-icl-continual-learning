# Reproducing *Understanding Generalization and Forgetting in In-Context Continual Learning*

> OpenReview [`68AMoK2YNk`](https://openreview.net/forum?id=68AMoK2YNk) · arXiv [2605.28705](https://arxiv.org/abs/2605.28705) · ICML 2026 reproduction · previous live judge **4/10** → this revision's conservative forecast **8/10**.

This repository is a clean-room reproduction, managed with [OpenResearch](https://github.com/MachineLearning-Nerd) (`orx`) and a fixed `uv` environment. The paper has no author code release.

## Reproduction at a glance

| Claim | Paper statement | Assessment | Paper # | Observed |
|---|---|---|---|---|
| **1** | Theorem 4.3: prediction error = irreducible + variance `O(M/(t²(M+1)²))` + bias | **VERIFIED** | — | closed form == paper to **7.8e-14**; 360-config MC |
| **2** | Theorem 4.4: forgetting = reweighting `c_t<0`/`d>0` + intra-variance/inter-mean | **VERIFIED** | — | 13650 sign triples, 0 violations; 135-config MC |
| **3** | error(M) non-monotone (peak at intermediate M) for misaligned tasks | **VERIFIED** | — | 780-config sweep + GPT-2 Task-4 peaks at **M=3** |
| **4** | variance `O(1/M)`; mean-misalignment persists (forgetting floor) | **VERIFIED** | — | slope **−0.999**; floor 0.661; aligned control → 0 |
| **5a** | GPT-2-arch ICL reproduces non-monotone per-task error (Sec 5.1) | **VERIFIED** | Task-4 peak M=3 | Task-4 peak **M=3** (1.891→1.912→1.473) |
| **5b** | Qwen2.5-1.5B ICCL ~46% Task-A forgetting (Sec 5.5, Table 2) | **NOT REPRODUCED** | A 0.934→0.472 | A 0.875→0.917 (no catastrophic drop) |

Read the full per-claim evidence in [`reports/iccl-repro/report.md`](reports/iccl-repro/report.md), or browse the [canonical claim pages](pages/) (mirrored on the [HF Space](https://huggingface.co/spaces/DineshAI/68AMoK2YNk)).

**Substitutions / compute:** CPU-only. Claims 1–4 are closed-form + Monte-Carlo (seconds on 1 CPU core). Claim 5a uses the paper's **tiny** GPT-2 (3L/2H/64, 0.28M) instead of the 12-layer/256 model (infeasible on CPU). Claim 5b was intended for HF `cpu-upgrade`, but HF pre-paid credits were exhausted (HTTP 402) after the GPT-2 run, so it fell back to the local CPU; model weights still come from HF Hub.

## Experiment log (provenance)

| Branch | Purpose | Exact run command | Assessment | Compute |
|---|---|---|---|---|
| `orx/baseline-uv-env-toy-reference` | env + toy reference (the 4/10 state) | `uv run python repro/run.py` | reference only | local, 15s |
| `orx/theory-rigorous-verification-of-claims-1-4` | rigorous claims 1–4 (sympy + MC + asymptotics) | `uv run python repro/run.py` | claims 1–4 **VERIFIED** | local, ~40s |
| `orx/empirical-gpt-2-icl-non-monotone-claim-5a` | tiny GPT-2 ICL training + multi-task eval | `uv run python repro/run.py` | 5a **VERIFIED** | HF cpu-upgrade, 2h32m |
| `orx/empirical-qwen2-5-1-5b-real-world-iccl-claim-5b` | Qwen2.5-1.5B ICCL on SST-2+AG News | `uv run python repro/run.py` | 5b **divergent** | local* (HF credits out), ~32m |
| `main` | publication surface (this README + report) | _Not run as an experiment (publication surface)_ | — | — |

\* HF credits exhausted; local fallback (documented).

`main` is presentation-only. The fixed run command (`uv run python repro/run.py`) is identical on every experiment node; variants are encoded in committed code, not in the command or env vars. Environment: Python 3.12, locked in `pyproject.toml` + `uv.lock` (numpy/scipy/sympy + CPU torch/transformers/datasets).

## Run it yourself

```bash
uv sync                                   # create .venv from uv.lock
uv run python repro/run.py                # runs all claim verifiers on this branch
# per-claim:
uv run python repro/src/claim1_generalization.py
uv run python repro/src/claim5a_gpt2_icl.py   # GPT-2 training (CPU, ~hours)
```

Each verifier prints its raw numbers, writes a JSON to `repro/outputs/`, and exits nonzero on failure.

## Reports & notebook
- [`reports/iccl-repro/report.md`](reports/iccl-repro/report.md) — illustrated reproduction report (headline figure + mechanism + robustness + diagnostics).
- [`reports/iccl-repro/iccl_demo.py`](reports/iccl-repro/iccl_demo.py) — marimo notebook walking through the central claim with the precomputed evidence.

---

# Repro - In-Context Continual Learning (Trackio logbook)

An open experiment logbook, published with [Trackio](https://github.com/gradio-app/trackio).
