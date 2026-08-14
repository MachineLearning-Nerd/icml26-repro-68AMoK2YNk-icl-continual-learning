# Repro — Understanding Generalization and Forgetting in In-Context Continual Learning

OpenReview `68AMoK2YNk` · arXiv [2605.28705](https://arxiv.org/abs/2605.28705) · previous live judge score **4/10**.

**Current verification** — start here:

| Page | Verdict |
| --- | --- |
| [Score & visibility summary](#/00-iccl-score-summary) | forecast 8/10 (conservative) |
| [Claim 1 — Theorem 4.3 generalization](#/claim-1-generalization) | **VERIFIED_SCOPED** |
| [Claim 2 — Theorem 4.4 forgetting](#/claim-2-forgetting) | **VERIFIED_SCOPED** |
| [Claim 3 — non-monotone error vs M](#/claim-3-nonmonotone) | **VERIFIED_SCOPED** |
| [Claim 4 — O(1/M) variance + persistent floor](#/claim-4-asymptotic) | **VERIFIED_SCOPED** |
| [Claim 5 — GPT-2 + Qwen experiments](#/claim-5-experiments) | 5a **RETAINED REPORT ONLY** · 5b divergent retained report |
| [Methods, env, commands](#/methods-rigorous) | fixed `uv run python repro/run.py` |

Historical rejected baseline (the prior 4/10 toy-scale pages, kept unchanged):
[Claim 1 — Interference (toy)](#/claim-1-interference) · [Claim 2 — Transfer (toy)](#/claim-2-transfer) · [Claim 3 — Order (toy)](#/claim-3-order-and-prompts) · [Non-monotone (toy)](#/claim-nonmonotone-error) · [Claim 4 asymptotics (toy)](#/claim-4-interference-asymptotics) · [Methods (old)](#/methods) · [Negative controls (old)](#/negative-controls) · [Conclusion (old)](#/conclusion)

The canonical release gate verifies Claims 1–4 locally. The model-scale results
remain separate because the later external run logs and checkpoints are not in
this GitHub clone; see `docs/CLAIM_AUDIT.md` in the repository.
