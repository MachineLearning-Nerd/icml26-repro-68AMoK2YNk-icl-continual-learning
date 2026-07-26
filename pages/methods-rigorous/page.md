# Methods, environment, commands, provenance

## Fixed run command (identical on every node)
```
uv run python repro/run.py
```
The harness `repro/run.py` discovers the claim verifiers committed on the branch, runs them in order, prints an evaluator-visible summary, and writes `outputs/run_summary.json`. Each verifier asserts its claim and exits nonzero on failure.

## Pinned environment (uv-managed, one repo-level `.venv`)
Python 3.12 · numpy 2.5.1 · scipy 1.18.0 · sympy 1.14.0 · matplotlib 3.11.1 · pandas 3.0.5 · **torch 2.13.0 (CPU)** · transformers 5.14.1 · datasets 5.0.0 · huggingface_hub · accelerate · tqdm. Locked in `pyproject.toml` + `uv.lock`.

## Experiment tree (OpenResearch `orx`)
| Branch | Purpose | Run | Backend | Wall |
|---|---|---|---|---|
| `orx/baseline-uv-env-toy-reference` (`2fa4eca`) | env + toy reference | 711f869d | local | 15s |
| `orx/theory-rigorous-verification-of-claims-1-4` (`d21038e`) | claims 1–4 rigorous | a294fef0 | local | ~40s |
| `orx/empirical-gpt-2-icl-non-monotone-claim-5a` (`a335e8c`) | GPT-2 ICL | dee18666 | HF cpu-upgrade | 2h32m |
| `orx/empirical-qwen2-5-1-5b-real-world-iccl-claim-5b` (`d17e68b`) | Qwen ICCL | 562de57a | local* | ~32m |
| `orx/release-publication-hf-space-github-report` | this artifact | — | — | — |

\* HF pre-paid credits exhausted (HTTP 402) after the GPT-2 run; claim 5b used the orx local backend (model weights from HF Hub). Documented deviation.

## Compute policy (per task)
- Claims 1–4 (closed-form + MC): local CPU, single-node, <1 min total. Estimates: ≤1 core, <5 min → local.
- Claim 5a (GPT-2 training): >1 core, >5 min → HF cpu-upgrade (`--image ghcr.io/astral-sh/uv:python3.12-bookworm`).
- Claim 5b (Qwen 1.5B inference): intended HF cpu-upgrade; credits exhausted → local fallback.

## Deterministic seeds
claim1 `{20241,20242,9000+}` · claim2 `{20243,7000+}` · claim3 `{20244}` · claim4 `{20245}` · claim5a `{12345}` · claim5b `{7}`.

## Source provenance
arXiv 2605.28705v1 tar SHA-256 `c28cca01207449b7dea91544161b5f8c764ef1001f729efd17073ba7b4101210`; `source/main.tex` SHA-256 `026aa3bdb1f7903172b9935e6181123260ff6e4dedc3a24965bfe1ce6e5f97f9`. No author code was released; the implementation is explicitly clean-room.
