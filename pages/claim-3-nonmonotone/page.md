# Claim 3 — non-monotonic error vs context length M (theory + GPT-2)

## Exact claim tested
The task-`t` generalization error as a function of in-context length `M` is **non-monotonic** when historical tasks are misaligned with the target (rises to a peak at an intermediate `M` — bias builds while variance is still high — then recovers), and **monotone-decreasing** for aligned tasks. Paper: Sec 4 (Context Length), Sec 5.1.

**Verdict: VERIFIED_SCOPED** (HIGH confidence) for the theory sweep. The GPT-2
curve is retained report evidence for C5a, not an input to the local theorem
gate.

## Theory verification (Theorem 4.3 closed form, Claim-1-verified)
Broad sweep — `d∈{2,4}`, `T∈{2,3}`, `t∈[1,T]`, `N∈{4,6,8,10,50,300}`, `θ∈[0,180]°`: **780 configs**, error(M) over `M∈[1,30]`.
- **Negative control 1 — aligned tasks (θ≤15°):** monotone-decreasing fraction **1.00**.
- **Negative control 2 — single task (t=1, no inter-task bias):** monotone-decreasing fraction **1.00**.
- **Claim — misaligned multi-task (θ≥90°, t≥2):** **47 / 252** configs show a strict interior peak. Example: `d=2, T=3, t=3, θ=120°, N=4` → error `1.109 → 1.122 (peak at M=3) → 1.112`.

Honest scope: at large `N` the Theorem-4.3 error is monotone for all alignments (the bias is then nearly M-independent); interior peaks emerge in the **multi-task, misaligned, modest-N** regime where the bias term is strongly M-sensitive. Pronounced peaks are reproduced empirically below.

## Retained empirical report (GPT-2, see Claim 5a)
A later external report says that a tiny GPT-2 (3L/2H/64) trained on
linear-regression ICL shows this pattern:
- **Task 1**: monotone `0.739 → 0.043`.
- **Task 4**: peak at **M=3** (`1.891 → 1.912 → 1.473`) — matches the paper's "peak at M=3".
- **Task 5**: peak at M=16.

![GPT-2 per-task NMSE vs M](https://raw.githubusercontent.com/MachineLearning-Nerd/icml26-in-context-continual-learning/main/reports/iccl-repro/images/fig_gpt2_nonmonotone.png)

![Theory non-monotone curves](https://raw.githubusercontent.com/MachineLearning-Nerd/icml26-in-context-continual-learning/main/reports/iccl-repro/images/fig_theory_nonmonotone.png)

## Commands, env, provenance
- Theory: `uv run python repro/src/claim3_nonmonotone.py`, run `a294fef0` (local), seed `20244`, ~2s.
- Empirical report: run `dee18666` (HF cpu-upgrade), seed `12345`, 2h32m. The
  committed branch JSON is a separate 50-step `PARTIAL` snapshot; the later
  checkpoint and raw metric table are not in this repository.
- Code: [`claim3_nonmonotone.py`](https://github.com/MachineLearning-Nerd/icml26-in-context-continual-learning/blob/audit/theory-claims-1-4/repro/src/claim3_nonmonotone.py), [`claim5a_gpt2_icl.py`](https://github.com/MachineLearning-Nerd/icml26-in-context-continual-learning/blob/experiment/gpt2-claim-5a/repro/src/claim5a_gpt2_icl.py). Raw JSON: `repro/outputs/claim3_nonmonotone.json`.

## Note on the prior toy evidence
The earlier toy page reported "θ=150°: 1.0→1.19→0.96" using a *simplified diagonal* formula. The **correct** Theorem-4.3 closed form (verified in Claim 1 to 7.8e-14) is monotone at large N for that single-config; the genuine non-monotonicity requires the multi-task modest-N regime characterized above. The prior toy over-stated the effect; this page corrects it with full-scale evidence.
