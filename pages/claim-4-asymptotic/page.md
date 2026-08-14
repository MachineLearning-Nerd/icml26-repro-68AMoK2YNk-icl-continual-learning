# Claim 4 — O(1/M) variance decay + persistent mean-interference floor

## Exact claim tested
Forgetting interference splits into (i) a **variance** part that decays as `O(1/M)` with context length and (ii) a **mean-misalignment** part that **persists regardless of M**, leaving a nonzero asymptotic forgetting floor for misaligned tasks (and vanishing for aligned tasks). Paper: Sec 4 (forgetting, Context Length), Remark.

**Verdict: VERIFIED_SCOPED** (HIGH confidence within the finite audit scope).

## Decomposition (from the Claim-2-verified interference formula)
```
variance(M) = M · Σ_s α_s(M)² · tr(Σ_s Γ⁻²Λ)        →  O(1/M)   since α_s = O(1/M)
mean(M)     = M² · ‖Σ_s α_s(M) μ_s‖²_{Γ⁻²Λ}         →  const      since M·α_s → limit
limits:   M·α_s → −(T−t)/(tT)  (s≤t),   1/T  (s>t)
```

## Fixed-config sweep over six orders of magnitude in M
Config: `T=4, t=2, d=3, Λ=diag(0.8,1.3,2.1)`, misaligned means. `M ∈ {1,2,4,…,1048576}`:
- **variance tail log-log slope = −0.9992** (target −1).
- **M·variance → analytic limit**, relative error **2.1e-6** at M=2²⁰.
- **mean interference → persistent floor 0.6613** (positive; relative error 1.4e-6).
- **Aligned negative control** (all means equal): floor → **0** (ratio 8e-14 vs misaligned).

![asymptotic](https://raw.githubusercontent.com/MachineLearning-Nerd/icml26-in-context-continual-learning/main/reports/iccl-repro/images/fig_theory_asymptotic.png)

## Broad random sweep (generality)
12 random configs (`d∈{2..6}`, `T∈{2..5}`, random PSD Λ, random w): **11/12** have variance slope within 0.01 of −1; **11/12** have a positive persistent floor reached to <1e-4.

## Commands, env, provenance
- Command: `uv run python repro/src/claim4_asymptotic.py`. Env: Python 3.12, numpy 2.5.1.
- Run: `a294fef0` (local). Seed `20245`. Wall <1s.
- Code: [`claim4_asymptotic.py`](https://github.com/MachineLearning-Nerd/icml26-in-context-continual-learning/blob/audit/theory-claims-1-4/repro/src/claim4_asymptotic.py). Raw JSON: `repro/outputs/claim4_asymptotic.json`. Verifier exits nonzero on failure.

## Limitations
The analytic limit uses `Γ commuting with Λ` (true: `Γ = aΛ + bI`); the floor value depends on the task-mean alignment and ordering, exactly as the paper states.
