# Claim 1 — Theorem 4.3 (generalization error decomposition)

## Exact claim tested
For task `t`, the prediction error of the converged masked linear self-attention model decomposes as
```
E[(ŷ_{t,q} − y_{t,q})²]  =  irreducible
   +  M/(t²(M+1)²) · Σ_{s≤t} tr(Σ_s Γ⁻²Λ)                      (variance; ↓ with M)
   +  Σ_i (1/λ_i)·[ (λ_i(α s_i − m_i) − m_i(λ_i+trΛ)/N)
                    / (λ_i + (λ_i+trΛ)/N) ]²                     (bias; task dissimilarity)
```
with `α = M/(t(M+1))`, `s_i = v_iᵀ Σ_{s≤t} μ_s`, `m_i = v_iᵀ μ_t`, `Γ = (1+1/N)Λ + (trΛ/N)I`.
Paper: Theorem `thm-generalization` (Sec 4.1), `source/main.tex` lines 315–332.

**Verdict: VERIFIED** (HIGH confidence).

## Source audit
- arXiv 2605.28705v1; `source/main.tex` SHA-256 `026aa3bdb1f7903172b9935e6181123260ff6e4dedc3a24965bfe1ce6e5f97f9`.
- Assumptions: `x_{s,i}∼N(0,Λ)`, `y=w_sᵀx`, masked linear self-attention at the gradient-flow limit (Theorem `limit`). Universally quantified over `T,M,t,Λ,{w_s}`.

## Three independent verification routes (non-circular)

**(A) Symbolic identity (sympy)** — derive `E[(ŷ−y)²]` from the prediction rule `ŷ = x_qᵀ Γ⁻¹ β_t Σ_{s≤t} S_s` and the Gaussian moment lemma, then show it equals the paper's 3-term form **per eigen-coordinate** for symbolic `λ, N, α, s_i, m_i`:
- bias-coordinate identity `mine − paper = 0` (sympy `simplify`).
- variance-coefficient identity `β_t² M − M/(t²(M+1)²) = 0`.
This is the machine-checkable reconstructed derivation.

**(B) Moment-lemma Monte-Carlo** — independently confirm `E[S_i S_jᵀ]` from raw Gaussian draws: 71 cells × 40000 trials, **max relative error 1.59%**.

**(C) Closed-form vs raw Monte-Carlo** — over a broad sweep (`d∈{2,4,6,10}`, `T∈{2,4,6}`, `M∈{1,2,4,8,16}`, random PSD Λ, random w, 2 reps): **360 configs × 6000 trials**.
- closed-form (route A) **==** paper's formula: max error **7.8e-14** (exact algebra).
- closed-form vs raw MC: max rel err 4.3% (MC is corroboration; the symbolic identity is the rigorous result).

## Negative control
The literal appendix denominator `T+M+1` (a documented typo) is rejected: it fails 9600/9600 checks vs the independent calculation; the theorem-consistent `TM+T+1` passes.

## Inline raw numbers (sample)
| d | T | M | t | closed | paper | MC(6000) |
|--|--|--|--|--|--|--|
| 4 | 4 | 8 | 2 | 0.713… | 0.713… | 0.718… |
| 6 | 6 | 16 | 6 | 1.492… | 1.492… | 1.521… |

## Commands, env, provenance
- Command: `uv run python repro/src/claim1_generalization.py` (or via harness `uv run python repro/run.py`).
- Env: Python 3.12, numpy 2.5.1, sympy 1.14.0 (uv lock).
- Run: `a294fef0` (local). Seeds `20241, 20242, 9000+`. Wall ~5s.
- Code: [`theory_core.py`](https://github.com/MachineLearning-Nerd/icml26-repro-68AMoK2YNk-icl-continual-learning/blob/orx/theory-rigorous-verification-of-claims-1-4/repro/src/theory_core.py), [`claim1_generalization.py`](https://github.com/MachineLearning-Nerd/icml26-repro-68AMoK2YNk-icl-continual-learning/blob/orx/theory-rigorous-verification-of-claims-1-4/repro/src/claim1_generalization.py) · Raw JSON: `repro/outputs/claim1_generalization.json`.
- Verifier exits **nonzero** on failure (assertions).

## Limitations / deviations
Irreducible term is exactly 0 here (y = w_tᵀx is a deterministic linear map); this matches the paper's single-task reduction. The symbolic identity is proved in Λ's eigenbasis (Γ commutes with Λ, so this is without loss of generality).
