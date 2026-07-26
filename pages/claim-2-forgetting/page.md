# Claim 2 — Theorem 4.4 (forgetting = attention reweighting, not information loss)

## Exact claim tested
After all T tasks,
```
Ŷ_{t,q} − ŷ_{t,q} = x_qᵀ Γ⁻¹ ( Σ_{i≤t} c_t S_i  +  Σ_{i>t} d S_i )
c_t = −((T−t)M + (T+1−t)) / ( t(M+1)(TM+T+1) )   < 0   (PAST tasks, NEGATIVE coeff)
d   =  1 / (TM+T+1)                                > 0   (FUTURE tasks, POSITIVE coeff)
E[(Ŷ−ŷ)²] = Σ_i α_i² tr[(MΣ_i + M²μ_iμ_iᵀ)Γ⁻²Λ]  +  M² Σ_{i≠j} α_iα_j tr(μ_iμ_jᵀΓ⁻²Λ)
```
i.e. forgetting decomposes into **intra-task variance** and **inter-task mean-interaction** terms; it arises from reweighting, not information loss. Paper: Theorem `forgetting` (Sec 4.2), lines 383–427.

**Verdict: VERIFIED** (HIGH confidence).

## Verification routes

**(A) Symbolic coefficient derivation (sympy)** — the coefficient of `S_i` (`i≤t`) in `(Ŷ−ŷ)` is `1/(T(M+1)+1) − 1/(t(M+1))`. Simplified:
`= −((T−t)M + (T+1−t)) / (t(M+1)(TM+T+1)) = c_t` exactly (sympy `simplify` → 0). So past tasks carry the negative coefficient; future tasks carry `d`. This proves the reweighting structure.

**(B) Exhaustive sign analysis + analytic argument** — over the full grid `T∈[1,20]`, `M∈[0,64]`, all valid `t`: **13650 (T,t,M) triples, 0 violations** of `c_t<0` and `d>0`. Analytic: `c_t`'s numerator `(T−t)M + (T+1−t) ≥ (T+1−t) ≥ 1 > 0` in the domain.

**(C) Closed-form vs raw Monte-Carlo** — `d∈{2,4,6}`, `T∈{2,4,6}`, `M∈{1,2,4,8,16}`, 3 task positions, random Λ/w: **135 configs × 20000 trials**.
- closed-form == paper's expanded form: max error **2.8e-14**.
- closed-form vs MC: max rel err 3.6%.

## Inline raw numbers (sample)
| d | T | M | t | closed | paper | MC(20000) |
|--|--|--|--|--|--|--|
| 4 | 4 | 8 | 2 | 0.094… | 0.094… | 0.096… |
| 6 | 6 | 16 | 3 | 0.318… | 0.318… | 0.312… |

## Commands, env, provenance
- Command: `uv run python repro/src/claim2_forgetting.py`. Env: Python 3.12, sympy 1.14.0.
- Run: `a294fef0` (local). Seeds `20243, 7000+`. Wall ~11s.
- Code: [`claim2_forgetting.py`](https://github.com/MachineLearning-Nerd/icml26-repro-68AMoK2YNk-icl-continual-learning/blob/orx/theory-rigorous-verification-of-claims-1-4/repro/src/claim2_forgetting.py). Raw JSON: `repro/outputs/claim2_forgetting.json`. Verifier exits nonzero on failure.

## Limitations
The interference quantity is a *prediction-drift* (Δ between task-t and final predictions), exactly the paper's definition. The intra-/inter-task decomposition follows by expanding `E[UUᵀ]` via the moment lemma (verified in Claim 1, route B).
