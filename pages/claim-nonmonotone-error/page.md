# Non-monotonic in-context error (Theorem 4.3)

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_ic_i", "created_at": "2026-07-22T03:00:00+00:00", "title": "Thm 4.3: error is non-monotonic in context length under misalignment"}
-->
### Claim — VERIFIED by exact evaluation of the closed-form error

Theorem 4.3 gives the test error as `irreducible + variance(M) + bias(M)`. When task means **misalign**, adding in-context examples can *increase* error below a threshold; for **aligned** tasks error decreases monotonically.

---
<!-- trackio-cell
{"type": "code", "id": "cell_ic_r", "created_at": "2026-07-22T03:00:00+00:00", "title": "Executed reproduction", "command": ["python", "repro/src/verify_incontext.py"], "exit_code": 0, "duration_s": 1.0}
-->
````bash
$ python repro/src/verify_incontext.py
````

````output
claim: InContext_ContinualLearning_Thm4.3_nonmonotone
Exact evaluation of Theorem 4.3 test-error formula; current task t=2, d=2, Lambda=I, N=100.

ALIGNED (theta=0):   err(M=0)=1.0, err(M=1)=0.5004, err(M=40)=0.0252 -> monotone decreasing: True
MISALIGNED (theta=150): err(M=0)=1.0, err(M=1)=1.1864 (peak at M=1, err=1.1864), err(M=40)=0.9556
   -> non-monotonic (context HURTS below threshold then HELPS): True; threshold M=1
verdict: supports
````

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_ic_c", "created_at": "2026-07-22T03:00:00+00:00", "title": "Interpretation"}
-->
**VERIFIED.** Evaluating the exact Theorem 4.3 error formula (current task t=2, d=2, Λ=I, N=100): for **aligned** tasks (θ=0) the error decreases monotonically with context length M (1.00 → 0.50 → 0.025). For **misaligned** tasks (θ=150°) the error is **non-monotonic** — it rises from 1.00 (M=0) to a peak of 1.19 at M=1, then decreases to 0.96 — exactly the paper's finding that *below a threshold, additional in-context examples mislead the model and degrade generalization*, with the benefit of longer context emerging only beyond that threshold.