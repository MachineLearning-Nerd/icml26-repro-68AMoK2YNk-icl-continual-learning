# Claim audit

This ledger distinguishes executable local evidence from retained external
reports. A `VERIFIED_SCOPED` result means the named producer and finite scope
pass locally; it does not mean that every theorem quantifier or every paper
experiment has been reproduced.

## Claim-to-evidence map

| Claim | Paper anchor | Producer | Evidence | Result |
| --- | --- | --- | --- | --- |
| C1 — generalization | Theorem 4.3 | `repro/src/claim1_generalization.py` | SymPy coordinate identity; 71 moment cells at 40,000 trials; 360 closed-form/Monte-Carlo configurations at 6,000 trials | `VERIFIED_SCOPED` |
| C2 — forgetting/interference | Theorem 4.4 | `repro/src/claim2_forgetting.py` | Exact `c_t`/`d` identity; 13,650 `(T,t,M)` sign cells; 135 closed-form/Monte-Carlo configurations at 20,000 trials | `VERIFIED_SCOPED` |
| C3 — non-monotone context effect | Sec. 4, context-length/task-similarity analysis | `repro/src/claim3_nonmonotone.py` | 780 configurations; 47/252 misaligned multi-task interior peaks; aligned and single-task controls monotone | `VERIFIED_SCOPED` |
| C4 — asymptotic interference | Sec. 4 forgetting/context-length analysis | `repro/src/claim4_asymptotic.py` | Tail slope `-0.999219`; `M`-scaled variance limit relative error `2.08e-6`; floor `0.661267`; aligned zero-floor control; 11/12 broad controls | `VERIFIED_SCOPED` |
| C5a — tiny GPT-2 experiment | Sec. 5.1 / Appendix D | `repro/src/claim5a_gpt2_icl.py` on `experiment/gpt2-claim-5a` | Later HF run is described by report/figures (`dee18666`, 34,000 steps); committed JSON is a separate 50-step `PARTIAL` run | `RETAINED_REPORT_ONLY` |
| C5b — Qwen real-world ICCL | Sec. 5.5 / Table 2 | `repro/src/claim5b_qwen_realworld.py` on `experiment/qwen-claim-5b` | Report describes a divergent Qwen2.5-1.5B run; committed JSON is an earlier Qwen2.5-0.5B/32-query `PARTIAL` snapshot | `DIVERGENT_RETAINED_REPORT` |

## C1 — Theorem 4.3

The producer derives the paper's irreducible, variance, and task-misalignment
bias terms. It independently checks the per-eigen-coordinate algebra with
SymPy, checks the `E[S_iS_j^T]` moment lemma using raw Gaussian draws, and
compares the closed form with direct Monte Carlo over 360 seeded configurations.
The generated report is `repro/outputs/claim1_generalization.json`.

This is a finite formula and moment audit. It is not a formal proof of the
paper's universal theorem statement.

## C2 — Theorem 4.4

The producer derives the difference between the task-prefix and final-query
weights. It proves the past coefficient is negative and the future coefficient
positive, exhausts `T <= 20`, `M <= 64`, and every valid task position, then
compares the paper-expanded interference expression with an independent
calculation and Monte Carlo. The generated report is
`repro/outputs/claim2_forgetting.json`.

The measured forgetting quantity is prediction drift between two queries, not a
target-risk increase. That distinction is part of the paper's definition and is
kept in the gate limitations.

## C3 — context length and task similarity

The producer sweeps dimension, task count, target position, training prompt
length, and task angle. It requires monotone decrease for aligned/single-task
controls and records interior peaks only in the declared misaligned multi-task
regime. The report is `repro/outputs/claim3_nonmonotone.json`.

The later GPT-2 curve shown in the report is not used as a locally regenerated
input to this claim gate. It is separately classified as C5a.

## C4 — asymptotic behavior

The producer sweeps context length through six orders of magnitude for a fixed
misaligned configuration, compares the scaled variance to its analytic limit,
checks the persistent mean-interference floor, and runs a broad control sweep.
The report is `repro/outputs/claim4_asymptotic.json`.

## C5a — GPT-2 artifact boundary

The committed branch code implements the tiny GPT-2 configuration, and the
publication report records a later HF run with Task-4 peaking at `M=3`. However,
the committed `repro/outputs/claim5a_gpt2_icl.json` says `PARTIAL`, with only 50
training steps and no 34k-step checkpoint. The figures are derived from the
reported later run, but its raw log/checkpoint is not in this repository.

Therefore this repository preserves the result as a retained external report,
not as a fresh-clone verified claim.

## C5b — Qwen artifact boundary and divergence

The source paper's target is Qwen2.5-1.5B-Instruct and reports a Task-A accuracy
change of `0.934 -> 0.472` at `M=1`. The report describes a later 1.5B run that
observed approximately `0.875 -> 0.917` instead, so it does not reproduce the
large drop and is not called a falsification.

The committed `repro/outputs/claim5b_qwen_realworld.json` is an earlier
`Qwen2.5-0.5B-Instruct`, 32-query, `PARTIAL` snapshot. It is retained to make
the branch history inspectable, but it cannot serve as the raw artifact for the
later 1.5B report. The exact 1.5B run remains pending raw logs, dataset sample
IDs, decoding details, and model-weight provenance.

## Legacy baseline

`outputs/full_evidence.json` and `outputs/PUBLICATION_GATE_PASSED.json` are the
earlier three-claim toy-scale publication marker. They remain available for
history, but the canonical gate for this repository is the combined root
`publication_gate.json` generated from C1–C4 and the C5 boundary checks.
