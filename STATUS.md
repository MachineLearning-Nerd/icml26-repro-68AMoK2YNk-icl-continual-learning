# Repository status

- Intended repository: `MachineLearning-Nerd/icml26-in-context-continual-learning`
- Original repository: `MachineLearning-Nerd/icml26-repro-68AMoK2YNk-icl-continual-learning`
- Paper: *Understanding Generalization and Forgetting in In-Context Continual Learning*
- OpenReview: `68AMoK2YNk`
- arXiv: `2605.28705`
- Authors: Guangyu Li, Meng Ding, and Lijie Hu
- Maintainer: `MachineLearning-Nerd`
- State: `verified_scoped_with_unreproduced_experiments`
- Local theorem gate: `VERIFIED_SCOPED`, claims 1–4
- Model-scale evidence: claim 5a retained-report-only; claim 5b divergent retained report
- Last recorded external verdict: 4/10; the 8/10 value is only a conservative forecast
- Strict paper-level gate: `NOT_READY`

## Verified locally

- Claim 1 passes the symbolic generalization identity, 71 moment cells, and 360
  closed-form/Monte-Carlo configurations.
- Claim 2 passes the coefficient identity, 13,650 sign/domain cells, and 135
  closed-form/Monte-Carlo configurations.
- Claim 3 passes 780 theory configurations with aligned and single-task negative
  controls.
- Claim 4 passes the `O(1/M)` slope, asymptotic-limit, persistent-floor, and
  aligned-floor controls.
- The focused test suite passes four tests.

## Evidence boundary

The later GPT-2 and Qwen results are described in the report and figures, but
their raw external run artifacts are not present in this GitHub history. The
committed C5 JSON files are partial branch snapshots: the GPT-2 file records a
50-step run, and the Qwen file records Qwen2.5-0.5B with 32 queries. They do not
prove the later 34k-step tiny-GPT-2 or Qwen2.5-1.5B report numbers.

The source paper includes no author implementation. The independent code and
its finite scope are documented in [`docs/CLAIM_AUDIT.md`](docs/CLAIM_AUDIT.md).
