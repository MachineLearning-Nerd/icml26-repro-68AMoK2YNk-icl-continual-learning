# Understanding Generalization and Forgetting in In-Context Continual Learning

Source-pinned clean-room audit for [*Understanding Generalization and Forgetting in
In-Context Continual Learning*](https://arxiv.org/abs/2605.28705) (arXiv
2605.28705, OpenReview `68AMoK2YNk`). The intended final collection entry is
[`MachineLearning-Nerd/icml26-in-context-continual-learning`](https://github.com/MachineLearning-Nerd/icml26-in-context-continual-learning).

The paper studies what happens when a frozen Transformer processes several
regression or classification tasks in one prompt. Its masked linear-attention
analysis separates variance, task-mean bias, and inter-task interference. This
repository contains an independent finite audit of the four theorem-level
claims, the retained experiment code and reports, and explicit boundaries for
the two model-scale experiments.

## Release status

| Surface | Result | Meaning |
| --- | --- | --- |
| Claims 1–4 | `VERIFIED_SCOPED` | Theorem identities, finite sign/domain checks, Monte-Carlo checks, non-monotonicity sweep, and asymptotic controls pass locally |
| Claim 5a — tiny GPT-2 | `RETAINED_REPORT_ONLY` | The report records a later 34k-step HF run, but the committed branch artifact is only a 50-step `PARTIAL` snapshot; no checkpoint or raw 34k-step table is in this repository |
| Claim 5b — Qwen real-world ICCL | `DIVERGENT_RETAINED_REPORT` | The report records a Qwen2.5-1.5B run that does not reproduce the paper's 46% drop; the committed JSON is an earlier 0.5B/32-query `PARTIAL` snapshot, so the report result is not fresh-clone reproducible |
| Last recorded external verdict | 4/10 | The later 8/10 value is a conservative forecast, not an official score |
| Evidence-release gate | `PASSED` | The finite evidence package and its limitations are hash-addressed |
| Strict paper-level gate | `NOT_READY` | Full-scale model runs and raw artifacts are not reproducible from this GitHub clone |

The canonical machine-readable decision is [`publication_gate.json`](publication_gate.json).
The older [`outputs/PUBLICATION_GATE_PASSED.json`](outputs/PUBLICATION_GATE_PASSED.json)
is retained as a historical three-claim baseline marker; it is not the combined
release decision.

## What the paper does

The theoretical setting uses task-specific linear predictors and a sequence of
in-context examples. A masked linear-attention model aggregates the current and
historical task statistics without updating its weights. The paper asks when
additional context improves generalization, when it causes positive or negative
transfer, and why earlier task predictions can drift after later tasks are
appended.

The source paper's main results are Theorem 4.3 (generalization error), Theorem
4.4 (interference/forgetting), the task-similarity and context-length analyses,
the `O(1/M)` variance decay with a persistent mean-misalignment floor, and
experiments on a GPT-2 architecture and Qwen2.5-1.5B-Instruct.

## Claim-to-evidence ledger

| Claim | Paper result | Producer | Local evidence | Status |
| --- | --- | --- | --- | --- |
| C1 | Theorem 4.3: irreducible + finite-sample variance + task-misalignment bias | `repro/src/claim1_generalization.py` | SymPy identity, 71 moment cells, 360 closed-form/Monte-Carlo configurations; `repro/outputs/claim1_generalization.json` | `VERIFIED_SCOPED` |
| C2 | Theorem 4.4: past coefficients `c_t<0`, future coefficients `d>0`, with variance and mean-interaction terms | `repro/src/claim2_forgetting.py` | Symbolic coefficient identity, 13,650 finite sign cells, and 135 closed-form/Monte-Carlo configurations; `repro/outputs/claim2_forgetting.json` | `VERIFIED_SCOPED` |
| C3 | Misaligned multi-task error can peak at an intermediate context length | `repro/src/claim3_nonmonotone.py` | 780 theory configurations; 47/252 misaligned multi-task cells have an interior peak, while aligned and single-task controls are monotone | `VERIFIED_SCOPED` |
| C4 | Variance interference decays as `O(1/M)` while mean interference persists | `repro/src/claim4_asymptotic.py` | Tail slope `-0.999219`, limit relative error `2.08e-6`, positive floor `0.661267`, and aligned zero-floor control | `VERIFIED_SCOPED` |
| C5a | Tiny GPT-2 reproduces non-monotone per-task error | `repro/src/claim5a_gpt2_icl.py` on `experiment/gpt2-claim-5a` | The report and figures retain the later external run; the committed JSON records only a 50-step `PARTIAL` run | `RETAINED_REPORT_ONLY` |
| C5b | Qwen2.5-1.5B shows the paper's large Task-A forgetting drop and Task-B negative transfer | `repro/src/claim5b_qwen_realworld.py` on `experiment/qwen-claim-5b` | The report records a divergent 1.5B run; the committed JSON is an earlier Qwen2.5-0.5B `PARTIAL` snapshot | `DIVERGENT_RETAINED_REPORT` |

The four local theorem claims are produced by the fixed harness:

```text
source/main.tex + source figures
  -> independent theory core and clean-room formulas
  -> symbolic identities and finite controls
  -> seeded Monte-Carlo / asymptotic checks
  -> JSON claim reports and publication gate
```

Claims 5a and 5b follow a different path:

```text
paper experiment protocol
  -> branch-specific model code
  -> external or local run report
  -> retained figures / narrative result
```

Those model-scale runs are not silently promoted to fresh-clone evidence. See
[`docs/CLAIM_AUDIT.md`](docs/CLAIM_AUDIT.md) for the exact artifact boundary.

## Repository contents

- `source/` — pinned arXiv source tree and figures; no author implementation was
  released with the paper.
- `repro/src/` — clean-room theory core, four theorem verifiers, two model-scale
  experiment programs, and the publication-gate builder.
- `repro/outputs/` — generated C1–C4 reports plus the branch-captured C5
  `PARTIAL` snapshots.
- `outputs/` — the retained historical baseline evidence and compatibility gate.
- `pages/` and `reports/iccl-repro/` — detailed claim pages and the illustrated
  report, preserved with corrected provenance notes.
- `.trackio/logbook/` — the earlier toy-scale experiment logbook, retained as
  historical provenance rather than merged with the rigorous claim ledger.
- `docs/` — claim, branch, source, manifest, and publication audits.

## Reproduce the local theorem evidence

The repository uses the pinned `uv.lock` environment. Claims 1–4 are CPU-scale
checks; no hours-long model training or 1.5B-parameter inference is launched by
the following command.

```bash
uv sync
uv run python repro/run.py
python -m unittest discover -s repro/tests -v
uv run python repro/src/build_publication_gate.py
```

The fixed harness runs the four theorem verifiers and writes their JSON reports.
The gate builder checks those reports, the focused tests, the source hash, and
the explicitly disclosed C5 artifact boundary. To inspect the model branches
without rerunning them:

```bash
uv run python repro/src/claim5a_gpt2_icl.py  # hours on CPU; not required by the gate
uv run python repro/src/claim5b_qwen_realworld.py  # downloads a 1.5B model; not required by the gate
```

The committed C5 JSON files are provenance snapshots, not claims that either
command has completed on a fresh clone.

## Branch map

| Final branch | Original branch | Purpose |
| --- | --- | --- |
| `main` | `main` | Canonical README, reports, theorem verifiers, generated C1–C4 evidence, and release gate |
| `baseline/toy-reference` | `orx/baseline-uv-env-toy-reference` | Historical 4/10 toy-scale reference and environment setup |
| `audit/theory-claims-1-4` | `orx/theory-rigorous-verification-of-claims-1-4` | Rigorous theorem-level verifier development |
| `experiment/gpt2-claim-5a` | `orx/empirical-gpt-2-icl-non-monotone-claim-5a` | Tiny GPT-2 experiment code and its partial/retained evidence |
| `experiment/qwen-claim-5b` | `orx/empirical-qwen2-5-1-5b-real-world-iccl-claim-5b` | Qwen real-world ICCL code, divergence report, and partial artifact |
| `release/publication` | `orx/release-publication-hf-space-github-report` | Historical publication-surface snapshot |

`main` is the default branch. The generic `orx/` prefix is removed from the
public branch names, while every experiment branch remains available for
provenance. See [`docs/BRANCH_AUDIT.md`](docs/BRANCH_AUDIT.md).

## Source and limitations

- Paper: [arXiv:2605.28705](https://arxiv.org/abs/2605.28705).
- OpenReview: `68AMoK2YNk`.
- Authors: Guangyu Li, Meng Ding, and Lijie Hu.
- `source/main.tex` SHA-256:
  `026aa3bdb1f7903172b9935e6181123260ff6e4dedc3a24965bfe1ce6e5f97f9`.
- The source tree is committed, but the historical arXiv tar hash is retained
  only as provenance; the tar file itself is not in this clone.
- The source paper has no author-code release. `repro/src/` is an independent
  implementation of the finite checks, not author code.
- The theorem checks cover their declared diagonal/Gaussian finite scopes; they
  are not a machine-checked proof of every universal theorem quantifier.
- The report's 34k-step GPT-2 and 1.5B Qwen results need their original raw
  logs/checkpoints to become fresh-clone reproducible. Their current statuses
  remain explicitly separated from C1–C4.

## Citation

```bibtex
@article{li2026understanding,
  title   = {Understanding Generalization and Forgetting in In-Context Continual Learning},
  author  = {Li, Guangyu and Ding, Meng and Hu, Lijie},
  journal = {arXiv preprint arXiv:2605.28705},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.28705}
}
```

Please cite the paper when discussing in-context continual learning. Cite this
repository when referring to the clean-room theorem audits, finite controls,
retained model-branch reports, or their stated limitations.

## Thank you

Thank you to Guangyu Li, Meng Ding, and Lijie Hu for making the paper and its
formal source available. The explicit attention model, theorem statements, and
experimental protocol make it possible to build an independent audit while
keeping the distinction between mathematical evidence, retained reports, and
fresh reproduction visible.

This companion is maintained by
[MachineLearning-Nerd](https://github.com/MachineLearning-Nerd).
