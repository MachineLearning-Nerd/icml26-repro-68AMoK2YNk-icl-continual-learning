# Source audit and provenance

## Paper

- Title: *Understanding Generalization and Forgetting in In-Context Continual Learning*
- Authors: Guangyu Li, Meng Ding, and Lijie Hu
- OpenReview: `68AMoK2YNk`
- arXiv: [`2605.28705`](https://arxiv.org/abs/2605.28705)
- Pinned source tree: `source/`
- `source/main.tex` SHA-256: `026aa3bdb1f7903172b9935e6181123260ff6e4dedc3a24965bfe1ce6e5f97f9`

The official arXiv record identifies the same title/authors and describes the
paper as a theoretical framework for sequential in-context tasks, inter-task
interference, and bias–variance–interference decomposition.

## Implementation provenance

The paper does not include an author-code repository. The files under
`repro/src/` are a clean-room implementation of the finite formulas and
controls. The source paper, figures, and bibliography are retained under
`source/`; the source archive itself is not committed.

The historical source tar SHA-256 recorded by the original logbook is
`c28cca01207449b7dea91544161b5f8c764ef1001f729efd17073ba7b4101210`, but no
`*.tar` file is present in this clone. A fresh-clone audit should therefore use
the committed source tree and its manifest, not imply that the tar hash can be
recomputed locally.

## Data and model boundary

- C1–C4 use seeded synthetic Gaussian/linear systems and their declared finite
  scopes; no author dataset is required.
- C5a uses the committed tiny-GPT-2 producer, but the reported 34k-step external
  run's checkpoint and raw metrics are absent.
- C5b's producer targets Qwen2.5-1.5B-Instruct, SST-2, and AG News. The
  committed result snapshot targets Qwen2.5-0.5B and is marked `PARTIAL`; the
  later 1.5B report is retained but not raw-artifact reproducible.
- The model-scale commands can download large dependencies or weights and are
  intentionally excluded from the default publication gate.

These boundaries are part of the release decision rather than hidden setup
assumptions.
