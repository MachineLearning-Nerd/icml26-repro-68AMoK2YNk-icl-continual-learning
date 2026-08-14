# Branch audit

## Final branch contract

| Final branch | Original branch | Intake tip | Purpose |
| --- | --- | --- | --- |
| `main` | `main` | `30f9e82473c414eb238bfb52b7d1269d35c25592` | Canonical publication surface and final gate |
| `baseline/toy-reference` | `orx/baseline-uv-env-toy-reference` | `2fa4eca5bbf9db3f831ad0d708dd5911b2da8617` | Historical toy-scale reference |
| `audit/theory-claims-1-4` | `orx/theory-rigorous-verification-of-claims-1-4` | `d21038ea1f0d2d57e5f4b64f1746d8bc37fbd40e` | Theorem verifier development and generated C1–C4 evidence |
| `experiment/gpt2-claim-5a` | `orx/empirical-gpt-2-icl-non-monotone-claim-5a` | `a335e8cdd27f1f09ddd38b7db77287b480e3aefb` | Tiny GPT-2 code and partial artifact |
| `experiment/qwen-claim-5b` | `orx/empirical-qwen2-5-1-5b-real-world-iccl-claim-5b` | `d17e68b1b1b749c66f6de4ccdc30cd55b3f24b32` | Qwen code, divergence report, and partial artifact |
| `release/publication` | `orx/release-publication-hf-space-github-report` | `8048c6889c8e6e2a0c10970862288d812a8e2390` | Historical publication snapshot |

The final default branch is `main`. Branch names describe the purpose of the
work and no `orx/*` references remain after publication cleanup. Every final
branch is retained because each one is evidence provenance, not an abandoned
working copy.

## Branch production paths

- `baseline/toy-reference` established the environment and the earlier 4/10
  toy-scale evidence.
- `audit/theory-claims-1-4` replaced the toy checks with independent symbolic,
  finite, and Monte-Carlo theorem checks.
- `experiment/gpt2-claim-5a` implements the tiny GPT-2 training/evaluation path;
  the committed output is partial and the later external run is report-only.
- `experiment/qwen-claim-5b` implements the Qwen/SST-2/AG News path; its
  committed output is a partial 0.5B snapshot and the later 1.5B report is
  divergent and not raw-artifact reproducible.
- `release/publication` records the earlier static report publication surface.

## Repository identity

- Original repository: `MachineLearning-Nerd/icml26-repro-68AMoK2YNk-icl-continual-learning`
- Final repository: `MachineLearning-Nerd/icml26-in-context-continual-learning`
- Maintainer: `MachineLearning-Nerd`
- Default branch: `main`
