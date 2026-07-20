# In-context continual learning — clean-room reproduction

Work in progress for OpenReview `68AMoK2YNk`, *Understanding Generalization and Forgetting in In-Context Continual Learning*.

The package pins arXiv:2605.28705v1 and will reproduce the three live claims with exact finite-dimensional masked-linear-attention calculations, not a reduced GPT training proxy. No author code was released.

The source requires careful scope: its stated forgetting metric is prediction drift; unchanged-prefix permutations are invariant in the analyzed model; and the appendix contains a self-referential `Gamma` definition and a missing `M` factor in one printed denominator. These are pre-registered for explicit negative controls.
