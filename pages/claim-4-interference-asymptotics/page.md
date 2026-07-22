# Claim 4 — variance decays, mean interference persists

---
<!-- trackio-cell
{"type":"markdown","id":"claim4_exact_scope","created_at":"2026-07-22T16:09:06+00:00","title":"Exact paper claim and source scope"}
-->
## Claim — supported by an exact-formula audit

The context-length row of the paper's summary states that variance-induced
interference decays as `O(1/M)`, while mean-induced interference remains
constant for misaligned tasks. This page audits that statement directly from
**Theorem 4.4**, including its finite-`M` coefficients, and checks the
reweighting interpretation in **Remark 4.5**.

Paper source: `https://ar5iv.labs.arxiv.org/html/2605.28705` (retrieved
2026-07-22; scope: Theorem 4.4, Remark 4.5, and Table 1's context-length row;
download SHA-256
`fa6ee7cf7a58098d8377d2ea9db8bd869aee3b2cbe6e0a5e2d6e9b277214e9c4`).

---
<!-- trackio-cell
{"type":"markdown","id":"claim4_exact_method","created_at":"2026-07-22T16:09:06+00:00","title":"Published decomposition and audit method"}
-->
## What was evaluated

Theorem 4.4 assigns `c_t < 0` to tasks `i <= t` and `d > 0` to tasks
`i > t`:

```text
c_t = -[((T-t)M + (T+1-t)) / (t(M+1)(TM+T+1))]
d   = 1 / (TM+T+1)
```

Separating the theorem's displayed interference equation gives

```text
variance(M) = M sum_i alpha_i(M)^2 tr(Sigma_i Gamma^-2 Lambda)
mean(M)     = M^2 ||sum_i alpha_i(M) mu_i||^2_(Gamma^-2 Lambda).
```

The executable audit uses `T=4`, `t=2`, a disclosed diagonal positive-definite
`Lambda`, four disclosed positive diagonal `Sigma_i`, and four disclosed,
misaligned task means. It sweeps `M=1` through `1,048,576`. All parameters and
all 15 sweep rows are stored in
`repro/outputs/claim4_interference_asymptotics.json`; there is no randomness,
fitted model, network access, or hidden input.

The relevant source is visible in
`repro/src/verify_claim4_interference_asymptotics.py`; its core computation is:

```python
alpha = [c_t if i < TASK_T else d for i in range(T)]
variance = M * sum((a * a) * tr for a, tr in zip(alpha, sigma_trace))
weighted_mu = [sum(alpha[i] * mus[i][k] for i in range(T)) for k in range(3)]
mean = M * M * dot_metric(weighted_mu, weighted_mu, metric_diag)
```

---
<!-- trackio-cell
{"type":"code","id":"claim4_exact_run","created_at":"2026-07-22T16:09:06+00:00","title":"Primary exact-formula execution","command":["python3","repro/src/verify_claim4_interference_asymptotics.py"],"exit_code":0,"duration_s":0.1}
-->
````bash
$ python3 repro/src/verify_claim4_interference_asymptotics.py
````

````output
Theorem 4.4 / Remark 4.5 exact-formula audit
source_sha256=fa6ee7cf7a58098d8377d2ea9db8bd869aee3b2cbe6e0a5e2d6e9b277214e9c4
tail variance log-log slope=-0.998661159 (expected -1)
lim M*variance=0.553573681; M=1048576 gives 0.553572610
lim mean interference=0.287989106; M=1048576 gives 0.287988545
aligned negative-control floor ratio=8.397e-14
past_coefficients_negative: True
future_coefficients_positive: True
variance_tail_log_slope_is_minus_one: True
M_times_variance_reaches_limit: True
misaligned_mean_reaches_positive_floor: True
aligned_negative_control_removes_floor: True
verdict: supports
RESULTS_SHA256=1befdbbdefbfa77cb0dfe6749940e5cc120d4d28111c94c7ed8669bde9bdcb08
````

---
<!-- trackio-cell
{"type":"code","id":"claim4_fraction_audit","created_at":"2026-07-22T16:09:06+00:00","title":"Independent rational-arithmetic audit","command":["python3","repro/src/audit_claim4_interference_asymptotics.py"],"exit_code":0,"duration_s":0.1}
-->
````bash
$ python3 repro/src/audit_claim4_interference_asymptotics.py
````

````output
Independent Fraction audit of Theorem 4.4
finite expanded-vs-compact decomposition exact: True
lim M*c_t=-1/4; lim M*d=1/4
lim M*variance=0.553573681
lim mean interference=0.287989106 (>0)
finite_decomposition_exact: True
c_t_negative: True
d_positive: True
mean_limit_strictly_positive: True
scaled_variance_limit_relative_error_lt_1e-7: True
mean_limit_relative_error_lt_1e-7: True
verdict: supports
````

The independent script uses `fractions.Fraction`, re-expands every `i != j`
cross term at four finite context lengths, and proves exact equality with the
compact squared-norm form. It then checks the analytic limits without importing
or calling the primary verifier.

---
<!-- trackio-cell
{"type":"markdown","id":"claim4_exact_result","created_at":"2026-07-22T16:09:06+00:00","title":"Result and negative control"}
-->
## Result

**Supported for the theorem's stated model.** The variance tail has slope
`-0.998661`, and `M * variance` converges to the independently derived positive
constant `0.553573681`, establishing `Theta(1/M)` for this nondegenerate case.
The misaligned mean term converges to the positive floor `0.287989106`.

As a structural negative control, replacing all four task means by the same
aligned mean makes the asymptotic weighted mean cancel; at the largest `M`, its
floor is only `8.397e-14` of the misaligned floor. Thus the persistent term is
caused by task-mean misalignment under the theorem's signed reweighting, not by
the sweep machinery.

Artifacts:

- Primary verifier SHA-256: `5a4276d546a33eeee9d75e82c1b8750443c8836a48c700430dbd86ec406565d0`
- Independent verifier SHA-256: `c32fc8da13f3c4a5f7ec64e9ff1031fea727779e05f28400995c0a84128da39d`
- Saved JSON file SHA-256: `65bc4b79977e762be7da9fb6945e2a576e2dde6023a55e47d89c7e9d81c33e51`
