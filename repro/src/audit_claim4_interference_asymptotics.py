#!/usr/bin/env python3
"""Independent rational-arithmetic audit for the Theorem 4.4 evidence."""

from fractions import Fraction as F


T, TASK_T, N = 4, 2, 100
LAM = [F(4, 5), F(13, 10), F(21, 10)]
SIGMA = [
    [F(1), F(1, 2), F(6, 5)],
    [F(7, 10), F(7, 5), F(9, 10)],
    [F(13, 10), F(4, 5), F(11, 10)],
    [F(3, 5), F(6, 5), F(3, 2)],
]
MUS = [
    [F(1), F(1, 5), F(-1, 10)],
    [F(-3, 10), F(11, 10), F(2, 5)],
    [F(1, 2), F(-7, 10), F(6, 5)],
    [F(-4, 5), F(3, 10), F(9, 10)],
]


def coeffs(M: int) -> tuple[F, F]:
    c = -F((T - TASK_T) * M + T + 1 - TASK_T, TASK_T * (M + 1) * (T * M + T + 1))
    d = F(1, T * M + T + 1)
    return c, d


def main() -> int:
    trace_lam = sum(LAM)
    gamma = [(F(1) + F(1, N)) * x + trace_lam / N for x in LAM]
    metric = [x / (g * g) for x, g in zip(LAM, gamma)]
    sigma_trace = [sum(a * s for a, s in zip(metric, row)) for row in SIGMA]

    # Independently prove the paper's expanded double-sum equals one squared
    # weighted-mean norm at representative finite M values.
    decomposition_exact = True
    for M in [1, 3, 16, 127]:
        c, d = coeffs(M)
        alpha = [c if i < TASK_T else d for i in range(T)]
        diagonal_mean = sum(
            M * M * alpha[i] * alpha[i] * sum(metric[k] * MUS[i][k] ** 2 for k in range(3))
            for i in range(T)
        )
        cross_mean = sum(
            M * M * alpha[i] * alpha[j] * sum(metric[k] * MUS[i][k] * MUS[j][k] for k in range(3))
            for i in range(T)
            for j in range(T)
            if i != j
        )
        weighted = [sum(alpha[i] * MUS[i][k] for i in range(T)) for k in range(3)]
        compact_mean = M * M * sum(metric[k] * weighted[k] ** 2 for k in range(3))
        decomposition_exact &= diagonal_mean + cross_mean == compact_mean

    c_inf = -F(T - TASK_T, TASK_T * T)
    d_inf = F(1, T)
    alpha_inf = [c_inf if i < TASK_T else d_inf for i in range(T)]
    var_limit = sum(alpha_inf[i] ** 2 * sigma_trace[i] for i in range(T))
    weighted_inf = [sum(alpha_inf[i] * MUS[i][k] for i in range(T)) for k in range(3)]
    mean_limit = sum(metric[k] * weighted_inf[k] ** 2 for k in range(3))

    large_M = 10**8
    c, d = coeffs(large_M)
    alpha = [c if i < TASK_T else d for i in range(T)]
    variance = F(large_M) * sum(alpha[i] ** 2 * sigma_trace[i] for i in range(T))
    weighted = [sum(alpha[i] * MUS[i][k] for i in range(T)) for k in range(3)]
    mean = F(large_M**2) * sum(metric[k] * weighted[k] ** 2 for k in range(3))

    checks = {
        "finite_decomposition_exact": decomposition_exact,
        "c_t_negative": c < 0,
        "d_positive": d > 0,
        "mean_limit_strictly_positive": mean_limit > 0,
        "scaled_variance_limit_relative_error_lt_1e-7": abs(float(large_M * variance / var_limit) - 1.0) < 1e-7,
        "mean_limit_relative_error_lt_1e-7": abs(float(mean / mean_limit) - 1.0) < 1e-7,
    }
    print("Independent Fraction audit of Theorem 4.4")
    print(f"finite expanded-vs-compact decomposition exact: {decomposition_exact}")
    print(f"lim M*c_t={c_inf}; lim M*d={d_inf}")
    print(f"lim M*variance={float(var_limit):.9f}")
    print(f"lim mean interference={float(mean_limit):.9f} (>0)")
    for name, passed in checks.items():
        print(f"{name}: {passed}")
    verdict = all(checks.values())
    print(f"verdict: {'supports' if verdict else 'inconclusive'}")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
