#!/usr/bin/env python3
"""Exact-formula audit of Theorem 4.4 and Remark 4.5 (arXiv:2605.28705).

The theorem decomposes the task-t/final-query interference into

  V(M) = M * sum_i alpha_i(M)^2 tr(Sigma_i Gamma^-2 Lambda)
  B(M) = M^2 * ||sum_i alpha_i(M) mu_i||^2_(Gamma^-2 Lambda)

with alpha_i=c_t for i<=t and alpha_i=d for i>t.  This deterministic
audit evaluates the published formula over six orders of magnitude in M and
checks its analytic limits.  It uses only the Python standard library.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


SOURCE_URL = "https://ar5iv.labs.arxiv.org/html/2605.28705"
SOURCE_SHA256 = "fa6ee7cf7a58098d8377d2ea9db8bd869aee3b2cbe6e0a5e2d6e9b277214e9c4"
SOURCE_SCOPE = "Theorem 4.4, Remark 4.5, and the context-length row of Table 1"

T = 4
TASK_T = 2
N_PRETRAIN = 100
LAMBDA_DIAG = [0.8, 1.3, 2.1]
SIGMA_DIAGS = [
    [1.0, 0.5, 1.2],
    [0.7, 1.4, 0.9],
    [1.3, 0.8, 1.1],
    [0.6, 1.2, 1.5],
]
MISALIGNED_MUS = [
    [1.0, 0.2, -0.1],
    [-0.3, 1.1, 0.4],
    [0.5, -0.7, 1.2],
    [-0.8, 0.3, 0.9],
]


def coefficients(M: int) -> tuple[float, float]:
    c_t = -(((T - TASK_T) * M) + (T + 1 - TASK_T)) / (
        TASK_T * (M + 1) * (T * M + T + 1)
    )
    d = 1.0 / (T * M + T + 1)
    return c_t, d


def metric_diag() -> list[float]:
    trace_lambda = sum(LAMBDA_DIAG)
    gamma = [
        (1.0 + 1.0 / N_PRETRAIN) * lam + trace_lambda / N_PRETRAIN
        for lam in LAMBDA_DIAG
    ]
    return [lam / (g * g) for lam, g in zip(LAMBDA_DIAG, gamma)]


def dot_metric(x: list[float], y: list[float], a_diag: list[float]) -> float:
    return sum(a * xi * yi for a, xi, yi in zip(a_diag, x, y))


def parts(M: int, mus: list[list[float]]) -> tuple[float, float, float, float]:
    c_t, d = coefficients(M)
    alpha = [c_t if i < TASK_T else d for i in range(T)]
    a_diag = metric_diag()
    sigma_trace = [sum(a * s for a, s in zip(a_diag, diag)) for diag in SIGMA_DIAGS]
    variance = M * sum((a * a) * tr for a, tr in zip(alpha, sigma_trace))
    weighted_mu = [sum(alpha[i] * mus[i][k] for i in range(T)) for k in range(3)]
    mean = M * M * dot_metric(weighted_mu, weighted_mu, a_diag)
    return c_t, d, variance, mean


def log_slope(xs: list[int], ys: list[float]) -> float:
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx = sum(lx) / len(lx)
    my = sum(ly) / len(ly)
    return sum((x - mx) * (y - my) for x, y in zip(lx, ly)) / sum(
        (x - mx) ** 2 for x in lx
    )


def main() -> int:
    a_diag = metric_diag()
    c_limit = -(T - TASK_T) / (TASK_T * T)
    d_limit = 1.0 / T
    alpha_limits = [c_limit if i < TASK_T else d_limit for i in range(T)]
    sigma_trace = [sum(a * s for a, s in zip(a_diag, diag)) for diag in SIGMA_DIAGS]
    variance_scaled_limit = sum((a * a) * tr for a, tr in zip(alpha_limits, sigma_trace))
    weighted_limit = [
        sum(alpha_limits[i] * MISALIGNED_MUS[i][k] for i in range(T))
        for k in range(3)
    ]
    mean_limit = dot_metric(weighted_limit, weighted_limit, a_diag)

    Ms = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096, 16384, 65536, 262144, 1048576]
    sweep = []
    for M in Ms:
        c_t, d, variance, mean = parts(M, MISALIGNED_MUS)
        sweep.append(
            {
                "M": M,
                "c_t": c_t,
                "d": d,
                "variance": variance,
                "M_times_variance": M * variance,
                "mean_interference": mean,
                "total_interference": variance + mean,
            }
        )

    tail = sweep[-8:]
    slope = log_slope([row["M"] for row in tail], [row["variance"] for row in tail])
    last = sweep[-1]
    rel_var_limit = abs(last["M_times_variance"] - variance_scaled_limit) / variance_scaled_limit
    rel_mean_limit = abs(last["mean_interference"] - mean_limit) / mean_limit

    aligned_mu = [[0.4, -0.2, 0.7] for _ in range(T)]
    _, _, _, aligned_mean_last = parts(Ms[-1], aligned_mu)
    negative_control_ratio = aligned_mean_last / mean_limit

    checks = {
        "past_coefficients_negative": all(row["c_t"] < 0 for row in sweep),
        "future_coefficients_positive": all(row["d"] > 0 for row in sweep),
        "variance_tail_log_slope_is_minus_one": abs(slope + 1.0) < 0.002,
        "M_times_variance_reaches_limit": rel_var_limit < 1e-5,
        "misaligned_mean_reaches_positive_floor": mean_limit > 0.01 and rel_mean_limit < 1e-5,
        "aligned_negative_control_removes_floor": negative_control_ratio < 1e-10,
    }
    result = {
        "claim": "Theorem 4.4 variance interference is O(1/M) while misaligned mean interference persists",
        "source": {
            "url": SOURCE_URL,
            "sha256": SOURCE_SHA256,
            "scope": SOURCE_SCOPE,
            "retrieved_utc": "2026-07-22",
        },
        "parameters": {
            "T": T,
            "t": TASK_T,
            "N": N_PRETRAIN,
            "lambda_diag": LAMBDA_DIAG,
            "sigma_diags": SIGMA_DIAGS,
            "misaligned_task_means": MISALIGNED_MUS,
        },
        "analytic_limits": {
            "limit_M_c_t": c_limit,
            "limit_M_d": d_limit,
            "limit_M_times_variance": variance_scaled_limit,
            "limit_mean_interference": mean_limit,
        },
        "tail_variance_log_log_slope": slope,
        "largest_M_relative_error": {
            "M_times_variance": rel_var_limit,
            "mean_interference": rel_mean_limit,
        },
        "aligned_negative_control_floor_ratio": negative_control_ratio,
        "sweep": sweep,
        "checks": checks,
        "verdict": "supports" if all(checks.values()) else "inconclusive",
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    output = Path(__file__).resolve().parents[1] / "outputs" / "claim4_interference_asymptotics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload + "\n", encoding="utf-8")

    print("Theorem 4.4 / Remark 4.5 exact-formula audit")
    print(f"source_sha256={SOURCE_SHA256}")
    print(f"tail variance log-log slope={slope:.9f} (expected -1)")
    print(f"lim M*variance={variance_scaled_limit:.9f}; M={Ms[-1]} gives {last['M_times_variance']:.9f}")
    print(f"lim mean interference={mean_limit:.9f}; M={Ms[-1]} gives {last['mean_interference']:.9f}")
    print(f"aligned negative-control floor ratio={negative_control_ratio:.3e}")
    for name, passed in checks.items():
        print(f"{name}: {passed}")
    print(f"verdict: {result['verdict']}")
    print("RESULTS_SHA256=" + hashlib.sha256(payload.encode()).hexdigest())
    return 0 if result["verdict"] == "supports" else 1


if __name__ == "__main__":
    raise SystemExit(main())
