# Claim 1 — Interference


---
<!-- trackio-cell
{"type": "code", "id": "cell_e7eee26059f1", "created_at": "2026-07-20T08:45:29+00:00", "title": "Full formula and sampling audit", "command": ["python", "repro/src/verify_iccl.py", "--output", "outputs/full_evidence.json"], "exit_code": 0, "duration_s": 4.772}
-->
````bash
$ python repro/src/verify_iccl.py --output outputs/full_evidence.json
````

exit 0 · 4.8s


````python title=verify_iccl.py
#!/usr/bin/env python3
"""Independent finite-moment audit for arXiv:2605.28705v1 (standard library)."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
from pathlib import Path

PAPER = "68AMoK2YNk"
SOURCE = {"main.tex": "026aa3bdb1f7903172b9935e6181123260ff6e4dedc3a24965bfe1ce6e5f97f9"}


def source_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2] / "source"
    got = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE}
    assert got == SOURCE, got
    return got


def gamma_diag(lam: list[float], n_train: int) -> list[float]:
    total = sum(lam)
    return [value + (value + total) / n_train for value in lam]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def squared(a: list[float]) -> float:
    return dot(a, a)


def add_scaled(out: list[float], vector: list[float], scale: float) -> None:
    for index, value in enumerate(vector):
        out[index] += scale * value


def task_prediction_weights(t: int, M: int) -> float:
    return 1.0 / (t * (M + 1))


def final_weight(T: int, M: int) -> float:
    return 1.0 / (T * (M + 1) + 1)


def source_interference_expanded(mu: list[list[float]], variances: list[list[float]], lam: list[float], M: int, t: int, *, literal_typo: bool = False) -> float:
    """Source Theorem 4.2 expanded diagonal form, including its cross terms."""
    T, gamma = len(mu), gamma_diag(lam, n_train=23)
    denom = (T + M + 1) if literal_typo else (T * M + T + 1)
    c = -((T - t) * M + (T + 1 - t)) / (t * (M + 1) * denom)
    d = 1.0 / denom
    alpha = [c if i < t else d for i in range(T)]
    result = 0.0
    for coordinate, eig in enumerate(lam):
        scale = eig / gamma[coordinate] ** 2
        for i in range(T):
            result += alpha[i] ** 2 * (M * variances[i][coordinate] + M * M * mu[i][coordinate] ** 2) * scale
        for i in range(T):
            for j in range(T):
                if i != j:
                    result += M * M * alpha[i] * alpha[j] * mu[i][coordinate] * mu[j][coordinate] * scale
    return result


def independent_interference(mu: list[list[float]], variances: list[list[float]], lam: list[float], M: int, t: int) -> float:
    """Independent calculation from mean/covariance of B=sum alpha_i S_i."""
    T, gamma = len(mu), gamma_diag(lam, n_train=23)
    c = -((T - t) * M + (T + 1 - t)) / (t * (M + 1) * (T * M + T + 1))
    d = 1.0 / (T * M + T + 1)
    alpha = [c if i < t else d for i in range(T)]
    value = 0.0
    for coordinate, eig in enumerate(lam):
        mean_B = M * sum(alpha[i] * mu[i][coordinate] for i in range(T))
        var_B = M * sum(alpha[i] ** 2 * variances[i][coordinate] for i in range(T))
        value += eig / gamma[coordinate] ** 2 * (var_B + mean_B * mean_B)
    return value


def generalization_error(prefix: list[list[float]], target: list[float], variance: float, M: int, lam: list[float], n_train: int) -> float:
    """Source Theorem 4.1 in its diagonal basis, with zero irreducible error."""
    t, gamma = len(prefix), gamma_diag(lam, n_train)
    alpha = M / (t * (M + 1))
    summed = [sum(vector[i] for vector in prefix) for i in range(len(lam))]
    variance_term = M * t * variance * sum(eig / value**2 for eig, value in zip(lam, gamma)) / (t * t * (M + 1) ** 2)
    bias = sum((alpha * summed[i] / gamma[i] - target[i] / lam[i]) ** 2 * lam[i] for i in range(len(lam)))
    return variance_term + bias


def aggregation_audit() -> dict:
    rng = random.Random(6801)
    changed_uniform = changed_causal = 0
    max_causal_difference = 0.0
    for _ in range(500):
        T, M, t, d = 5, 7, 3, 3
        values = [[rng.uniform(-2, 2) for _ in range(d)] for _ in range(T)]
        altered = [row[:] for row in values]
        for future in range(t, T):
            altered[future] = [value + rng.uniform(0.2, 1.0) for value in altered[future]]
        uniform = [sum(row[i] for row in values) / (T * (M + 1) + 1) for i in range(d)]
        uniform_changed = [sum(row[i] for row in altered) / (T * (M + 1) + 1) for i in range(d)]
        causal = [sum(row[i] for row in values[:t]) / (t * (M + 1)) for i in range(d)]
        causal_changed = [sum(row[i] for row in altered[:t]) / (t * (M + 1)) for i in range(d)]
        changed_uniform += squared([a - b for a, b in zip(uniform, uniform_changed)]) > 1e-16
        gap = math.sqrt(squared([a - b for a, b in zip(causal, causal_changed)]))
        max_causal_difference = max(max_causal_difference, gap)
        changed_causal += gap > 1e-16
    assert changed_uniform == 500 and changed_causal == 0
    return {"future_task_mutations": 500, "uniform_predictions_changed": changed_uniform,
            "causal_prefix_predictions_changed": changed_causal, "max_causal_prefix_difference": max_causal_difference}


def identity_audit() -> dict:
    rng, max_error, literal_failures, systems = random.Random(6802), 0.0, 0, 2400
    for system in range(systems):
        T, d, M = 2 + system % 5, 2 + system % 3, 1 + (system // 5) % 11
        lam = [0.4 + rng.random() * 2.0 for _ in range(d)]
        mu = [[rng.uniform(-2, 2) for _ in range(d)] for _ in range(T)]
        variances = [[0.1 + rng.random() for _ in range(d)] for _ in range(T)]
        for t in range(1, T + 1):
            source = source_interference_expanded(mu, variances, lam, M, t)
            independent = independent_interference(mu, variances, lam, M, t)
            max_error = max(max_error, abs(source - independent))
            literal = source_interference_expanded(mu, variances, lam, M, t, literal_typo=True)
            literal_failures += abs(literal - independent) > 1e-10
    assert max_error < 1e-11 and literal_failures > 0
    return {"random_systems": systems, "max_expanded_vs_independent_error": max_error,
            "negative_control_literal_T_plus_M_plus_1_failures": literal_failures}


def monte_carlo_moment_audit() -> dict:
    """Independent sampling readback of the source interference moment identity."""
    rng, cells, trials = random.Random(6803), 12, 60_000
    max_z, max_relative_error = 0.0, 0.0
    for cell in range(cells):
        T, M, t, d = 3 + cell % 3, 2 + cell % 5, 1 + cell % (3 + cell % 3), 2
        if t > T:
            t = T
        lam = [0.7 + rng.random() for _ in range(d)]
        mu = [[rng.uniform(-1.2, 1.2) for _ in range(d)] for _ in range(T)]
        variances = [[0.15 + rng.random() for _ in range(d)] for _ in range(T)]
        gamma = gamma_diag(lam, n_train=23)
        c = -((T - t) * M + (T + 1 - t)) / (t * (M + 1) * (T * M + T + 1))
        alpha = [c if index < t else 1.0 / (T * M + T + 1) for index in range(T)]
        direct = independent_interference(mu, variances, lam, M, t)
        values = []
        for _ in range(trials):
            prediction_difference = 0.0
            for coordinate in range(d):
                mean_b = M * sum(alpha[i] * mu[i][coordinate] for i in range(T))
                var_b = M * sum(alpha[i] ** 2 * variances[i][coordinate] for i in range(T))
                b = rng.gauss(mean_b, math.sqrt(var_b))
                x = rng.gauss(0.0, math.sqrt(lam[coordinate]))
                prediction_difference += x * b / gamma[coordinate]
            values.append(prediction_difference * prediction_difference)
        empirical = sum(values) / trials
        variance_estimate = sum((value - empirical) ** 2 for value in values) / (trials - 1)
        standard_error = math.sqrt(variance_estimate / trials)
        max_z = max(max_z, abs(empirical - direct) / standard_error)
        max_relative_error = max(max_relative_error, abs(empirical - direct) / direct)
    assert max_z < 4.0
    return {"cells": cells, "trials_per_cell": trials, "total_trials": cells * trials,
            "maximum_absolute_z_score": max_z, "maximum_relative_error": max_relative_error}


def transfer_audit() -> dict:
    # Full 361-angle construction: aligned history reduces variance; anti-alignment introduces bias.
    M, lam, n_train, variance = 9, [1.0, 1.0], 10_000_000, 0.25
    target = [1.0, 0.0]
    direct = generalization_error([target], target, variance, M, lam, n_train)
    positive = negative = 0
    min_delta, max_delta = float("inf"), -float("inf")
    for index in range(361):
        angle = math.pi * index / 360.0
        history = [math.cos(angle), math.sin(angle)]
        with_history = generalization_error([history, target], target, variance, M, lam, n_train)
        delta = with_history - direct
        positive += delta < -1e-12
        negative += delta > 1e-12
        min_delta, max_delta = min(min_delta, delta), max(max_delta, delta)
    assert positive > 0 and negative > 0
    return {"angle_cells": 361, "positive_transfer_cells": positive, "negative_transfer_cells": negative,
            "minimum_risk_change": min_delta, "maximum_risk_change": max_delta,
            "baseline_target_risk": direct}


def order_and_long_prompt_audit() -> dict:
    # Task position changes the causal prefix, while a fixed prefix's internal permutation is sum-invariant.
    M, lam, variance = 6, [1.0, 1.0], 0.2
    tasks = [[1.0, 0.0], [-1.0, 0.2], [0.1, 1.0], [-0.3, -0.8], [0.7, -0.4]]
    target = tasks[0]
    risks, drifts = [], []
    for order in itertools.permutations(tasks):
        position = order.index(target) + 1
        prefix = list(order[:position])
        risks.append(generalization_error(prefix, target, variance, M, lam, 10_000))
        drifts.append(independent_interference(list(order), [ [variance, variance] for _ in order], lam, M, position))
    # Adding anti-aligned history yields a strict long-prompt degradation construction.
    long_risks = [generalization_error([target] + [[-1.0, 0.0]] * k, target, variance, M, lam, 10_000) for k in range(22)]
    fixed_prefix = [tasks[1], tasks[2], target]
    fixed_prefix_risks = {round(generalization_error(list(order), target, variance, M, lam, 10_000), 14)
                          for order in itertools.permutations(fixed_prefix[:-1])}
    assert max(risks) > min(risks) and max(drifts) > min(drifts)
    assert all(long_risks[i + 1] > long_risks[i] for i in range(len(long_risks) - 1))
    assert len(fixed_prefix_risks) == 1
    return {"task_permutations": math.factorial(5), "generalization_range": max(risks) - min(risks),
            "prediction_drift_range": max(drifts) - min(drifts), "long_prompt_cells": len(long_risks),
            "long_prompt_start_risk": long_risks[0], "long_prompt_end_risk": long_risks[-1],
            "fixed_prefix_permutation_invariant": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/full_evidence.json"))
    args = parser.parse_args()
    result = {"paper_id": PAPER, "source": {"arxiv": "2605.28705v1", "tar_sha256": "c28cca01207449b7dea91544161b5f8c764ef1001f729efd17073ba7b4101210", "files": source_manifest()},
              "claims": {"claim_1_interference": aggregation_audit(), "claim_2_positive_negative_transfer": transfer_audit(),
                         "claim_3_order_and_long_prompt": order_and_long_prompt_audit()}, "independent_formula_audit": identity_audit(),
              "monte_carlo_readback": monte_carlo_moment_audit(),
              "scope_disclosures": ["Theorem 4.2 metric is prediction drift, not target-risk increase.", "Fixed-prefix permutations are sum-invariant in the analyzed model.", "Appendix literal denominator T+M+1 is rejected; theorem-consistent TM+T+1 is used."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

````


````json title=full_evidence.json
{
  "claims": {
    "claim_1_interference": {
      "causal_prefix_predictions_changed": 0,
      "future_task_mutations": 500,
      "max_causal_prefix_difference": 0.0,
      "uniform_predictions_changed": 500
    },
    "claim_2_positive_negative_transfer": {
      "angle_cells": 361,
      "baseline_target_risk": 0.055000027000068855,
      "maximum_risk_change": 0.9674999594999373,
      "minimum_risk_change": -0.022499986500006085,
      "negative_transfer_cells": 326,
      "positive_transfer_cells": 35
    },
    "claim_3_order_and_long_prompt": {
      "fixed_prefix_permutation_invariant": true,
      "generalization_range": 1.1549746663545082,
      "long_prompt_cells": 22,
      "long_prompt_end_risk": 3.1670200516991933,
      "long_prompt_start_risk": 0.06943189400248316,
      "prediction_drift_range": 0.4992313730225818,
      "task_permutations": 120
    }
  },
  "independent_formula_audit": {
    "max_expanded_vs_independent_error": 6.217248937900877e-15,
    "negative_control_literal_T_plus_M_plus_1_failures": 9600,
    "random_systems": 2400
  },
  "monte_carlo_readback": {
    "cells": 12,
    "maximum_absolute_z_score": 1.672334667931274,
    "maximum_relative_error": 0.011734351861158364,
    "total_trials": 720000,
    "trials_per_cell": 60000
  },
  "paper_id": "68AMoK2YNk",
  "scope_disclosures": [
    "Theorem 4.2 metric is prediction drift, not target-risk increase.",
    "Fixed-prefix permutations are sum-invariant in the analyzed model.",
    "Appendix literal denominator T+M+1 is rejected; theorem-consistent TM+T+1 is used."
  ],
  "source": {
    "arxiv": "2605.28705v1",
    "files": {
      "main.tex": "026aa3bdb1f7903172b9935e6181123260ff6e4dedc3a24965bfe1ce6e5f97f9"
    },
    "tar_sha256": "c28cca01207449b7dea91544161b5f8c764ef1001f729efd17073ba7b4101210"
  }
}

````


````output
{
  "claims": {
    "claim_1_interference": {
      "causal_prefix_predictions_changed": 0,
      "future_task_mutations": 500,
      "max_causal_prefix_difference": 0.0,
      "uniform_predictions_changed": 500
    },
    "claim_2_positive_negative_transfer": {
      "angle_cells": 361,
      "baseline_target_risk": 0.055000027000068855,
      "maximum_risk_change": 0.9674999594999373,
      "minimum_risk_change": -0.022499986500006085,
      "negative_transfer_cells": 326,
      "positive_transfer_cells": 35
    },
    "claim_3_order_and_long_prompt": {
      "fixed_prefix_permutation_invariant": true,
      "generalization_range": 1.1549746663545082,
      "long_prompt_cells": 22,
      "long_prompt_end_risk": 3.1670200516991933,
      "long_prompt_start_risk": 0.06943189400248316,
      "prediction_drift_range": 0.4992313730225818,
      "task_permutations": 120
    }
  },
  "independent_formula_audit": {
    "max_expanded_vs_independent_error": 6.217248937900877e-15,
    "negative_control_literal_T_plus_M_plus_1_failures": 9600,
    "random_systems": 2400
  },
  "monte_carlo_readback": {
    "cells": 12,
    "maximum_absolute_z_score": 1.672334667931274,
    "maximum_relative_error": 0.011734351861158364,
    "total_trials": 720000,
    "trials_per_cell": 60000
  },
  "paper_id": "68AMoK2YNk",
  "scope_disclosures": [
    "Theorem 4.2 metric is prediction drift, not target-risk increase.",
    "Fixed-prefix permutations are sum-invariant in the analyzed model.",
    "Appendix literal denominator T+M+1 is rejected; theorem-consistent TM+T+1 is used."
  ],
  "source": {
    "arxiv": "2605.28705v1",
    "files": {
      "main.tex": "026aa3bdb1f7903172b9935e6181123260ff6e4dedc3a24965bfe1ce6e5f97f9"
    },
    "tar_sha256": "c28cca01207449b7dea91544161b5f8c764ef1001f729efd17073ba7b4101210"
  }
}

````


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_828677343a30", "created_at": "2026-07-20T08:45:43+00:00", "title": "Outcome"}
-->
Verified: changing future task statistics changes all 500 uniform predictions but changes zero causal-prefix predictions. The 2400-system theorem expansion agrees with an independent covariance calculation to 6.22e-15, and a 720000-draw sampling audit has maximum 1.673 standard errors.
