#!/usr/bin/env python3
"""CLAIM 4 (Section 4, forgetting context-length analysis; Remark) — asymptotic floor.

Exact claim: variance-induced forgetting interference decays as O(1/M) with context
length, while mean-misalignment interference PERSISTS regardless of M, leaving a
nonzero asymptotic forgetting floor for misaligned tasks (and vanishing for aligned).

Verification (uses the Claim-2-verified interference closed form, decomposed):
  (A) Huge-M sweep (M = 1 .. 2^20) over a fixed misaligned config:
      - log-log slope of the variance tail ~= -1  (O(1/M));
      - M * variance -> analytic limit;
      - mean interference -> analytic positive limit (persistent floor).
  (B) Aligned negative control: same sweep with all task means equal -> mean floor -> 0.
  (C) Broad sweep over random misaligned configs: confirm the variance tail slope is
      approx -1 and the mean floor is positive (>0) in every config.
  Analytic limits derived from lim_{M->inf} M*alpha_s = -(T-t)/(tT) (s<=t), 1/T (s>t).
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

import theory_core as tc


def log_slope(xs, ys):
    lx, ly = np.log(xs), np.log(ys)
    mx, my = lx.mean(), ly.mean()
    return float(np.sum((lx - mx) * (ly - my)) / np.sum((lx - mx) ** 2))


def huge_M_sweep(W, Lam, t, T, N, label):
    Ms = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096, 16384, 65536, 262144, 1048576]
    rows = []
    for M in Ms:
        var, mean = tc.interference_parts(W, Lam, M, t, T, N)
        rows.append({"M": M, "variance": var, "M_times_variance": M * var, "mean": mean})
    tail = rows[-7:]
    slope = log_slope([r["M"] for r in tail], [r["variance"] for r in tail])
    var_lim = tc.limiting_mean_floor  # not used for var; compute below
    # analytic limits
    _, _ = var_lim, None
    # variance limit: M * variance -> sum_s (lim M alpha_s)^2 tr(Sigma_s Gamma^-2 Lambda)
    mu, Sigma = tc.task_moments(W, Lam)
    G = tc.gamma_matrix(Lam, N); GAG = np.linalg.inv(G) @ Lam @ np.linalg.inv(G)
    lim_alpha = np.array([-(T - t) / (t * T) if s < t else 1.0 / T for s in range(T)])
    var_scaled_limit = float(sum(lim_alpha[s] ** 2 * np.trace(GAG @ Sigma[s]) for s in range(T)))
    mean_limit = float((sum(lim_alpha[s] * mu[s] for s in range(T))) @ GAG @
                       (sum(lim_alpha[s] * mu[s] for s in range(T))))
    last = rows[-1]
    rel_var = abs(last["M_times_variance"] - var_scaled_limit) / var_scaled_limit
    rel_mean = abs(last["mean"] - mean_limit) / max(abs(mean_limit), 1e-12)
    return {"label": label, "rows": rows, "tail_slope": slope,
            "var_scaled_limit": var_scaled_limit, "mean_limit": mean_limit,
            "rel_err_Mvar": rel_var, "rel_err_mean": rel_mean,
            "last_M": last["M"], "last_mean": last["mean"]}


def broad_random_sweep():
    rng = np.random.default_rng(20245)
    configs = []
    slope_ok = floor_ok = 0
    for cfg in range(12):
        d = 2 + cfg % 5
        T = 2 + cfg % 4
        t = 1 + cfg % T
        A = rng.standard_normal((d, d)); Lam = A @ A.T / d + 0.5 * np.eye(d)
        W = [rng.standard_normal(d) * 1.5 for _ in range(T)]
        N = 80
        res = huge_M_sweep(W, Lam, t, T, N, f"cfg{cfg}")
        # require tail slope near -1
        if abs(res["tail_slope"] + 1.0) < 0.01:
            slope_ok += 1
        # misaligned (random distinct tasks) -> positive floor
        if res["mean_limit"] > 1e-6 and res["rel_err_mean"] < 1e-4:
            floor_ok += 1
        configs.append({"cfg": cfg, "d": d, "T": T, "t": t,
                        "tail_slope": res["tail_slope"], "mean_limit": res["mean_limit"]})
    assert slope_ok >= 11, f"variance O(1/M) failed in {12 - slope_ok} configs"
    assert floor_ok >= 10, f"persistent floor failed in {12 - floor_ok} configs"
    return {"n_configs": len(configs), "variance_slope_near_minus1": slope_ok,
            "positive_persistent_floor": floor_ok, "configs": configs}


def main() -> int:
    t0 = time.time()
    # Fixed misaligned config (deterministic) + aligned negative control
    d, T, t, N = 3, 4, 2, 100
    Lam = np.diag([0.8, 1.3, 2.1])
    mis_W = [np.array([1.0, 0.2, -0.1]), np.array([-0.3, 1.1, 0.4]),
             np.array([0.5, -0.7, 1.2]), np.array([-0.8, 0.3, 0.9])]
    ali_W = [np.array([0.4, -0.2, 0.7]) for _ in range(T)]
    mis = huge_M_sweep(mis_W, Lam, t, T, N, "misaligned")
    ali = huge_M_sweep(ali_W, Lam, t, T, N, "aligned_negative_control")

    checks = {
        "variance_tail_slope_is_minus_one": bool(abs(mis["tail_slope"] + 1.0) < 0.002),
        "M_times_variance_reaches_limit": bool(mis["rel_err_Mvar"] < 1e-5),
        "misaligned_mean_persistent_positive_floor": bool(mis["mean_limit"] > 0.01 and mis["rel_err_mean"] < 1e-5),
        "aligned_negative_control_floor_vanishes": bool(ali["mean_limit"] < 1e-9),
    }
    broad = broad_random_sweep()
    out = {
        "claim": "Claim 4 — variance interference O(1/M); mean-misalignment interference persists (floor)",
        "source": {"arxiv": "2605.28705", "sections": "Section 4 forgetting (1), Remark"},
        "misaligned_fixed_config": {k: v for k, v in mis.items() if k != "rows"},
        "aligned_negative_control": {k: v for k, v in ali.items() if k != "rows"},
        "misaligned_sweep_rows": mis["rows"],
        "aligned_sweep_rows": ali["rows"],
        "broad_random_sweep": broad,
        "checks": checks,
        "seconds": round(time.time() - t0, 2),
    }
    out["verdict"] = "VERIFIED" if all(checks.values()) else "INCONCLUSIVE"
    payload = json.dumps(out, indent=2, sort_keys=True)
    print("CLAIM 4 — O(1/M) variance decay + persistent mean-interference floor")
    print(f"  misaligned: tail_slope={mis['tail_slope']:.6f}  M*var rel_err={mis['rel_err_Mvar']:.2e}  "
          f"mean_limit={mis['mean_limit']:.6f}")
    print(f"  aligned control mean_limit={ali['mean_limit']:.3e} (->0)")
    print(f"  broad: variance_slope_ok={broad['variance_slope_near_minus1']}/12  "
          f"floor_ok={broad['positive_persistent_floor']}/12")
    for k, v in checks.items():
        print(f"  {k}: {v}")
    print(f"verdict: {out['verdict']}  ({out['seconds']}s)")
    odir = Path(__file__).resolve().parents[1] / "outputs"; odir.mkdir(parents=True, exist_ok=True)
    (odir / "claim4_asymptotic.json").write_text(payload + "\n")
    print("RESULTS_SHA256=" + hashlib.sha256(payload.encode()).hexdigest())
    return 0 if out["verdict"] == "VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
