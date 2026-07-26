#!/usr/bin/env python3
"""CLAIM 2 (Theorem 4.4, forgetting / interference) — rigorous verification.

Exact claim: after all T tasks,
  Yhat_{t,q} - yhat_{t,q} = x_q^T Gamma^{-1}( sum_{i<=t} c_t S_i + sum_{i>t} d S_i ),
  c_t = -(((T-t)M) + (T+1-t)) / ( t (M+1) (T M + T +1) )  < 0   (past tasks, NEGATIVE coeff)
  d   =  1 / (T M + T +1)                                  > 0   (future tasks, POSITIVE coeff)
and the interference decomposes into intra-task variance terms tr(Sigma_i Gamma^{-2} Lambda)
and inter-task mean-interaction terms tr(mu_i mu_j^T Gamma^{-2} Lambda) (i != j).

Three routes:
  (A) SYMBOLIC: derive the coefficients from the two prediction rules and confirm
      1/(T(M+1)+1) - 1/(t(M+1)) == c_t (so past tasks carry the negative coefficient);
      confirm the reweighting structure (past<0, future>0).
  (B) EXHAUSTIVE SIGN + domain argument: c_t<0 and d>0 for every (T,t,M) in a large
      finite grid, plus the analytic positivity argument.
  (C) CLOSED-FORM vs RAW-MC over a broad sweep.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import sympy as sp

import theory_core as tc


def symbolic_coefficients() -> dict:
    M, T, t = sp.symbols("M T t", positive=True, integers=True)
    # final coefficient for every task: 1/(T(M+1)+1) ; task-t coefficient: 1/(t(M+1))
    final = 1 / (T * (M + 1) + 1)
    task_t = 1 / (t * (M + 1))
    c_t_expr = -(sp.expand((T - t) * M) + (T + 1 - t)) / (t * (M + 1) * (T * M + T + 1))
    # the coefficient of S_i (i<=t) in (Yhat - yhat) is final - task_t ; must equal c_t
    diff = sp.simplify(sp.expand(final - task_t) - sp.expand(c_t_expr))
    assert diff == 0, f"coefficient identity c_t failed: {diff}"
    # d is just the final coefficient for future tasks
    d_expr = 1 / (T * M + T + 1)
    assert sp.simplify(final - d_expr) == 0
    # numerator of c_t factorizes as (T-t)M + (T+1-t): clearly > 0 for 1<=t<=T, M>=0
    numer = sp.expand(-c_t_expr * t * (M + 1) * (T * M + T + 1))
    assert sp.simplify(numer - ((T - t) * M + (T + 1 - t))) == 0
    return {"coefficient_c_t_identity_holds": True, "coefficient_d_identity_holds": True,
            "c_t_numerator": str((T - t) * M + (T + 1 - t)) + "  (>0 for 1<=t<=T, M>=0)"}


def exhaustive_sign_check() -> dict:
    bad_c, bad_d, total = 0, 0, 0
    for T in range(1, 21):
        for M in range(0, 65):
            for t in range(1, T + 1):
                c_t, d = tc.forgetting_coefficients(T, t, M)
                total += 1
                if not (c_t < 0):
                    bad_c += 1
                if not (d > 0):
                    bad_d += 1
    assert bad_c == 0 and bad_d == 0, f"sign violations c_t={bad_c} d={bad_d}"
    return {"grid_T_up_to": 20, "grid_M_up_to": 64, "total_(T,t,M)_checked": total,
            "c_t_negative_violations": bad_c, "d_positive_violations": bad_d,
            "analytic_argument": "c_t numerator = (T-t)M + (T+1-t) >= (T+1-t) >= 1 > 0 in domain; "
                                  "d = 1/(TM+T+1) > 0"}


def closed_vs_mc_sweep() -> dict:
    rng = np.random.default_rng(20243)
    records, max_rel, max_id = [], 0.0, 0.0
    cfg = 0
    for d in (2, 4, 6):
        for T in (2, 4, 6):
            for M in (1, 2, 4, 8, 16):
                A = rng.standard_normal((d, d)); Lam = A @ A.T / d + 0.5 * np.eye(d)
                W = [rng.standard_normal(d) * 1.5 for _ in range(T)]
                for t in (1, T // 2 + 1, T):
                    if t > T:
                        continue
                    N = 64
                    closed = tc.interference_error_closed(W, Lam, M, t, T, N)
                    paper = tc.interference_error_paper(W, Lam, M, t, T, N)
                    mc = tc.mc_interference_error(W, Lam, M, t, T, N, n_trials=20000, seed=7000 + cfg)
                    cfg += 1
                    id_err = abs(closed - paper)
                    rel = abs(closed - mc) / max(abs(closed), 1e-9)
                    max_rel = max(max_rel, rel); max_id = max(max_id, id_err)
                    records.append({"d": d, "T": T, "M": M, "t": t, "closed": closed,
                                    "paper": paper, "mc": mc, "identity_err": id_err,
                                    "abs_err": abs(closed - mc), "rel_err": rel})
    assert max_id < 1e-9, f"closed!=paper max err {max_id}"
    # MC is corroboration: pass if relative error small OR absolute error small (small-magnitude configs)
    bad = [r for r in records if r["rel_err"] >= 0.06 and r["abs_err"] >= 0.03]
    assert not bad, f"closed vs MC failures: {bad[:3]}"
    return {"sweep_configs": len(records), "trials_per_config": 20000,
            "max_closed_vs_paper_err": float(max_id),
            "max_closed_vs_mc_rel_err": float(max_rel), "sample_records": records[:6]}

def main() -> int:
    t0 = time.time()
    out = {
        "claim": "Claim 2 — Theorem 4.4 forgetting: reweighting (c_t<0 past, d>0 future) + "
                 "intra-task variance + inter-task mean interaction",
        "source": {"arxiv": "2605.28705", "theorem": "forgetting (Theorem 4.4)",
                   "main_tex_sha256": "026aa3bdb1f7903172b9935e6181123260ff6e4dedc3a24965bfe1ce6e5f97f9"},
        "route_A_symbolic_coefficients": symbolic_coefficients(),
        "route_B_exhaustive_signs": exhaustive_sign_check(),
        "route_C_closed_vs_mc_sweep": closed_vs_mc_sweep(),
        "seconds": round(time.time() - t0, 2),
    }
    out["verdict"] = "VERIFIED"
    payload = json.dumps(out, indent=2, sort_keys=True)
    print("CLAIM 2 — Theorem 4.4 (forgetting / interference)")
    print("Route A (symbolic coefficients):", out["route_A_symbolic_coefficients"])
    print("Route B (exhaustive signs):", out["route_B_exhaustive_signs"])
    print("Route C (closed vs MC):", {k: v for k, v in out["route_C_closed_vs_mc_sweep"].items() if k != "sample_records"})
    print(f"verdict: {out['verdict']}  ({out['seconds']}s)")
    odir = Path(__file__).resolve().parents[1] / "outputs"; odir.mkdir(parents=True, exist_ok=True)
    (odir / "claim2_forgetting.json").write_text(payload + "\n")
    print("RESULTS_SHA256=" + hashlib.sha256(payload.encode()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
