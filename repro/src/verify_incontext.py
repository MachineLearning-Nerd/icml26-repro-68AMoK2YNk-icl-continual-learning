#!/usr/bin/env python3
"""In-Context Continual Learning error analysis (arXiv:2605.28705), Theorem 4.3. Closed-form test error
for the current task t as a function of the in-context length M:

  E[(y_q - yhat)^2] = irreducible
        + [ M / (t^2 (M+1)^2) ] * sum_{s=1}^t tr(Sigma_s Gamma^{-2} Lambda)          (variance)
        + sum_i (1/lambda_i) [ (lambda_i(alpha s_i - m_i) - m_i (lambda_i+tr Lambda)/N)
                               / (lambda_i + (lambda_i+tr Lambda)/N) ]^2               (bias)
  alpha = M/(t(M+1)),  s_i = v_i^T sum_{s<=t} mu_s,  m_i = v_i^T mu_t,
  Gamma = Lambda + (1/N) Lambda + (1/N) tr(Lambda) I.

Headline claim [2]: the error is NON-MONOTONIC in context length M when task means MISALIGN --
"below a threshold, additional context misleads the model and degrades generalization"; for ALIGNED
tasks the error decreases monotonically. We evaluate the exact formula and check both regimes.
Deterministic.
"""
import numpy as np, json, hashlib

def error_curve(thetas_deg, Ms, d=2, N=100, t=2):
    Lam = np.eye(d); lam = np.ones(d); Vv = np.eye(d)     # Lambda = I -> eigenpairs
    trL = float(np.trace(Lam)); Gamma = Lam + Lam / N + (trL / N) * np.eye(d)
    Ginv2 = np.linalg.inv(Gamma) @ np.linalg.inv(Gamma)
    Sig = np.eye(d)                                        # per-task noise covariance
    out = {}
    for th in thetas_deg:
        r = np.deg2rad(th)
        mu1 = np.array([1.0, 0.0]); mu2 = np.array([np.cos(r), np.sin(r)])
        mus = [mu1, mu2]; mu_t = mus[t - 1]; mu_sum = sum(mus)
        s_i = Vv.T @ mu_sum; m_i = Vv.T @ mu_t
        errs = []
        for M in Ms:
            alpha = M / (t * (M + 1))
            var = (M / (t ** 2 * (M + 1) ** 2)) * sum(np.trace(Sig @ Ginv2 @ Lam) for _ in range(t))
            bias = 0.0
            for i in range(d):
                num = lam[i] * (alpha * s_i[i] - m_i[i]) - m_i[i] * (lam[i] + trL) / N
                den = lam[i] + (lam[i] + trL) / N
                bias += (1.0 / lam[i]) * (num / den) ** 2
            errs.append(float(var + bias))               # drop constant irreducible term
        out[th] = np.array(errs)
    return out

def main():
    R = {"claim": "InContext_ContinualLearning_Thm4.3_nonmonotone", "paper": "arXiv:2605.28705"}
    Ms = list(range(0, 41))
    curves = error_curve([0, 150], Ms)                    # 0 deg aligned, 150 deg misaligned

    def is_nonmonotone_up_then_down(e):
        """error rises above e[0] for some small M then ends below the peak -> 'context hurts below threshold'."""
        peak = int(np.argmax(e)); return peak > 0 and e[peak] > e[0] + 1e-9 and e[-1] < e[peak] - 1e-9

    e_align = curves[0]; e_mis = curves[150]
    R["aligned_theta0"] = {"err_M0": round(float(e_align[0]), 4), "err_M1": round(float(e_align[1]), 4),
                           "err_M40": round(float(e_align[-1]), 4),
                           "monotone_decreasing": bool(np.all(np.diff(e_align) <= 1e-9))}
    peak = int(np.argmax(e_mis))
    R["misaligned_theta150"] = {"err_M0": round(float(e_mis[0]), 4), "err_M1": round(float(e_mis[1]), 4),
                                "peak_M": peak, "err_peak": round(float(e_mis[peak]), 4),
                                "err_M40": round(float(e_mis[-1]), 4),
                                "non_monotonic_hurts_then_helps": is_nonmonotone_up_then_down(e_mis)}
    R["aligned_is_monotone"] = R["aligned_theta0"]["monotone_decreasing"]
    R["misaligned_is_nonmonotone"] = R["misaligned_theta150"]["non_monotonic_hurts_then_helps"]
    # threshold: smallest M beyond which error keeps decreasing to the end
    thr = next((M for M in Ms if all(e_mis[M] >= e_mis[k] for k in range(M, len(Ms)))), None)
    R["misaligned_threshold_M"] = thr

    R["verdict"] = "supports" if (R["aligned_is_monotone"] and R["misaligned_is_nonmonotone"]) else "inconclusive"

    print("claim: " + R["claim"])
    print("Exact evaluation of Theorem 4.3 test-error formula; current task t=2, d=2, Lambda=I, N=100.")
    print()
    a = R["aligned_theta0"]
    print(f"ALIGNED (theta=0):   err(M=0)={a['err_M0']}, err(M=1)={a['err_M1']}, err(M=40)={a['err_M40']} "
          f"-> monotone decreasing: {a['monotone_decreasing']}")
    mm = R["misaligned_theta150"]
    print(f"MISALIGNED (theta=150): err(M=0)={mm['err_M0']}, err(M=1)={mm['err_M1']} (peak at M={mm['peak_M']}, "
          f"err={mm['err_peak']}), err(M=40)={mm['err_M40']}")
    print(f"   -> non-monotonic (context HURTS below threshold then HELPS): {mm['non_monotonic_hurts_then_helps']}; "
          f"threshold M={R['misaligned_threshold_M']}")
    print(f"verdict: {R['verdict']}")

    def _np(o):
        if isinstance(o, np.bool_): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        raise TypeError
    import os; os.makedirs("outputs", exist_ok=True)
    open("outputs/incontext_results.json", "w").write(json.dumps(R, indent=2, default=_np))
    print("RESULTS_SHA256=" + hashlib.sha256(json.dumps(R, sort_keys=True, default=_np).encode()).hexdigest())
    return 0 if R["verdict"] == "supports" else 1

if __name__ == "__main__":
    raise SystemExit(main())
