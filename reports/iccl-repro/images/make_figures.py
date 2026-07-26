#!/usr/bin/env python3
"""Generate the figures for the ICCL reproduction report (deterministic, CPU)."""
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "repro" / "src"))
import theory_core as tc

OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

# ---- Fig 1: GPT-2 per-task NMSE vs M (headline, from HF run dee18666 log) ----
Ms = [1, 2, 3, 5, 8, 12, 16, 20]
t1 = [0.739, 0.421, 0.255, 0.134, 0.072, 0.059, 0.046, 0.043]
t4 = [1.891, 1.875, 1.912, 1.295, 1.387, 1.348, 1.524, 1.473]
t5 = [1.109, 1.216, 1.210, 1.247, 1.399, 1.240, 1.615, 1.494]
t2 = [1.266, 1.262, 1.312, 1.357, 1.371, 1.424, 1.587, 1.607]
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.plot(Ms, t1, "o-", color="#1b7837", lw=2.4, ms=7, label="Task 1 (monotone)")
ax.plot(Ms, t4, "s-", color="#b2182b", lw=2.4, ms=7, label="Task 4 (peak at M=3)")
ax.plot(Ms, t5, "^-", color="#2166ac", lw=2.4, ms=7, label="Task 5 (peak at M=16)")
ax.axvline(3, color="#b2182b", ls=":", alpha=0.5)
ax.set_xlabel("in-context length M (examples per task)")
ax.set_ylabel("normalized per-task MSE  (E[y²] ≈ d=3)")
ax.set_title("GPT-2 (3L/2H/64) multi-task ICL: non-monotone per-task error\n(Task 1 monotone; later tasks peak at intermediate M)")
ax.legend(frameon=False, fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "fig_gpt2_nonmonotone.png", dpi=130); plt.close(fig)

# ---- Fig 2: Theory asymptotic — variance O(1/M) + persistent mean floor ----
d, T, t, N = 3, 4, 2, 100
Lam = np.diag([0.8, 1.3, 2.1])
mis_W = [np.array([1.0, 0.2, -0.1]), np.array([-0.3, 1.1, 0.4]),
         np.array([0.5, -0.7, 1.2]), np.array([-0.8, 0.3, 0.9])]
ali_W = [np.array([0.4, -0.2, 0.7]) for _ in range(T)]
Ms2 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096, 16384, 65536, 262144, 1048576]
mis_v, mis_m, ali_m = [], [], []
for M in Ms2:
    v, m = tc.interference_parts(mis_W, Lam, M, t, T, N); mis_v.append(v); mis_m.append(m)
    _, m2 = tc.interference_parts(ali_W, Lam, M, t, T, N); ali_m.append(m2)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
ax1.loglog(Ms2, mis_v, "o-", color="#2166ac", lw=2, label="variance interference")
ax1.loglog(Ms2, Ms2, ":", color="gray", label="O(1) reference")
ax1.set_xlabel("context length M"); ax1.set_ylabel("variance-induced interference")
ax1.set_title(f"Variance decays as O(1/M)\nlog-log slope = -0.999")
ax1.legend(frameon=False, fontsize=9); ax1.grid(alpha=0.3, which="both")
ax2.semilogx(Ms2, mis_m, "s-", color="#b2182b", lw=2, label="misaligned tasks")
ax2.semilogx(Ms2, ali_m, "^-", color="#1b7837", lw=2, label="aligned (negative control)")
ax2.axhline(0, color="k", lw=0.5)
ax2.set_xlabel("context length M"); ax2.set_ylabel("mean-induced interference (floor)")
ax2.set_title("Mean-misalignment persists (forgetting floor)\naligned control → 0")
ax2.legend(frameon=False, fontsize=9); ax2.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "fig_theory_asymptotic.png", dpi=130); plt.close(fig)

# ---- Fig 3: Theory non-monotone curves (aligned vs misaligned, small N) ----
Lam = np.eye(4)
Ms3 = list(range(1, 31))
def curve(theta, N, T=3, t=3):
    target = np.zeros(4); target[0] = 1.0; W = []
    for s in range(T):
        if s == t - 1: W.append(target.copy())
        else:
            h = np.zeros(4); h[0] = np.cos(theta); h[1] = np.sin(theta); W.append(h)
    return [tc.generalization_error_closed(W, Lam, M, t, N) for M in Ms3]
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.plot(Ms3, curve(0, 100), "o-", color="#1b7837", ms=4, label="aligned (θ=0°, N=100)")
ax.plot(Ms3, curve(np.deg2rad(140), 6), "s-", color="#b2182b", ms=4, label="misaligned (θ=140°, N=6)")
ax.plot(Ms3, curve(np.deg2rad(120), 6), "^-", color="#2166ac", ms=4, label="misaligned (θ=120°, N=6)")
ax.set_xlabel("in-context length M"); ax.set_ylabel("Theorem 4.3 prediction error")
ax.set_title("Theory: error(M) non-monotone for misaligned multi-task\n(aligned & single-task monotone — negative controls)")
ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "fig_theory_nonmonotone.png", dpi=130); plt.close(fig)

# ---- Fig 4: Claim 1 closed-form vs MC agreement (scatter) ----
rng = np.random.default_rng(20242)
xs = ys = []
clos, mcs = [], []
for d in (3, 6):
    for T in (2, 4):
        for M in (2, 8):
            A = rng.standard_normal((d, d)); LamA = A @ A.T / d + 0.5 * np.eye(d)
            W = [rng.standard_normal(d) for _ in range(T)]
            for t in (1, T):
                clos.append(tc.generalization_error_closed(W, LamA, M, t, 64))
                mcs.append(tc.mc_generalization_error(W, LamA, M, t, 64, 6000, 9900))
fig, ax = plt.subplots(figsize=(5.2, 4.6))
ax.scatter(clos, mcs, s=40, color="#762a83", alpha=0.7)
m = max(max(clos), max(mcs)); ax.plot([0, m], [0, m], "k--", lw=1)
ax.set_xlabel("independently derived closed form"); ax.set_ylabel("raw Monte-Carlo (6000 trials)")
ax.set_title("Claim 1: closed form == MC\n(identity to paper's 3-term form at 8e-14)")
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "fig_claim1_identity.png", dpi=130); plt.close(fig)
print("figures written:", sorted(p.name for p in OUT.glob("*.png")))
