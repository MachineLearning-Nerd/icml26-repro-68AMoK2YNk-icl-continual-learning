"""A marimo notebook — In-context continual learning: why the error peaks.

Run locally:  marimo edit reports/iccl-repro/iccl_demo.py
              marimo run  reports/iccl-repro/iccl_demo.py
"""
import marimo

__generated_with__ = "0.8.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "repro" / "src"))
    import theory_core as tc
    mo.md("# In-context continual learning: where does the error peak, and why?")
    return mo, np, tc


@app.cell
def _(mo):
    mo.md(
        "The paper's Theorem 4.3 gives the prediction error for task `t` in a masked "
        "linear-attention model trained on in-context linear regression. We re-derive it "
        "from the prediction rule `ŷ = x_qᵀ Γ⁻¹ β_t Σ_{s≤t} S_s` and check it against the "
        "paper's three-term (variance + bias) form, then against raw Monte-Carlo."
    )
    return


@app.cell
def _(np, tc):
    # Closed form vs raw Monte-Carlo over a random configuration.
    rng = np.random.default_rng(20242)
    d, T, M, t, N = 4, 4, 8, 2, 64
    A = rng.standard_normal((d, d)); Lam = A @ A.T / d + 0.5 * np.eye(d)
    W = [rng.standard_normal(d) * 1.5 for _ in range(T)]
    closed = tc.generalization_error_closed(W, Lam, M, t, N)
    paper = tc.generalization_error_paper(W, Lam, M, t, N)
    mc = tc.mc_generalization_error(W, Lam, M, t, N, n_trials=8000, seed=9900)
    print(f"derived closed form = {closed:.6f}")
    print(f"paper's 3-term form = {paper:.6f}   (identity error {abs(closed-paper):.2e})")
    print(f"raw Monte-Carlo     = {mc:.6f}   (6000-trial corroboration)")
    return


@app.cell
def _(mo):
    mo.md("### The non-monotone error curve\nVariance falls with M; the bias from misaligned history grows. The sum peaks at an intermediate M.")
    return


@app.cell
def _(np, tc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Ms = list(range(1, 31))
    Lam2 = np.eye(4)
    def curve(theta, N, T=3, t=3):
        target = np.zeros(4); target[0] = 1.0; W = []
        for s in range(T):
            if s == t - 1:
                W.append(target.copy())
            else:
                h = np.zeros(4); h[0] = np.cos(theta); h[1] = np.sin(theta); W.append(h)
        return [tc.generalization_error_closed(W, Lam2, m, t, N) for m in Ms]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(Ms, curve(0, 100), "o-", color="#1b7837", ms=4, label="aligned (θ=0°)")
    ax.plot(Ms, curve(np.deg2rad(140), 6), "s-", color="#b2182b", ms=4, label="misaligned (θ=140°, N=6)")
    ax.set_xlabel("in-context length M"); ax.set_ylabel("Theorem 4.3 error")
    ax.legend(frameon=False); ax.grid(alpha=0.3)
    fig.tight_layout(); fig
    return fig, ax


if __name__ == "__main__":
    app.run()
