"""Core model for in-context continual learning (arXiv:2605.28705, Sec. 3-4).

Clean-room implementation of the masked-linear-self-attention prediction rules
and their closed-form error expressions. Everything here is derived from first
principles (the prediction rule + Gaussian moment identities) so it can serve as
an INDEPENDENT reference against the paper's stated formulas.

Notation (matches the paper):
  T tasks, M in-context examples per task, task index t (1-indexed).
  x_{s,i} ~ N(0, Lambda),  y_{s,i} = w_s^T x_{s,i}.
  mu_s   = E[x y]      = Lambda w_s
  Sigma_s= Var(x y)    = (w_s^T Lambda w_s) Lambda + mu_s mu_s^T   (Gaussian 4th moment)
  Gamma  = (1 + 1/N) Lambda + (tr(Lambda)/N) I
  S_s    = sum_{i=1}^M x_{s,i} y_{s,i}            (d-vector)

Prediction rules (Theorem 4.2 / Lemma `prediction`, masked self-attention):
  task-t query : yhat_{t,q} = x_q^T Gamma^{-1} ( beta_t * sum_{s<=t} S_s ),  beta_t = 1/(t(M+1))
  final query  : Yhat_{t,q} = x_q^T Gamma^{-1} ( d_hat  * sum_{s<=T} S_s ),  d_hat  = 1/(T(M+1)+1)
"""
from __future__ import annotations

import numpy as np


def gamma_matrix(Lam: np.ndarray, N: int) -> np.ndarray:
    d = Lam.shape[0]
    return (1.0 + 1.0 / N) * Lam + (np.trace(Lam) / N) * np.eye(d)


def task_moments(W: list[np.ndarray], Lam: np.ndarray):
    """Return (mu, Sigma) lists for each task, derived analytically from Gaussian moments."""
    mu = [Lam @ w for w in W]
    Sigma = []
    for w, m in zip(W, mu):
        wLw = float(w @ Lam @ w)
        Sigma.append(wLw * Lam + np.outer(m, m))
    return mu, Sigma


def moment_lemma_EZZ_blocks(mu, Sigma, M: int):
    """E[S_i S_j^T] blocks (Lemma in App. C): i=j -> M Sigma_i + M^2 mu_i mu_i^T ; i!=j -> M^2 mu_i mu_j^T."""
    T = len(mu)
    blocks = np.empty((T, T), dtype=object)
    for i in range(T):
        for j in range(T):
            if i == j:
                blocks[i, j] = M * Sigma[i] + (M * M) * np.outer(mu[i], mu[i])
            else:
                blocks[i, j] = (M * M) * np.outer(mu[i], mu[j])
    return blocks


# ---------------------------------------------------------------------------------------------
# Closed-form errors (INDEPENDENT derivation: rule -> E[.] via moment lemma). Basis-free.
# ---------------------------------------------------------------------------------------------

def generalization_error_closed(W, Lam, M, t, N):
    """E[(yhat_{t,q} - y_{t,q})^2].

    Derived: with Z = sum_{s<=t} S_s,  yhat-y = x_q^T(Gamma^{-1} beta_t Z - w_t);
    E_{x_q,S} = E_S[(Gamma^{-1} beta_t Z - w_t)^T Lambda (Gamma^{-1} beta_t Z - w_t)].
    """
    d = Lam.shape[0]
    G = gamma_matrix(Lam, N)
    Ginv = np.linalg.inv(G)
    GAG = Ginv @ Lam @ Ginv                      # Gamma^{-1} Lambda Gamma^{-1}  (= Gamma^{-2} Lambda, commutes)
    mu, Sigma = task_moments(W, Lam)
    beta_t = 1.0 / (t * (M + 1))
    mu_sum = sum(mu[:t])
    EZ = M * mu_sum                               # E[Z]
    EZZT = M * sum(Sigma[:t]) + (M * M) * np.outer(mu_sum, mu_sum)   # E[Z Z^T]
    wt = W[t - 1]
    val = (beta_t * beta_t) * np.trace(GAG @ EZZT)
    val -= 2.0 * beta_t * float(wt @ Lam @ Ginv @ EZ)
    val += float(wt @ Lam @ wt)                   # irreducible term (0 here since y=w_t^T x exactly)
    return val


def interference_error_closed(W, Lam, M, t, T, N):
    """E[(Yhat_{t,q} - yhat_{t,q})^2]  (Theorem 4.4 forgetting / interference).

    Yhat - yhat = x_q^T Gamma^{-1}( sum_{s<=t} c_t S_s + sum_{s>t} d S_s ), with
      c_t = -(((T-t)M) + (T+1-t)) / ( t (M+1) (T M + T +1) ),   d = 1/(T M + T +1).
    E[.] = tr(Gamma^{-1} Lambda Gamma^{-1} E[U U^T])  with  U = sum_s alpha_s S_s.
    """
    G = gamma_matrix(Lam, N)
    Ginv = np.linalg.inv(G)
    GAG = Ginv @ Lam @ Ginv
    mu, Sigma = task_moments(W, Lam)
    c_t, d = forgetting_coefficients(T, t, M)
    alpha = np.array([c_t if s < t else d for s in range(T)])
    EU = M * sum(alpha[s] * mu[s] for s in range(T))
    EUUT = M * sum(alpha[s] ** 2 * Sigma[s] for s in range(T)) + np.outer(EU, EU)
    return float(np.trace(GAG @ EUUT))


def forgetting_coefficients(T, t, M):
    c_t = -(((T - t) * M) + (T + 1 - t)) / (t * (M + 1) * (T * M + T + 1))
    d = 1.0 / (T * M + T + 1)
    return c_t, d


# ---------------------------------------------------------------------------------------------
# Paper-stated formulas (eigenbasis of Lambda), for the symbolic-identity cross-check.
# ---------------------------------------------------------------------------------------------

def generalization_error_paper(W, Lam, M, t, N):
    """The paper's 3-term decomposition (Theorem 4.3) evaluated numerically in Lambda's eigenbasis."""
    d = Lam.shape[0]
    lam, V = np.linalg.eigh(Lam)                  # Lambda = V diag(lam) V^T, columns v_i
    trL = float(np.trace(Lam))
    G = gamma_matrix(Lam, N)
    gamma = (1.0 + 1.0 / N) * lam + trL / N        # eigenvalues of Gamma (Gamma commutes with Lambda)
    mu, Sigma = task_moments(W, Lam)
    wt = W[t - 1]
    # rotate mu_t and sum mu into eigenbasis
    mu_t_eig = V.T @ mu[t - 1]
    mu_sum_eig = V.T @ sum(mu[:t])
    alpha = M / (t * (M + 1))
    # variance term: M/(t^2(M+1)^2) sum_s tr(Sigma_s Gamma^{-2} Lambda)
    var = 0.0
    for s in range(t):
        # tr(Sigma_s Gamma^{-2} Lambda) = sum_i (lam_i/gamma_i^2) (v_i^T Sigma_s v_i)
        Sigma_s_eig_diag = np.array([float(V[:, i] @ Sigma[s] @ V[:, i]) for i in range(d)])
        var += np.sum((lam / gamma ** 2) * Sigma_s_eig_diag)
    var *= M / (t * t * (M + 1) ** 2)
    # bias term (per eigen-coordinate i)
    bias = 0.0
    for i in range(d):
        s_i = mu_sum_eig[i]
        m_i = mu_t_eig[i]
        delta_i = (lam[i] + trL) / N
        num = lam[i] * (alpha * s_i - m_i) - m_i * delta_i
        den = lam[i] + delta_i
        bias += (1.0 / lam[i]) * (num / den) ** 2
    return var + bias                              # + irreducible (0)


def interference_error_paper(W, Lam, M, t, T, N):
    """Paper's Theorem 4.4 expanded form (intra-task variance + inter-task mean interaction)."""
    d = Lam.shape[0]
    lam, V = np.linalg.eigh(Lam)
    trL = float(np.trace(Lam))
    G = gamma_matrix(Lam, N)
    gamma = (1.0 + 1.0 / N) * lam + trL / N
    mu, Sigma = task_moments(W, Lam)
    c_t, d_coef = forgetting_coefficients(T, t, M)
    alpha = [c_t if s < t else d_coef for s in range(T)]
    # helper: tr(M Gamma^{-2} Lambda) = sum_i (lam_i/gamma_i^2) v_i^T M v_i
    def trG2L(Mmat):
        return sum((lam[i] / gamma[i] ** 2) * float(V[:, i] @ Mmat @ V[:, i]) for i in range(d))
    val = 0.0
    for i in range(T):
        val += alpha[i] ** 2 * trG2L(M * Sigma[i] + (M * M) * np.outer(mu[i], mu[i]))
    for i in range(T):
        for j in range(T):
            if i != j:
                val += (M * M) * alpha[i] * alpha[j] * trG2L(np.outer(mu[i], mu[j]))
    return val


def interference_parts(W, Lam, M, t, T, N):
    """Decompose E[(Yhat-yhat)^2] into (variance_part, mean_part).

    variance_part = M * sum_s alpha_s^2 tr(Sigma_s Gamma^{-2} Lambda)   -> O(1/M)
    mean_part     = M^2 * ||sum_s alpha_s mu_s||^2_{Gamma^{-2} Lambda}  -> O(1) (persistent floor)
    """
    G = gamma_matrix(Lam, N)
    Ginv = np.linalg.inv(G)
    GAG = Ginv @ Lam @ Ginv
    mu, Sigma = task_moments(W, Lam)
    c_t, d_coef = forgetting_coefficients(T, t, M)
    alpha = np.array([c_t if s < t else d_coef for s in range(T)])
    var = M * sum(alpha[s] ** 2 * float(np.trace(GAG @ Sigma[s])) for s in range(T))
    combo = sum(alpha[s] * mu[s] for s in range(T))
    mean = (M * M) * float(combo @ GAG @ combo)
    return var, mean


def limiting_mean_floor(W, Lam, t, T, N):
    """Analytic M->infty limit of the mean-interference floor: ||sum_s (lim M*alpha_s) mu_s||^2_GAG,
    with lim M*alpha_s = -(T-t)/(tT) for s<=t and 1/T for s>t."""
    G = gamma_matrix(Lam, N)
    Ginv = np.linalg.inv(G)
    GAG = Ginv @ Lam @ Ginv
    mu, _ = task_moments(W, Lam)
    lim = np.array([-(T - t) / (t * T) if s < t else 1.0 / T for s in range(T)])
    combo = sum(lim[s] * mu[s] for s in range(T))
    return float(combo @ GAG @ combo)


# ---------------------------------------------------------------------------------------------
# Raw Monte-Carlo sampler (independent of all closed forms): samples x,y and the prediction.
# Fully vectorized (batched) so broad sweeps finish in seconds.
# ----------------------------------------------------------------------------------------------

def _sample_block(rng, Wsub, Lam, M, n_trials):
    """Sample Z = sum_{s,i} x_{s,i} y_{s,i} for tasks in Wsub, vectorized. Returns (n_trials, d)."""
    t = len(Wsub); d = Lam.shape[0]
    Lchol = np.linalg.cholesky(Lam)
    Wt = np.array(Wsub)                                  # (t, d)
    X = rng.standard_normal((n_trials, t, M, d)) @ Lchol.T          # (n_trials, t, M, d)
    Y = np.einsum("nsmd,sd->nsm", X, Wt)                 # (n_trials, t, M)
    XY = X * Y[..., None]                                # (n_trials, t, M, d)
    return XY.sum(axis=(1, 2))                           # (n_trials, d)


def mc_generalization_error(W, Lam, M, t, N, n_trials, seed):
    """Direct MC of E[(yhat-y)^2] using the semi-analytic x_q integration (low variance):
       E_S[(Gamma^{-1} beta_t Z - w_t)^T Lambda (Gamma^{-1} beta_t Z - w_t)]."""
    rng = np.random.default_rng(seed)
    G = gamma_matrix(Lam, N); Ginv = np.linalg.inv(G)
    beta_t = 1.0 / (t * (M + 1)); wt = W[t - 1]
    Z = _sample_block(rng, W[:t], Lam, M, n_trials)      # (n_trials, d)
    A = (beta_t * Z) @ Ginv.T - wt                       # Ginv symmetric
    return float(np.mean(np.sum((A @ Lam) * A, axis=1)))


def mc_interference_error(W, Lam, M, t, T, N, n_trials, seed):
    """Direct MC of E[(Yhat-yhat)^2] = E_U[(Gamma^{-1} U)^T Lambda (Gamma^{-1} U)],
       U = sum_s alpha_s S_s, sampled from raw Gaussian draws."""
    rng = np.random.default_rng(seed)
    G = gamma_matrix(Lam, N); Ginv = np.linalg.inv(G)
    c_t, d_coef = forgetting_coefficients(T, t, M)
    alpha = np.array([c_t if s < t else d_coef for s in range(T)])
    # sample S_s per task, then weight by alpha_s
    Lchol = np.linalg.cholesky(Lam)
    Wt = np.array(W)                                     # (T, d)
    X = rng.standard_normal((n_trials, T, M, Lam.shape[0])) @ Lchol.T
    Y = np.einsum("nsmd,sd->nsm", X, Wt)
    XY = X * Y[..., None]                                # (n_trials, T, M, d)
    S = XY.sum(axis=2)                                   # (n_trials, T, d)
    U = np.einsum("nsd,s->nd", S, alpha)                # (n_trials, d)
    GU = U @ Ginv.T
    return float(np.mean(np.sum((GU @ Lam) * GU, axis=1)))


def mc_moment_lemma(W, Lam, M, n_trials, seed):
    """Empirical E[S_i S_j^T] from raw samples, to validate the moment lemma independently."""
    rng = np.random.default_rng(seed)
    T = len(W); d = Lam.shape[0]
    Lchol = np.linalg.cholesky(Lam)
    Wt = np.array(W)
    X = rng.standard_normal((n_trials, T, M, d)) @ Lchol.T
    Y = np.einsum("nsmd,sd->nsm", X, Wt)
    XY = X * Y[..., None]                                # (n_trials, T, M, d)
    S = XY.sum(axis=2)                                   # (n_trials, T, d)
    acc = np.zeros((T, T, d, d))
    for i in range(T):
        for j in range(T):
            acc[i, j] = S[:, i, :].T @ S[:, j, :] / n_trials
    return acc
