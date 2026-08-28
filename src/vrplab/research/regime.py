"""Gaussian hidden Markov model: estimation, selection, and honest filtering.

What was wrong with the version this replaces
---------------------------------------------
The Quant Guild "Markov chain regime switching bot" labels each bar by which
tercile of the last 60 bars its high-low range falls into, counts the
transitions between those labels to get a transition matrix, then at run time
runs a Bayes filter that produces a *different* label sequence from the one the
matrix was estimated on.  Four things follow:

  * the state is not hidden -- it is a deterministic function of the
    observation -- so the filter can and does contradict its own definition;
  * the transition matrix describes the percentile-label process, not the
    process the filter generates, so its persistence is double counted;
  * with 60 bars there are 59 transitions spread over 9 cells, and the
    arbitrary Laplace pseudo-count dominates the off-diagonals;
  * nothing is ever tested: not the Markov property, not the number of states,
    not whether the estimated stickiness differs from what independent draws
    would produce by chance.

This module fixes all four.  States are latent and estimated by EM
(Baum-Welch); the number of states is chosen by BIC or by out-of-sample
log-likelihood; the Markov property and time-homogeneity are testable; and the
only quantity exposed for trading is the **filtered** probability, which uses
information up to and including t and nothing after it.

The distinction between filtering and smoothing is the whole ballgame.  A
smoothed or Viterbi state sequence looks beautiful on a chart and is pure
look-ahead: it tells you what regime you were in using data from after the
fact.  ``GaussianHMM.filter()`` is what you may trade; ``smooth()`` and
``viterbi()`` are for description only and are labelled as such.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = ["GaussianHMM", "select_n_states", "markov_order_test", "homogeneity_test",
           "transition_matrix_from_labels", "stationary_distribution", "expected_durations"]

_LOG_ZERO = -1e300


# --------------------------------------------------------------------------- #
#  Model
# --------------------------------------------------------------------------- #
@dataclass
class GaussianHMM:
    """Univariate Gaussian-emission HMM fitted by Baum-Welch.

    Parameters
    ----------
    n_states:
        Number of latent regimes.  Choose it with :func:`select_n_states`, not
        by assertion.
    n_iter, tol:
        EM stopping rules.
    n_init:
        Random restarts.  The Baum-Welch likelihood surface is multimodal; a
        single initialisation is a coin flip.
    min_var:
        Variance floor.  Without it EM will happily collapse a state onto a
        single observation and send the likelihood to infinity.
    """

    n_states: int = 2
    n_iter: int = 300
    tol: float = 1e-6
    n_init: int = 8
    min_var: float = 1e-10
    random_state: int = 0
    order_by: str = "var"
    """Canonical ordering of the fitted states, so state 0 is always the calmest
    and results are comparable across refits.

    ``"var"`` (default) orders by emission variance, which is the right rule
    for volatility regimes -- where the states differ in dispersion, not level,
    and ordering by mean would shuffle them arbitrarily.  ``"mean"`` orders by
    emission mean, for observables such as returns where the level is what
    separates the states."""

    # fitted parameters
    startprob_: np.ndarray | None = None
    transmat_: np.ndarray | None = None
    means_: np.ndarray | None = None
    vars_: np.ndarray | None = None
    loglik_: float = -np.inf
    n_obs_: int = 0
    converged_: bool = False

    # ---------------- core recursions ---------------- #
    def _emission_logprob(self, x: np.ndarray) -> np.ndarray:
        """(T, K) matrix of log N(x_t | mu_k, var_k)."""
        x = np.asarray(x, float).ravel()[:, None]
        var = self.vars_[None, :]
        return -0.5 * (np.log(2.0 * np.pi * var) + (x - self.means_[None, :]) ** 2 / var)

    @staticmethod
    def _forward(log_b: np.ndarray, startprob: np.ndarray, transmat: np.ndarray):
        """Scaled forward pass. Returns (alpha, scale, loglik).

        Scaling rather than log-sum-exp: it is faster, and ``alpha`` comes out
        already normalised, which is precisely the filtered probability we want
        to expose.
        """
        T, K = log_b.shape
        b = np.exp(log_b - log_b.max(axis=1, keepdims=True))
        offset = log_b.max(axis=1)
        alpha = np.zeros((T, K))
        scale = np.zeros(T)

        a = startprob * b[0]
        scale[0] = a.sum()
        if scale[0] <= 0:
            scale[0] = 1e-300
        alpha[0] = a / scale[0]
        for t in range(1, T):
            a = (alpha[t - 1] @ transmat) * b[t]
            s = a.sum()
            if s <= 0:
                s = 1e-300
            scale[t] = s
            alpha[t] = a / s
        loglik = float(np.sum(np.log(scale)) + np.sum(offset))
        return alpha, scale, loglik

    @staticmethod
    def _backward(log_b: np.ndarray, transmat: np.ndarray, scale: np.ndarray):
        T, K = log_b.shape
        b = np.exp(log_b - log_b.max(axis=1, keepdims=True))
        beta = np.zeros((T, K))
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = (transmat @ (b[t + 1] * beta[t + 1])) / scale[t + 1]
        return beta

    # ---------------- fitting ---------------- #
    def fit(self, x) -> "GaussianHMM":
        x = np.asarray(x, dtype=float).ravel()
        x = x[np.isfinite(x)]
        T = x.size
        if T < 10 * self.n_states:
            raise ValueError(
                f"need at least {10 * self.n_states} observations for "
                f"{self.n_states} states, got {T}"
            )
        self.n_obs_ = T
        rng = np.random.default_rng(self.random_state)
        best = None

        for init in range(self.n_init):
            # Initialise means on quantiles, jittered; this is a starting point
            # only -- unlike the original, the final labelling is not tied to it.
            qs = np.linspace(0.5 / self.n_states, 1 - 0.5 / self.n_states, self.n_states)
            means = np.quantile(x, qs) + rng.normal(0, x.std(ddof=1) * 0.1, self.n_states)
            means = np.sort(means)
            variances = np.full(self.n_states, max(x.var(ddof=1), self.min_var))
            transmat = np.full((self.n_states, self.n_states), 0.1 / max(self.n_states - 1, 1))
            np.fill_diagonal(transmat, 0.9)
            transmat = transmat / transmat.sum(axis=1, keepdims=True)
            startprob = np.full(self.n_states, 1.0 / self.n_states)

            self.means_, self.vars_, self.transmat_, self.startprob_ = (
                means, variances, transmat, startprob)

            prev_ll, converged = -np.inf, False
            for _ in range(self.n_iter):
                log_b = self._emission_logprob(x)
                alpha, scale, ll = self._forward(log_b, self.startprob_, self.transmat_)
                beta = self._backward(log_b, self.transmat_, scale)

                gamma = alpha * beta
                gsum = gamma.sum(axis=1, keepdims=True)
                gamma = gamma / np.where(gsum > 0, gsum, 1e-300)

                # Expected transition counts, vectorised.  The textbook form is
                #   xi_ij = sum_t alpha[t,i] * A_ij * b[t+1,j] * beta[t+1,j] / c[t+1]
                # and the sum over t is an outer product, so the whole thing is
                # one matrix multiply rather than a Python loop over T.
                b = np.exp(log_b - log_b.max(axis=1, keepdims=True))
                fwd_back = (b[1:] * beta[1:]) / scale[1:, None]
                xi = self.transmat_ * (alpha[:-1].T @ fwd_back)
                xi_sum = xi.sum()
                if xi_sum > 0:
                    xi /= xi_sum

                # M step
                self.startprob_ = gamma[0] / gamma[0].sum()
                row = xi.sum(axis=1, keepdims=True)
                self.transmat_ = np.where(row > 0, xi / np.where(row > 0, row, 1), 1.0 / self.n_states)
                w = gamma.sum(axis=0)
                w = np.where(w > 1e-12, w, 1e-12)
                self.means_ = (gamma * x[:, None]).sum(axis=0) / w
                self.vars_ = np.maximum(
                    (gamma * (x[:, None] - self.means_[None, :]) ** 2).sum(axis=0) / w,
                    self.min_var,
                )

                if abs(ll - prev_ll) < self.tol * max(abs(prev_ll), 1.0):
                    converged = True
                    prev_ll = ll
                    break
                prev_ll = ll

            if best is None or prev_ll > best[0]:
                best = (prev_ll, self.startprob_.copy(), self.transmat_.copy(),
                        self.means_.copy(), self.vars_.copy(), converged)

        ll, sp, tm, mu, var, conv = best
        if self.order_by == "var":
            order = np.argsort(var)
        elif self.order_by == "mean":
            order = np.argsort(mu)
        else:
            raise ValueError(f"order_by must be 'var' or 'mean', got {self.order_by!r}")
        self.startprob_ = sp[order]
        self.transmat_ = tm[np.ix_(order, order)]
        self.means_ = mu[order]
        self.vars_ = var[order]
        self.loglik_ = float(ll)
        self.converged_ = bool(conv)
        return self

    # ---------------- inference ---------------- #
    def filter(self, x) -> np.ndarray:
        """**Tradeable.** ``P(state_t = k | x_1..x_t)`` for every t.

        Uses no information after t.  This is the only state estimate that may
        be fed to a trading rule.
        """
        self._check_fitted()
        log_b = self._emission_logprob(np.asarray(x, float).ravel())
        alpha, _, _ = self._forward(log_b, self.startprob_, self.transmat_)
        return alpha

    def smooth(self, x) -> np.ndarray:
        """**Descriptive only.** ``P(state_t = k | x_1..x_T)``.

        Conditions on the whole sample including the future.  Excellent for a
        chart or a paper, look-ahead if traded.
        """
        self._check_fitted()
        log_b = self._emission_logprob(np.asarray(x, float).ravel())
        alpha, scale, _ = self._forward(log_b, self.startprob_, self.transmat_)
        beta = self._backward(log_b, self.transmat_, scale)
        g = alpha * beta
        return g / g.sum(axis=1, keepdims=True)

    def viterbi(self, x) -> np.ndarray:
        """**Descriptive only.** Most likely whole state path. Look-ahead."""
        self._check_fitted()
        log_b = self._emission_logprob(np.asarray(x, float).ravel())
        T, K = log_b.shape
        with np.errstate(divide="ignore"):
            log_A = np.log(np.where(self.transmat_ > 0, self.transmat_, np.exp(_LOG_ZERO)))
            log_pi = np.log(np.where(self.startprob_ > 0, self.startprob_, np.exp(_LOG_ZERO)))
        delta = np.zeros((T, K))
        psi = np.zeros((T, K), dtype=int)
        delta[0] = log_pi + log_b[0]
        for t in range(1, T):
            m = delta[t - 1][:, None] + log_A
            psi[t] = np.argmax(m, axis=0)
            delta[t] = m[psi[t], np.arange(K)] + log_b[t]
        path = np.zeros(T, dtype=int)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    def predict_proba_ahead(self, filtered_row: np.ndarray, steps: int = 1) -> np.ndarray:
        """``P(state_{t+h} | info up to t)`` -- the actual forecast.

        The original build never does this: it uses the transition matrix only
        as a smoothing prior on the *current* label, which is why "regime
        switching" in its title is not what the code does.
        """
        self._check_fitted()
        p = np.asarray(filtered_row, float).ravel()
        for _ in range(int(steps)):
            p = p @ self.transmat_
        return p

    def score(self, x) -> float:
        self._check_fitted()
        log_b = self._emission_logprob(np.asarray(x, float).ravel())
        _, _, ll = self._forward(log_b, self.startprob_, self.transmat_)
        return ll

    # ---------------- diagnostics ---------------- #
    @property
    def n_params(self) -> int:
        k = self.n_states
        return (k - 1) + k * (k - 1) + 2 * k   # start + transitions + (mu, var)

    def bic(self, x=None) -> float:
        ll = self.loglik_ if x is None else self.score(x)
        n = self.n_obs_ if x is None else len(np.asarray(x).ravel())
        return float(-2.0 * ll + self.n_params * np.log(n))

    def aic(self, x=None) -> float:
        ll = self.loglik_ if x is None else self.score(x)
        return float(-2.0 * ll + 2.0 * self.n_params)

    def stationary(self) -> np.ndarray:
        self._check_fitted()
        return stationary_distribution(self.transmat_)

    def expected_durations(self) -> np.ndarray:
        self._check_fitted()
        return expected_durations(self.transmat_)

    def _check_fitted(self):
        if self.transmat_ is None:
            raise RuntimeError("model is not fitted; call fit() first")

    def summary(self) -> str:
        self._check_fitted()
        lines = [
            f"GaussianHMM(n_states={self.n_states})  loglik={self.loglik_:.2f}  "
            f"BIC={self.bic():.2f}  converged={self.converged_}",
            "state       mean        sd    stationary   exp.duration",
        ]
        pi, dur = self.stationary(), self.expected_durations()
        for k in range(self.n_states):
            lines.append(
                f"{k:>5} {self.means_[k]: 11.6f} {np.sqrt(self.vars_[k]): 9.6f} "
                f"{pi[k]: 12.4f} {dur[k]: 13.2f}"
            )
        lines.append("transition matrix:")
        for k in range(self.n_states):
            lines.append("  " + "  ".join(f"{v:.4f}" for v in self.transmat_[k]))
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Selection and tests
# --------------------------------------------------------------------------- #
def select_n_states(x, candidates=(1, 2, 3, 4), holdout: float = 0.0, **kw) -> dict:
    """Pick the number of regimes by BIC, and optionally by held-out
    log-likelihood.

    ``candidates`` may include 1: a one-state model is the "there are no
    regimes" null, and it wins more often than regime enthusiasts expect.  If
    it wins, the honest conclusion is that the regime overlay adds nothing.
    """
    x = np.asarray(x, float).ravel()
    x = x[np.isfinite(x)]
    if holdout > 0:
        cut = int(len(x) * (1 - holdout))
        x_tr, x_te = x[:cut], x[cut:]
    else:
        x_tr, x_te = x, None

    rows = []
    for k in candidates:
        try:
            if k == 1:
                mu, var = x_tr.mean(), max(x_tr.var(ddof=1), 1e-12)
                ll = float(np.sum(stats.norm.logpdf(x_tr, mu, np.sqrt(var))))
                n_par, bic = 2, -2 * ll + 2 * np.log(len(x_tr))
                ll_te = (float(np.sum(stats.norm.logpdf(x_te, mu, np.sqrt(var))))
                         if x_te is not None else np.nan)
                rows.append({"n_states": 1, "loglik": ll, "n_params": n_par,
                             "bic": bic, "aic": -2 * ll + 2 * n_par,
                             "holdout_loglik": ll_te, "converged": True})
                continue
            m = GaussianHMM(n_states=k, **kw).fit(x_tr)
            rows.append({
                "n_states": k, "loglik": m.loglik_, "n_params": m.n_params,
                "bic": m.bic(), "aic": m.aic(),
                "holdout_loglik": m.score(x_te) if x_te is not None else np.nan,
                "converged": m.converged_,
            })
        except ValueError:
            continue
    import pandas as pd
    table = pd.DataFrame(rows).set_index("n_states")
    best_bic = int(table["bic"].idxmin())
    best_oos = (int(table["holdout_loglik"].idxmax())
                if x_te is not None and table["holdout_loglik"].notna().any() else None)
    return {"table": table, "best_bic": best_bic, "best_holdout": best_oos}


def transition_matrix_from_labels(labels, n_states: int | None = None,
                                  alpha: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Count matrix and row-normalised transition matrix from an observed label
    sequence.  ``alpha`` is a Dirichlet pseudo-count; the default of 0 means
    "report what you saw".  Add smoothing deliberately and say so."""
    labels = np.asarray(labels, dtype=int).ravel()
    k = int(n_states or labels.max() + 1)
    counts = np.zeros((k, k))
    np.add.at(counts, (labels[:-1], labels[1:]), 1.0)
    smoothed = counts + alpha
    rows = smoothed.sum(axis=1, keepdims=True)
    P = np.divide(smoothed, rows, out=np.full_like(smoothed, np.nan), where=rows > 0)
    return counts, P


def markov_order_test(labels, n_states: int | None = None) -> dict:
    """Likelihood-ratio test of first-order Markov dependence against
    independence (order zero).

    H0: ``P(s_t | s_{t-1}) = P(s_t)`` -- the labels are serially independent and
    any apparent "stickiness" is an artefact.
    H1: first-order Markov.

    ``2 * (ll_1 - ll_0) ~ chi2 with (k-1)^2 df`` under H0.  If you cannot reject
    H0 you do not have a regime process, you have autocorrelated noise, and the
    entire transition matrix is decoration.  The original build never runs this.
    """
    labels = np.asarray(labels, dtype=int).ravel()
    k = int(n_states or labels.max() + 1)
    counts, _ = transition_matrix_from_labels(labels, k)
    n_trans = counts.sum()
    if n_trans < 1:
        return {"lr_stat": np.nan, "df": np.nan, "pvalue": np.nan, "n_transitions": 0}

    with np.errstate(divide="ignore", invalid="ignore"):
        row = counts.sum(axis=1, keepdims=True)
        p1 = np.where(row > 0, counts / np.where(row > 0, row, 1), 0.0)
        ll1 = np.nansum(np.where(counts > 0, counts * np.log(np.where(p1 > 0, p1, 1)), 0.0))
        p0 = counts.sum(axis=0) / n_trans
        ll0 = np.nansum(np.where(counts > 0,
                                 counts * np.log(np.where(p0 > 0, p0, 1))[None, :], 0.0))
    lr = float(2.0 * (ll1 - ll0))
    df = (k - 1) ** 2
    return {
        "lr_stat": lr, "df": df, "pvalue": float(stats.chi2.sf(lr, df)),
        "n_transitions": int(n_trans),
        "counts": counts,
        "min_cell": float(counts.min()),
        "warning": ("cells below 5 observations; chi2 approximation unreliable, "
                    "use a permutation test") if counts.min() < 5 else "",
    }


def homogeneity_test(labels, n_splits: int = 2, n_states: int | None = None) -> dict:
    """Test whether the transition matrix is stable across sub-periods.

    A Markov model assumes time-homogeneity.  Financial regimes famously are not
    homogeneous, which is the deep reason a transition matrix estimated on the
    last five minutes tells you little about the next five.  Splits the sample,
    estimates a matrix in each block, and runs a likelihood-ratio test of a
    common matrix against block-specific ones.
    """
    labels = np.asarray(labels, dtype=int).ravel()
    k = int(n_states or labels.max() + 1)
    blocks = np.array_split(labels, n_splits)

    def _ll(counts):
        row = counts.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(row > 0, counts / np.where(row > 0, row, 1), 0.0)
            return float(np.nansum(np.where(counts > 0,
                                            counts * np.log(np.where(p > 0, p, 1)), 0.0)))

    pooled = np.zeros((k, k))
    ll_free = 0.0
    for b in blocks:
        c, _ = transition_matrix_from_labels(b, k)
        pooled += c
        ll_free += _ll(c)
    lr = float(2.0 * (ll_free - _ll(pooled)))
    df = (n_splits - 1) * k * (k - 1)
    return {"lr_stat": lr, "df": df, "pvalue": float(stats.chi2.sf(lr, df)),
            "n_splits": n_splits}


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Long-run distribution: the left eigenvector of P for eigenvalue 1."""
    P = np.asarray(P, float)
    vals, vecs = np.linalg.eig(P.T)
    i = int(np.argmin(np.abs(vals - 1.0)))
    v = np.real(vecs[:, i])
    v = np.abs(v)
    s = v.sum()
    return v / s if s > 0 else np.full(P.shape[0], 1.0 / P.shape[0])


def expected_durations(P: np.ndarray) -> np.ndarray:
    """Expected number of periods spent in each state, ``1 / (1 - P_ii)``.

    Sanity check any fitted model against this: a "regime" with an expected
    duration of 1.4 bars is not a regime.
    """
    d = np.diag(np.asarray(P, float))
    with np.errstate(divide="ignore"):
        return np.where(d < 1.0, 1.0 / (1.0 - d), np.inf)
