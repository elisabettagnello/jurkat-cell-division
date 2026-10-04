"""Identify cell generations from the dilution of a proliferation dye.

At each division the dye (CellTrace Yellow) is split between the two daughters,
so in log2 scale every generation g sits about one unit to the left of the
previous one. The log2 intensity histogram is modelled as a Gaussian mixture

    P(y) = sum_g  pi_g * N(y | m0 - g * delta, sigma_g^2),     y = log2(intensity)

with the constraint that component means are equally spaced (spacing delta,
fitted or fixed). The constraint keeps peaks in the right order and lets the
fit separate generations that overlap. Parameters are estimated with a small
expectation-maximisation (EM) loop written for this constrained model.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

LOG2PI = np.log(2 * np.pi)


@dataclass
class GenerationFit:
    generations: np.ndarray      # generation index of each component
    weights: np.ndarray          # mixture weights pi_g
    means: np.ndarray            # component means in log2 units
    sigmas: np.ndarray           # component standard deviations in log2 units
    m0: float                    # (extrapolated) position of generation 0
    delta: float                 # spacing between generations in log2 units
    loglik: float
    n_events: int
    n_iter: int
    converged: bool
    extra: dict = field(default_factory=dict)

    def n_params(self, fixed_delta=False):
        k = len(self.generations)
        return (k - 1) + k + (1 if fixed_delta else 2)  # weights + sigmas + (m0, delta)

    def bic(self, fixed_delta=False):
        return -2 * self.loglik + self.n_params(fixed_delta) * np.log(self.n_events)

    def log_component_densities(self, y):
        y = np.asarray(y, dtype=float)[:, None]
        return (np.log(self.weights)[None, :]
                - 0.5 * LOG2PI - np.log(self.sigmas)[None, :]
                - 0.5 * ((y - self.means[None, :]) / self.sigmas[None, :]) ** 2)

    def responsibilities(self, y):
        """Posterior probability of each generation for every event (n x k)."""
        logp = self.log_component_densities(y)
        logp -= logp.max(axis=1, keepdims=True)
        p = np.exp(logp)
        return p / p.sum(axis=1, keepdims=True)

    def assign(self, y):
        """Most probable generation for every event."""
        return self.generations[np.argmax(self.responsibilities(y), axis=1)]

    def density(self, y):
        """Mixture density, for plotting against the histogram."""
        y = np.asarray(y, dtype=float)
        return sum(w * norm.pdf(y, m, s) for w, m, s in zip(self.weights, self.means, self.sigmas))


def fit_generations(y, generations, delta=None, init_m0=None, init_delta=1.0,
                    min_sigma=0.03, max_iter=500, tol=1e-7):
    """Fit the equally spaced Gaussian mixture to log2 intensities `y`.

    generations: generation indices expected in this sample, e.g. [0, 1, 2]
                 on day 1 or [3, 4, 5] on day 3.
    delta:       fix the spacing (e.g. 1.0) or None to fit it. With a single
                 component the spacing cannot be estimated and init_delta is used.
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    g = np.asarray(sorted(generations), dtype=float)
    k, n = len(g), len(y)
    if n < 10 * k:
        raise ValueError(f"too few events ({n}) for {k} components")

    d = float(delta) if delta is not None else float(init_delta)
    # Initialise so that the mixture's centre of mass matches the data.
    m0 = float(init_m0) if init_m0 is not None else float(np.mean(y) + np.mean(g) * d)
    weights = np.full(k, 1.0 / k)
    sigmas = np.full(k, max(0.25 * d, min_sigma))

    prev = -np.inf
    converged = False
    for it in range(1, max_iter + 1):
        means = m0 - g * d
        # E-step: log responsibilities, computed stably.
        logp = (np.log(weights)[None, :] - 0.5 * LOG2PI - np.log(sigmas)[None, :]
                - 0.5 * ((y[:, None] - means[None, :]) / sigmas[None, :]) ** 2)
        lmax = logp.max(axis=1, keepdims=True)
        lse = lmax[:, 0] + np.log(np.exp(logp - lmax).sum(axis=1))
        loglik = float(lse.sum())
        r = np.exp(logp - lse[:, None])

        # M-step.
        nk = r.sum(axis=0) + 1e-12
        weights = nk / n
        w = r / sigmas[None, :] ** 2  # precision-weighted responsibilities
        A, B, C = w.sum(), (w * g).sum(), (w * g ** 2).sum()
        Sy, Sgy = (w * y[:, None]).sum(), (w * g * y[:, None]).sum()
        if delta is None and k > 1:
            # Weighted least squares for y ~ m0 - g * delta.
            det = A * C - B ** 2
            m0 = (C * Sy - B * Sgy) / det
            d = (B * Sy - A * Sgy) / det
        else:
            m0 = (Sy + d * B) / A
        means = m0 - g * d
        sigmas = np.sqrt((r * (y[:, None] - means[None, :]) ** 2).sum(axis=0) / nk)
        sigmas = np.maximum(sigmas, min_sigma)

        if abs(loglik - prev) < tol * abs(loglik):
            converged = True
            break
        prev = loglik

    return GenerationFit(generations=g.astype(int), weights=weights, means=m0 - g * d,
                         sigmas=sigmas, m0=float(m0), delta=float(d), loglik=loglik,
                         n_events=n, n_iter=it, converged=converged)


def fit_generations_multistart(y, generations, init_m0s=None, **kwargs):
    """Run the EM from several starting positions and keep the best likelihood.

    By default the brightest component starts at the 95th, 97.5th and 99th
    percentiles of the data, plus the centre-of-mass guess.
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    g0 = min(generations)
    if init_m0s is None:
        d = kwargs.get('delta') or kwargs.get('init_delta', 1.0)
        init_m0s = [None] + [np.percentile(y, q) + g0 * d for q in (95, 97.5, 99)]
    fits = []
    for m in init_m0s:
        try:
            fits.append(fit_generations(y, generations, init_m0=m, **kwargs))
        except (ValueError, FloatingPointError):
            continue
    fits = [f for f in fits if np.isfinite(f.loglik)]
    if not fits:
        raise RuntimeError(f"no EM start converged for generations {list(generations)}")
    return max(fits, key=lambda f: f.loglik)


def _is_degenerate(fit, min_weight):
    return fit.weights.min() < min_weight


def select_generations(y, candidates, **kwargs):
    """Fit each candidate set of generations and return the fit with lowest BIC."""
    fits = [fit_generations_multistart(y, c, **kwargs) for c in candidates]
    fixed = kwargs.get('delta') is not None

    def score(f):
        b = f.bic(fixed_delta=fixed or len(f.generations) == 1)
        return b if np.isfinite(b) else np.inf
    return min(fits, key=score)


def relabel(fit, offset):
    """Shift generation labels by `offset` (the mixture itself is unchanged)."""
    return GenerationFit(generations=fit.generations + offset, weights=fit.weights, means=fit.means,
                         sigmas=fit.sigmas, m0=fit.m0 + offset * fit.delta, delta=fit.delta,
                         loglik=fit.loglik, n_events=fit.n_events, n_iter=fit.n_iter,
                         converged=fit.converged)


def fit_tracked(y, m0_ref, max_components=4, max_generation=8, **kwargs):
    """Fit a sample whose generation labels are not known in advance.

    A single sample only fixes the relative position of its peaks: shifting
    every label by one gives the same likelihood. The number of peaks is chosen
    by BIC, then labels are anchored so that the extrapolated generation-0
    position m0 is closest to `m0_ref` (e.g. the previous time point).
    """
    min_weight = kwargs.pop('min_weight', 0.01)
    fits = [fit_generations_multistart(y, list(range(w)), **kwargs) for w in range(1, max_components + 1)]
    # A component holding < min_weight of the cells is an artefact (outliers), not a generation.
    fits = [f for f in fits if not _is_degenerate(f, min_weight)] or fits[:1]
    fixed = kwargs.get('delta') is not None
    best = min(fits, key=lambda f: f.bic(fixed_delta=fixed or len(f.generations) == 1))
    w = len(best.generations)
    offsets = range(0, max_generation - w + 2)
    s = min(offsets, key=lambda o: abs(best.m0 + o * best.delta - m0_ref))
    return relabel(best, s)


def generation_moments(fit, y, intensity=None):
    """Per-generation summary, weighting each event by its responsibility.

    Moments are computed directly from the events, both in log2 units and in
    linear intensity units, so no log-normal conversion formula is needed.
    """
    y = np.asarray(y, dtype=float)
    x = 2.0 ** y if intensity is None else np.asarray(intensity, dtype=float)
    r = fit.responsibilities(y)
    rows = []
    for j, gen in enumerate(fit.generations):
        w = r[:, j]
        nw = w.sum()
        mean_lin = np.sum(w * x) / nw
        var_lin = np.sum(w * (x - mean_lin) ** 2) / nw
        rows.append({'generation': int(gen), 'weight': fit.weights[j], 'count': nw,
                     'mu_log2': fit.means[j], 'var_log2': fit.sigmas[j] ** 2,
                     'mean_lin': mean_lin, 'var_lin': var_lin})
    return rows
