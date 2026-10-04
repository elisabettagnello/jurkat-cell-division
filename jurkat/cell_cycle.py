"""Cell-cycle phases from DNA content (DAPI), Watson-Pragmatic style.

The DAPI histogram is decomposed into a Gaussian G1 peak (2n DNA), a Gaussian
G2/M peak (4n, at about twice the G1 intensity) and an S-phase component in
between, modelled as a second-order polynomial broadened with the same
coefficient of variation (CV) as the G1 peak.
"""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

SQRT2PI = np.sqrt(2 * np.pi)


def gaussian(x, area, mu, sigma):
    return area / (SQRT2PI * sigma) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def broadened_poly(x, x_s, a, b, c, cv):
    """Polynomial a + b x + c x^2 defined on the S region `x_s`, where each
    point is smeared by a Gaussian of width cv * x (instrument resolution)."""
    amp = a + b * x_s + c * x_s ** 2
    sig = np.maximum(cv * x_s, 1e-9)
    kernel = np.exp(-0.5 * ((x[:, None] - x_s[None, :]) / sig[None, :]) ** 2) / (SQRT2PI * sig[None, :])
    return kernel @ amp


def _width_at_fraction(h, peak, frac):
    thr = frac * h[peak]
    left = peak
    while left > 0 and h[left] > thr:
        left -= 1
    right = peak
    while right < len(h) - 1 and h[right] > thr:
        right += 1
    return left, right


@dataclass
class CellCycleFit:
    x: np.ndarray            # bin centres
    counts: np.ndarray       # histogram used for the fit
    params: np.ndarray       # A1, mu1, s1, A2, mu2, s2, a, b, c
    errors: np.ndarray
    x_s: np.ndarray          # S-phase support

    @property
    def cv(self):
        return self.params[2] / self.params[1]

    def components(self, x=None):
        x = self.x if x is None else np.asarray(x, dtype=float)
        A1, mu1, s1, A2, mu2, s2, a, b, c = self.params
        g1 = gaussian(x, A1, mu1, s1)
        g2 = gaussian(x, A2, mu2, s2)
        s = np.clip(broadened_poly(x, self.x_s, a, b, c, s1 / mu1), 0, None)
        return g1, s, g2

    def fractions(self):
        g1, s, g2 = self.components()
        tot = g1.sum() + s.sum() + g2.sum()
        return {'G1': g1.sum() / tot, 'S': s.sum() / tot, 'G2': g2.sum() / tot}

    def phase_probabilities(self, values):
        """Probability of G1, S and G2 for each event, from the fitted densities."""
        values = np.asarray(values, dtype=float)
        A1, mu1, s1, A2, mu2, s2 = self.params[:6]
        g1 = gaussian(values, A1, mu1, s1)
        g2 = gaussian(values, A2, mu2, s2)
        s_curve = self.components()[1]
        s = interp1d(self.x, s_curve, bounds_error=False, fill_value=0.0)(values)
        tot = g1 + s + g2
        tot[tot == 0] = 1.0
        return np.column_stack([g1 / tot, s / tot, g2 / tot])


def fit_cell_cycle(dna, bin_width=225.0, x_max=120000.0, x_min_signal=30000.0,
                   g2_search_factor=1.95):
    """Fit the G1 / S / G2 decomposition to DNA-content values of one sample."""
    dna = np.asarray(dna, dtype=float)
    dna = dna[np.isfinite(dna)]
    edges = np.arange(0, x_max + bin_width, bin_width)
    h, _ = np.histogram(dna, bins=edges)
    h = h.astype(float)
    x = 0.5 * (edges[:-1] + edges[1:])
    h[x < x_min_signal] = 0  # debris / noise region

    # 1. Peak guesses: G1 is the global maximum, G2 the maximum beyond ~2x G1.
    i1 = int(np.argmax(h))
    mu1 = x[i1]
    left, _ = _width_at_fraction(h, i1, 0.4)
    s1 = max((mu1 - x[left]) / 2, bin_width)
    start = np.searchsorted(x, g2_search_factor * mu1)
    i2 = start + int(np.argmax(h[start:]))
    mu2 = x[i2]
    _, right = _width_at_fraction(h, i2, 0.6)
    s2 = max(x[right] - mu2, bin_width)

    # 2. Two Gaussians fitted on the outer flanks of the peaks.
    def two_gauss(xx, A1, m1, sd1, A2, m2, sd2):
        return gaussian(xx, A1, m1, sd1) + gaussian(xx, A2, m2, sd2)

    flank = ((x > x_min_signal) & (x < mu1 + 2 * s1)) | ((x > mu2 - 2 * s2) & (x < x_max))
    p0 = [h[i1] * 2.5 * s1, mu1, s1, h[i2] * 2.5 * s2, mu2, s2]
    pg, _ = curve_fit(two_gauss, x[flank], h[flank], p0=p0, maxfev=10000)

    # 3. S phase: residual between the peaks, fitted with the broadened polynomial.
    resid = np.clip(h - two_gauss(x, *pg), 0, None)
    in_s = (x > pg[1] + 2 * pg[2]) & (x < pg[4] - 2 * pg[5])
    x_s = x[in_s]
    if len(x_s) < 3:
        raise ValueError("S-phase region is empty: check the gating or the peak guesses")
    cv0 = pg[2] / pg[1]
    # Start from a flat S phase at the mean residual level (amplitude x bin width).
    p0_s = [max(resid[in_s].mean(), 1.0) * bin_width, 0.0, 0.0]
    pp, _ = curve_fit(lambda xx, a, b, c: broadened_poly(xx, x_s, a, b, c, cv0),
                      x_s, resid[in_s], p0=p0_s, maxfev=5000)

    # 4. Global fit of all nine parameters.
    def total(xx, A1, m1, sd1, A2, m2, sd2, a, b, c):
        return (gaussian(xx, A1, m1, sd1) + gaussian(xx, A2, m2, sd2)
                + np.clip(broadened_poly(xx, x_s, a, b, c, sd1 / m1), 0, None))

    p_all, cov = curve_fit(total, x, h, p0=list(pg) + list(pp), maxfev=20000)
    return CellCycleFit(x=x, counts=h, params=p_all, errors=np.sqrt(np.diag(cov)), x_s=x_s)
