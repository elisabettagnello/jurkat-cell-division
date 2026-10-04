"""Asymmetric binomial model of cytoplasm partitioning at division.

At each division the mother's content is split with fraction p to one daughter
and q = 1 - p to the other (p = 0.5: symmetric division). Starting from
generation 0 with mean mu0 and variance var0 (linear intensity units):

    mean:      mu_g      = mu0 * (1/2)^g
    variance:  sigma_g^2 = a (1/2)^(g-1) (1 - (2b)^g) / (1 - 2b)
                           + b^g (var0 + mu0^2) - (1/2)^(2g) mu0^2

with a = p q mu0 and b = (p^2 + q^2) / 2. Fitting the variance across
generations estimates p.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit


def mean_model(g, mu0):
    return mu0 * 0.5 ** np.asarray(g, dtype=float)


def variance_model(g, p, mu0, var0):
    g = np.asarray(g, dtype=float)
    a = p * (1.0 - p) * mu0
    b = (p ** 2 + (1.0 - p) ** 2) / 2.0
    if np.isclose(b, 0.5):
        term1 = a * 0.5 ** (g - 1.0) * g
    else:
        term1 = a * 0.5 ** (g - 1.0) * (1.0 - (2.0 * b) ** g) / (1.0 - 2.0 * b)
    term1 = np.where(g >= 1, term1, 0.0)
    var = term1 + b ** g * (var0 + mu0 ** 2) - 0.25 ** g * mu0 ** 2
    return np.maximum(var, 1e-12)


def log2_to_linear(mu_log2, var_log2, method='exact'):
    """Convert mean/variance of y = log2(x) into mean/variance of x.

    'exact' assumes y is Gaussian (x log-normal). 'original' reproduces the
    approximation mean = 2^mu, var = (2^var - 1) 2^(2 mu) used in the first
    version of this analysis; it is kept only for comparison.
    """
    mu = np.asarray(mu_log2, dtype=float)
    s2 = np.asarray(var_log2, dtype=float)
    if method == 'original':
        return 2.0 ** mu, (2.0 ** s2 - 1.0) * 2.0 ** (2 * mu)
    l2 = np.log(2.0) ** 2
    mean = 2.0 ** mu * np.exp(0.5 * s2 * l2)
    var = (np.exp(s2 * l2) - 1.0) * 2.0 ** (2 * mu) * np.exp(s2 * l2)
    return mean, var


def aggregate_snapshots(table, exclude_exps=(), min_count=0):
    """Average each generation's linear mean and variance over time points.

    table: one row per (Exp, generation) with columns mean_lin, var_lin, count.
    Returns one row per generation with the average over snapshots, its
    standard error (SEM) and the number of snapshots used.
    """
    t = table[~table['Exp'].isin(exclude_exps) & (table['count'] >= min_count)]
    out = t.groupby('generation').agg(
        mean=('mean_lin', 'mean'), mean_sem=('mean_lin', 'sem'),
        var=('var_lin', 'mean'), var_sem=('var_lin', 'sem'),
        n_snapshots=('mean_lin', 'size')).reset_index()
    return out


@dataclass
class FitResult:
    value: float
    error: float
    chi2: float
    dof: int
    p_value: float

    @property
    def chi2_red(self):
        return self.chi2 / self.dof if self.dof > 0 else np.nan

    def __str__(self):
        return (f"{self.value:.4f} ± {self.error:.4f}  "
                f"(chi2/dof = {self.chi2:.2f}/{self.dof}, p-value = {self.p_value:.3f})")


def _fit(func, x, y, sigma, p0, bounds=(-np.inf, np.inf)):
    popt, pcov = curve_fit(func, x, y, sigma=sigma, absolute_sigma=True, p0=p0, bounds=bounds)
    resid = (y - func(x, *popt)) / sigma
    chi2 = float(np.sum(resid ** 2))
    dof = len(x) - len(popt)
    pval = float(stats.chi2.sf(chi2, dof)) if dof > 0 else np.nan
    return FitResult(float(popt[0]), float(np.sqrt(pcov[0, 0])), chi2, dof, pval)


def usable(agg, column):
    """Generations with at least two snapshots and a positive SEM."""
    return agg[(agg['n_snapshots'] >= 2) & (agg[f'{column}_sem'] > 0) & (agg[column] > 0)]


def fit_partition_p(agg):
    """Estimate p from the variance across generations.

    mu0 and var0 are taken from generation 0, as in the model's definition.
    """
    gen0 = agg[agg['generation'] == 0].iloc[0]
    mu0, var0 = gen0['mean'], gen0['var']
    d = usable(agg, 'var')
    res = _fit(lambda g, p: variance_model(g, p, mu0, var0),
               d['generation'].to_numpy(float), d['var'].to_numpy(), d['var_sem'].to_numpy(),
               p0=[0.5], bounds=(1e-4, 1 - 1e-4))
    return res, mu0, var0


def fit_mu0(agg):
    """Fit mu0 in mean_g = mu0 / 2^g: checks that the dye halves at each division."""
    d = usable(agg, 'mean')
    return _fit(mean_model, d['generation'].to_numpy(float), d['mean'].to_numpy(),
                d['mean_sem'].to_numpy(), p0=[d['mean'].iloc[0] * 2 ** d['generation'].iloc[0]])


def per_snapshot_table(rows):
    return pd.DataFrame(rows)
