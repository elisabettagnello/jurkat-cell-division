"""Cell size and growth along the cell cycle, using forward scatter (FSC-A).

FSC is a proxy of cell size but not proportional to volume. Assuming the
volume doubles between G1 and G2 and FSC ~ V^(alpha/3):

    alpha = 3 * log2( <FSC>_G2 / <FSC>_G1 )
    D     = FSC^(1/alpha)          (proportional to V^(1/3), a linear size)

Growth rates beta are the slopes of D versus time within each phase.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

PHASES = ['G1', 'S', 'G2']


def weighted_linear_fit(x, y, w):
    """Weighted least squares y = m x + q; R^2 against the flat model m = 0."""
    x, y, w = (np.asarray(v, dtype=float) for v in (x, y, w))
    sw, sx, sy = w.sum(), (w * x).sum(), (w * y).sum()
    sxx, sxy = (w * x * x).sum(), (w * x * y).sum()
    den = sw * sxx - sx ** 2
    m = (sw * sxy - sx * sy) / den
    q = (sxx * sy - sx * sxy) / den
    rss = (w * (y - (m * x + q)) ** 2).sum()
    tss = (w * (y - sy / sw) ** 2).sum()
    return m, q, (1 - rss / tss) if tss > 0 else np.nan


@dataclass
class Alpha:
    value: float
    error: float
    per_exp: pd.DataFrame


def growth_exponent(cells, exps, size='FSC-A'):
    """alpha from the G2/G1 mean-size ratio, averaged over `exps` weighted by cell number."""
    rows = []
    for exp in exps:
        d = cells[cells['Exp'] == exp]
        g1 = d.loc[d['phase'] == 'G1', size].mean()
        g2 = d.loc[d['phase'] == 'G2', size].mean()
        if np.isfinite(g1) and np.isfinite(g2):
            rows.append({'Exp': exp, 'alpha': 3 * np.log2(g2 / g1), 'n': len(d)})
    per_exp = pd.DataFrame(rows)
    a, w = per_exp['alpha'].to_numpy(), per_exp['n'].to_numpy(float)
    mean = np.average(a, weights=w)
    err = np.sqrt(np.average((a - mean) ** 2, weights=w)) / np.sqrt(len(a))
    return Alpha(mean, err, per_exp)


def phase_growth_rates(cells, exps, times_min, alpha, size='FSC-A'):
    """Mean linearised size D per phase and time point, and its slope beta.

    Each cell is weighted by the share of its generation in that sample, so
    every generation contributes in proportion to its abundance.
    Returns (points, rates): rates has beta in D units per hour.
    """
    pts = []
    for exp in exps:
        d = cells[cells['Exp'] == exp]
        if d.empty:
            continue
        share = d['generation'].map(d['generation'].value_counts(normalize=True))
        for ph in PHASES:
            sel = d['phase'] == ph
            if sel.sum() == 0:
                continue
            D = d.loc[sel, size].to_numpy(float) ** (1.0 / alpha)
            pts.append({'Exp': exp, 'phase': ph, 'minutes': times_min[exp],
                        'D': np.average(D, weights=share[sel]), 'n': int(sel.sum())})
    points = pd.DataFrame(pts)
    rates = []
    for ph, grp in points.groupby('phase'):
        if len(grp) >= 2:
            m, q, r2 = weighted_linear_fit(grp['minutes'], grp['D'], grp['n'])
            rates.append({'phase': ph, 'beta_per_h': 60 * m, 'D0': q, 'R2': r2})
    rates = pd.DataFrame(rates).set_index('phase').reindex(PHASES).reset_index()
    return points, rates


def generation_fractions(cells):
    """Fraction of cells in each generation at each time point (balanced-growth check)."""
    return pd.crosstab(cells['Exp'], cells['generation'], normalize='index')


def fit_added_size(size_birth, size_division, weights=None):
    """Size-control slope from Delta = S_div - S_birth = m * S_birth + q.

    m = -1: sizer, m = 0: adder, m > 0: timer-like. How birth and division
    sizes are paired is a methodological choice and is left to the caller.
    """
    s_b = np.asarray(size_birth, dtype=float)
    delta = np.asarray(size_division, dtype=float) - s_b
    w = np.ones_like(s_b) if weights is None else np.asarray(weights, dtype=float)
    return weighted_linear_fit(s_b, delta, w)
