"""Sanity checks on synthetic inputs with known answers (run with: pytest)."""

import numpy as np

from jurkat import generations, partitioning


def test_constrained_gmm_recovers_peaks():
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.normal(15.0, 0.12, 3000), rng.normal(14.0, 0.14, 5000),
                        rng.normal(13.0, 0.16, 2000)])
    fit = generations.fit_generations_multistart(y, [0, 1, 2])
    assert abs(fit.delta - 1.0) < 0.02
    assert np.allclose(fit.means, [15.0, 14.0, 13.0], atol=0.02)
    assert np.allclose(fit.weights, [0.3, 0.5, 0.2], atol=0.02)


def test_tracking_assigns_absolute_labels():
    rng = np.random.default_rng(1)
    y = np.concatenate([rng.normal(13.0, 0.15, 4000), rng.normal(12.0, 0.15, 4000)])
    fit = generations.fit_tracked(y, m0_ref=15.0)
    assert list(fit.generations) == [2, 3]


def test_mean_halves_and_symmetric_variance():
    mu0, var0 = 30000.0, 2e7
    g = np.arange(5)
    assert np.allclose(partitioning.mean_model(g, mu0), mu0 / 2 ** g)
    # With p = 0.5, apart from the binomial (molecule-counting) term, the
    # coefficient of variation is conserved across generations.
    binomial = np.where(g >= 1, 0.25 * mu0 * 0.5 ** (g - 1) * (1 - 0.5 ** g) / 0.5, 0.0)
    var = partitioning.variance_model(g, 0.5, mu0, var0) - binomial
    assert np.allclose(np.sqrt(var) / partitioning.mean_model(g, mu0), np.sqrt(var0) / mu0, rtol=1e-6)


def test_log2_to_linear_exact_matches_sampling():
    rng = np.random.default_rng(2)
    y = rng.normal(14.0, 0.2, 400000)
    mean, var = partitioning.log2_to_linear(14.0, 0.04)
    assert abs(mean / np.mean(2 ** y) - 1) < 1e-3
    assert abs(var / np.var(2 ** y) - 1) < 2e-2
