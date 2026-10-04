"""End-to-end analysis of a flow cytometry time course.

    python run_pipeline.py config/synthetic.toml

Steps: gating -> generations (dye dilution) -> partitioning parameter p ->
cell-cycle phases (DAPI) -> growth exponent alpha and growth rates beta.
Tables go to <output>/tables, figures to <output>/figures.
"""

import argparse
import json
import tomllib
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jurkat import cell_cycle, gating, generations, growth, partitioning
from scipy.optimize import OptimizeWarning

from jurkat.io import exp_order, load_events_table, safe_log2

warnings.filterwarnings('ignore', category=OptimizeWarning)

PHASE_COLORS = {'G1': '#2f4b7c', 'S': '#d45087', 'G2': '#ff7c43'}
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.3, 'font.size': 11})


def grid(n, ncols=4, size=3.2):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * size * 1.25, nrows * size), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes.flat


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / f'{name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved figures/{name}.png")


# ---------------------------------------------------------------------------
def step_gating(events, cfg, ch, figs):
    print("\n[1] Gating")
    sc = cfg['gates']['scatter']
    keep_scatter = gating.polygon_gate(events, ch['fsc'], ch['ssc'], sc['vertices'])
    gating.report('FSC/SSC (cells vs debris)', keep_scatter)
    cells = events[keep_scatter].copy()

    keep_dna = None
    if 'dapi' in cfg['gates'] and ch.get('dna') in events.columns:
        keep_dna = gating.dapi_gate(cells, cfg['gates']['dapi'], ch['fsc'], ch['dna'], ch['dna_width'])
        gating.report('DAPI (aggregates, noise, doublets)', keep_dna)

    # Figure: one representative sample.
    exp = cfg['gates'].get('plot_exp', exp_order(events)[0])
    e = events[events['Exp'] == exp]
    k = keep_scatter[events['Exp'].to_numpy() == exp]
    ncol = 3 if keep_dna is not None else 1
    fig, axes = plt.subplots(1, ncol, figsize=(5.2 * ncol, 4.4), squeeze=False)
    ax = axes[0, 0]
    ax.scatter(e[ch['fsc']][~k], e[ch['ssc']][~k], s=2, c='#bbbbbb', label='removed')
    ax.scatter(e[ch['fsc']][k], e[ch['ssc']][k], s=2, c='#2f4b7c', alpha=0.4, label='cells')
    v = np.array(sc['vertices'] + [sc['vertices'][0]])
    ax.plot(v[:, 0], v[:, 1], c='#d45087', lw=2)
    ax.set(xlabel=ch['fsc'], ylabel=ch['ssc'], title=f'Scatter gate ({exp})')
    ax.legend(markerscale=5, loc='upper left')
    if keep_dna is not None:
        c = cells[cells['Exp'] == exp]
        kd = keep_dna[cells['Exp'].to_numpy() == exp]
        g = cfg['gates']['dapi']
        ax = axes[0, 1]
        ax.scatter(c[ch['fsc']][~kd], c[ch['dna']][~kd], s=2, c='#bbbbbb')
        ax.scatter(c[ch['fsc']][kd], c[ch['dna']][kd], s=2, c='#2f4b7c', alpha=0.4)
        xx = np.array(ax.get_xlim())
        ax.plot(xx, g['slope'] * (xx - g['x0']) + g['y0'], c='#d45087', lw=2, ls='--')
        ax.set(xlabel=ch['fsc'], ylabel=ch['dna'], title='Aggregates')
        ax = axes[0, 2]
        ax.scatter(c[ch['dna_width']][~kd], c[ch['dna']][~kd], s=2, c='#bbbbbb')
        ax.scatter(c[ch['dna_width']][kd], c[ch['dna']][kd], s=2, c='#2f4b7c', alpha=0.4)
        for key in ('width_low', 'width_high'):
            ax.axvline(g[key], c='#ff7c43', ls='--')
        ax.set(xlabel=ch['dna_width'], ylabel=ch['dna'], title='Doublets (pulse width)')
    save(fig, figs, 'gating')
    return cells, keep_dna


def step_generations(cells, cfg, ch, figs):
    print("\n[2] Generations from dye dilution")
    gcfg = cfg['generations']
    delta = gcfg.get('delta')  # number to fix the spacing, absent to fit it
    explicit = gcfg.get('per_exp', {})
    exps = exp_order(cells)  # chronological order: labels are tracked in time
    y_all = safe_log2(cells[ch['dye']])
    m0_ref = None
    rows, fits = [], {}
    cells['generation'] = -1
    fig, axes = grid(len(exps))
    for ax, exp in zip(axes, exps):
        m = (cells['Exp'] == exp).to_numpy()
        y = y_all[m]
        if exp in explicit:
            fit = generations.fit_generations_multistart(y, explicit[exp], delta=delta)
        elif m0_ref is None:
            # First sample: the sorted starting population is generation 0.
            fit = generations.fit_generations(y, [0], delta=delta)
        else:
            fit = generations.fit_tracked(y, m0_ref, gcfg.get('max_components', 4),
                                          gcfg.get('max_generation', 8), delta=delta)
        m0_ref = fit.m0
        fits[exp] = fit
        cells.loc[m, 'generation'] = fit.assign(y)
        for r in generations.generation_moments(fit, y, cells.loc[m, ch['dye']].to_numpy()):
            rows.append({'Exp': exp, **r})
        print(f"  {exp}: generations {list(fit.generations)}, spacing = {fit.delta:.3f} log2 units")

        bins = np.linspace(np.percentile(y_all, 0.5), np.percentile(y_all, 99.9), 120)
        ax.hist(y, bins=bins, density=True, color='#cccccc')
        xx = np.linspace(bins[0], bins[-1], 400)
        cmap = plt.get_cmap('viridis')
        for j, gen in enumerate(fit.generations):
            comp = fit.weights[j] * np.exp(-0.5 * ((xx - fit.means[j]) / fit.sigmas[j]) ** 2) / (np.sqrt(2 * np.pi) * fit.sigmas[j])
            ax.fill_between(xx, comp, color=cmap(gen / 7), alpha=0.6, label=f'gen {gen}')
        ax.plot(xx, fit.density(xx), c='k', lw=1)
        ax.set(title=exp, xlabel=f'log2 {ch["dye"]}')
        ax.legend(fontsize=7)
    save(fig, figs, 'generations')
    return pd.DataFrame(rows), fits


def step_partitioning(table, cfg, figs, tables):
    print("\n[3] Partitioning parameter p")
    pcfg = cfg.get('partitioning', {})
    agg = partitioning.aggregate_snapshots(table, exclude_exps=pcfg.get('exclude_exps', []),
                                           min_count=pcfg.get('min_count', 0))
    agg.to_csv(tables / 'generation_moments_averaged.csv', index=False)
    res_p, mu0, var0 = partitioning.fit_partition_p(agg)
    res_mu = partitioning.fit_mu0(agg)
    print(f"  p   = {res_p}")
    print(f"  mu0 = {res_mu}")

    d = partitioning.usable(agg, 'var')
    g = d['generation'].to_numpy(float)
    gg = np.linspace(0, g.max(), 200)
    sigma0 = np.sqrt(var0)
    sig = np.sqrt(d['var'].to_numpy())
    sig_err = d['var_sem'].to_numpy() / (2 * sig)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    a1.errorbar(g, sig / sigma0, yerr=sig_err / sigma0, fmt='o', c='#d45087', capsize=4, label='data ± SEM')
    a1.plot(gg, np.sqrt(partitioning.variance_model(gg, res_p.value, mu0, var0)) / sigma0, c='#2f4b7c',
            ls='-.', label=f'fit, p = {res_p.value:.3f}')
    a1.plot(gg, np.sqrt(partitioning.variance_model(gg, 0.5, mu0, var0)) / sigma0, c='#ff7c43',
            ls=':', label='symmetric, p = 0.5')
    a1.set(yscale='log', xlabel='generation g', ylabel=r'$\sigma_g / \sigma_0$', title='Normalised standard deviation')
    a1.set_yscale('log', base=2)
    a1.set_xticks(range(int(g.max()) + 1))
    a1.legend()
    dm = partitioning.usable(agg, 'mean')
    gm = dm['generation'].to_numpy(float)
    mu0_obs = agg.loc[agg['generation'] == 0, 'mean'].iloc[0]
    a2.errorbar(gm, dm['mean'] / mu0_obs, yerr=dm['mean_sem'] / mu0_obs, fmt='o', c='#d45087', capsize=4, label='data ± SEM')
    a2.plot(gg, 0.5 ** gg, c='#2f4b7c', ls='--', label=r'$(1/2)^g$')
    a2.set_yscale('log', base=2)
    a2.set_xticks(range(int(gm.max()) + 1))
    a2.set(xlabel='generation g', ylabel=r'$\mu_g / \mu_0$', title='Normalised mean intensity')
    a2.legend()
    save(fig, figs, 'partitioning')
    return {'p': res_p.value, 'p_err': res_p.error, 'p_chi2_red': res_p.chi2_red, 'p_pvalue': res_p.p_value,
            'mu0_fit': res_mu.value, 'mu0_err': res_mu.error}


def step_cell_cycle(cells, keep_dna, cfg, ch, figs, tables):
    print("\n[4] Cell-cycle phases (DAPI)")
    ccfg = cfg.get('cell_cycle', {})
    dna_cells = cells[keep_dna].copy()
    exps = exp_order(dna_cells)
    fracs, parts = [], []
    fig, axes = grid(len(exps))
    for ax, exp in zip(axes, exps):
        d = dna_cells[dna_cells['Exp'] == exp].copy()
        if len(d) < ccfg.get('min_events', 2000):
            print(f"  {exp}: too few events, skipped")
            continue
        try:
            fit = cell_cycle.fit_cell_cycle(d[ch['dna']], **ccfg.get('fit', {}))
        except (RuntimeError, ValueError) as err:
            print(f"  {exp}: fit failed ({err})")
            continue
        probs = fit.phase_probabilities(d[ch['dna']])
        d['phase'] = np.array(['G1', 'S', 'G2'])[probs.argmax(axis=1)]
        parts.append(d)
        f = fit.fractions()
        fracs.append({'Exp': exp, **f, 'cv_G1': fit.cv})
        print(f"  {exp}: G1 {100 * f['G1']:.1f}%  S {100 * f['S']:.1f}%  G2 {100 * f['G2']:.1f}%")

        g1, s, g2 = fit.components()
        ax.step(fit.x, fit.counts, where='mid', c='k', lw=0.8)
        for comp, ph in zip((g1, s, g2), ('G1', 'S', 'G2')):
            ax.fill_between(fit.x, comp, color=PHASE_COLORS[ph], alpha=0.5, label=ph)
        ax.plot(fit.x, g1 + s + g2, c='#d45087', ls='--', lw=1)
        ax.set(xlim=(0.5 * fit.params[1], 1.5 * fit.params[4]), title=exp, xlabel=ch['dna'])
        ax.legend(fontsize=7)
    save(fig, figs, 'cell_cycle')
    pd.DataFrame(fracs).to_csv(tables / 'cell_cycle_fractions.csv', index=False)
    return pd.concat(parts, ignore_index=True)


def step_growth(dna_cells, times, cfg, ch, figs, tables):
    print("\n[5] Size: growth exponent alpha and growth rates beta")
    gcfg = cfg['growth']
    alpha = growth.growth_exponent(dna_cells, gcfg['alpha_exps'], size=ch['fsc'])
    print(f"  alpha = {alpha.value:.3f} ± {alpha.error:.3f}")
    points, rates = growth.phase_growth_rates(dna_cells, gcfg['rate_exps'], times, alpha.value, size=ch['fsc'])
    rates.to_csv(tables / 'growth_rates.csv', index=False)
    print(rates.to_string(index=False, float_format=lambda v: f'{v:.3f}'))

    fr = growth.generation_fractions(dna_cells)
    fr.to_csv(tables / 'generation_fractions.csv')
    fig, ax = plt.subplots(figsize=(8, 4.4))
    hrs = [times[e] / 60 for e in fr.index]
    cmap = plt.get_cmap('viridis')
    for gen in fr.columns:
        ax.plot(hrs, fr[gen], 'o--', c=cmap(gen / 7), label=f'gen {gen}')
    ax.set(xlabel='time (h)', ylabel='fraction of cells', title='Generation fractions over time')
    ax.legend(fontsize=8, ncol=2)
    save(fig, figs, 'generation_fractions')

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    pe = alpha.per_exp
    a1.scatter([times[e] / 60 for e in pe['Exp']], pe['alpha'], s=20 + 80 * pe['n'] / pe['n'].max(), c='#2f4b7c')
    a1.axhline(alpha.value, c='#d45087', ls='--', label=f'weighted mean = {alpha.value:.2f} ± {alpha.error:.2f}')
    a1.set(xlabel='time (h)', ylabel=r'$\alpha = 3\,\log_2(\langle FSC\rangle_{G2} / \langle FSC\rangle_{G1})$',
           title='Growth exponent')
    a1.legend()
    for ph, grp in points.groupby('phase'):
        a2.scatter(grp['minutes'] / 60, grp['D'], c=PHASE_COLORS[ph], s=20 + 80 * grp['n'] / points['n'].max())
        r = rates[rates['phase'] == ph].iloc[0]
        tt = np.linspace(grp['minutes'].min(), grp['minutes'].max(), 50)
        a2.plot(tt / 60, r['beta_per_h'] / 60 * tt + r['D0'], c=PHASE_COLORS[ph],
                label=f"{ph}: β = {r['beta_per_h']:.2f} /h")
    a2.set(xlabel='time (h)', ylabel=r'$D = FSC^{1/\alpha}$', title='Linearised size by phase')
    a2.legend()
    save(fig, figs, 'growth')
    return {'alpha': alpha.value, 'alpha_err': alpha.error,
            'beta_per_h': dict(zip(rates['phase'], rates['beta_per_h']))}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config', help='TOML configuration file')
    args = ap.parse_args()
    cfg = tomllib.loads(Path(args.config).read_text())
    ch = cfg['channels']
    out = Path(cfg['output']['dir'])
    figs, tables = out / 'figures', out / 'tables'
    figs.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    events = load_events_table(cfg['input']['events'])
    times = pd.read_csv(cfg['input']['times']).set_index('Exp')['minutes'].to_dict()
    print(f"Loaded {len(events)} events, {events['Exp'].nunique()} time points")

    cells, keep_dna = step_gating(events, cfg, ch, figs)
    gen_table, _ = step_generations(cells, cfg, ch, figs)
    gen_table.to_csv(tables / 'generation_moments.csv', index=False)
    summary = {'partitioning': step_partitioning(gen_table, cfg, figs, tables)}

    if keep_dna is not None:
        dna_cells = step_cell_cycle(cells, keep_dna, cfg, ch, figs, tables)
        summary['growth'] = step_growth(dna_cells, times, cfg, ch, figs, tables)
        dna_cells.to_csv(tables / 'cells_annotated.csv.gz', index=False, compression='gzip')

    (out / 'summary.json').write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nDone. Summary in {out / 'summary.json'}")


if __name__ == '__main__':
    main()
