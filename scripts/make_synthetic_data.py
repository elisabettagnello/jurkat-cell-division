"""Generate a synthetic time course with known ground truth.

The real measurements belong to the host laboratory and are not distributed,
so this script produces data with the same structure to test the pipeline:

- proliferation dye: molecules split binomially at each division, with
  probability p_true to one daughter (exactly the partitioning model);
- DNA content: G1 / S / G2 mixture plus doublets;
- FSC/SSC: phase-dependent size plus debris and aggregates.

Usage: python scripts/make_synthetic_data.py [--p 0.53] [--cells 12000]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HOURS = {'Exp0': 0, 'Exp1': 18, 'Exp2': 21, 'Exp3': 24, 'Exp4': 42, 'Exp5': 45,
         'Exp6': 48, 'Exp7': 66, 'Exp8': 69, 'Exp9': 72, 'Exp10': 90}
DOUBLING_H = 22.0
PHASE_P = {'G1': 0.5, 'S': 0.3, 'G2': 0.2}
FSC_G1, SIZE_RATIO = 115000.0, {'G1': 1.0, 'S': 1.2, 'G2': 1.4}
DNA_G1, DNA_CV = 48000.0, 0.04


def simulate_sample(rng, hours, n, p_true):
    # Generations: spread around the expected number of divisions so far.
    gen = np.clip(np.rint(rng.normal(hours / DOUBLING_H, 0.55, n)), 0, None).astype(int)
    gen[hours == 0] = 0

    # Dye molecules: log-normal start, then one binomial split per division.
    mol = np.rint(2 ** rng.normal(15.0, 0.12, n)).astype(np.int64)
    for k in range(gen.max()):
        side = np.where(rng.random(n) < 0.5, p_true, 1 - p_true)
        mol = np.where(gen > k, rng.binomial(mol, side), mol)

    phase = rng.choice(list(PHASE_P), size=n, p=list(PHASE_P.values()))
    dna = np.where(phase == 'G1', rng.normal(DNA_G1, DNA_CV * DNA_G1, n),
          np.where(phase == 'G2', rng.normal(1.95 * DNA_G1, DNA_CV * 1.95 * DNA_G1, n),
                   rng.uniform(DNA_G1, 1.95 * DNA_G1, n) * rng.normal(1, DNA_CV, n)))
    ratio = np.vectorize(SIZE_RATIO.get)(phase)
    fsc = rng.normal(FSC_G1 * ratio * (1 + 0.0005 * hours), 9000, n)
    ssc = 0.55 * fsc + rng.normal(0, 7000, n)
    width = rng.normal(80000, 2500, n)
    dye = mol * rng.lognormal(0, 0.02, n)

    # Doublets: two cells through the laser together (wider pulse, double DNA).
    dbl = rng.random(n) < 0.03
    dna[dbl] *= 2.0
    width[dbl] *= 1.25
    fsc[dbl] *= 1.3
    # Debris: small, dim events.
    deb = rng.random(n) < 0.05
    fsc[deb] = rng.uniform(5000, 45000, deb.sum())
    ssc[deb] = rng.uniform(2000, 30000, deb.sum())
    dna[deb] = rng.uniform(1000, 30000, deb.sum())
    dye[deb] = 2 ** rng.uniform(6, 12, deb.sum())

    return pd.DataFrame({'FSC-A': fsc, 'SSC-A': ssc, 'PE-A': dye,
                         'V450-A': dna, 'V450-W': width})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--p', type=float, default=0.53, help='true partitioning parameter')
    ap.add_argument('--cells', type=int, default=12000, help='events per time point')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', default='data/synthetic')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames, times = [], []
    for exp, h in HOURS.items():
        frames.append(simulate_sample(rng, h, args.cells, args.p).assign(Exp=exp))
        times.append({'Exp': exp, 'minutes': 60 * h + rng.uniform(-20, 20) * (h > 0)})
    pd.concat(frames, ignore_index=True).to_csv(out / 'events.csv', index=False, float_format='%.1f')
    pd.DataFrame(times).to_csv(out / 'times.csv', index=False)
    print(f"Wrote {out / 'events.csv'} and {out / 'times.csv'} (p_true = {args.p})")


if __name__ == '__main__':
    main()
