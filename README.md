# Cytoplasm partitioning and cell-cycle growth in Jurkat cells

Quantitative flow cytometry analysis from my internship at the Istituto Italiano di Tecnologia
(IIT). The project asks two questions about non-genetic
heterogeneity in Jurkat cells (a human T-cell lymphoma line):

1. **Is the cytoplasm split symmetrically between daughter cells?** If total cytoplasm is
   partitioned symmetrically, the known asymmetric segregation of organelles such as mitochondria
   must be an active, regulated process rather than a random one.
2. **How do cells control their size along the cell cycle?** Sizer, timer or adder, and does the
   growth rate change between phases?

## Experiment

Cells were stained with CellTrace Yellow (cytoplasm) and MitoTracker Far Red (mitochondria), and a
narrow, bright starting population (generation 0) was isolated by fluorescence-activated cell sorting
(FACS). Samples were measured three times a day for three days, over two independent weeks. At each
time point a fixed aliquot was stained with DAPI to measure DNA content.

## Materials

- [Internship report (PDF)](report/Internship_report_IIT.pdf)
- [Slides](https://elisabettagnello.github.io/jurkat-cell-division/presentation/)
- 
## Analysis pipeline

| Step | Method | Module |
|---|---|---|
| Gating | FSC/SSC polygon to remove debris; DAPI area vs FSC to remove aggregates; DAPI area vs pulse width to remove doublets | `jurkat/gating.py` |
| Generations | The dye halves at each division, so in log2 scale generations are equally spaced peaks. A Gaussian mixture with equally spaced means is fitted with a purpose-written EM algorithm; the number of peaks is chosen by BIC and generation labels are tracked across time points | `jurkat/generations.py` |
| Partitioning | Asymmetric binomial model: mean and variance of the dye per generation as a function of the split fraction p, fitted to the per-generation moments | `jurkat/partitioning.py` |
| Cell cycle | Watson-Pragmatic decomposition of the DAPI histogram: Gaussian G1 and G2/M peaks, S phase as a broadened polynomial; each cell gets phase probabilities | `jurkat/cell_cycle.py` |
| Size and growth | Growth exponent α = 3 log2(⟨FSC⟩G2 / ⟨FSC⟩G1), linearised size D = FSC^(1/α), growth rate β per phase from weighted linear fits | `jurkat/growth.py` |

## Results (original analysis, see the slides)

| Dataset | p (partitioning) | χ²/dof |
|---|---|---|
| Week 1, live cells | 0.545 ± 0.007 | 1.06 |
| Week 1, fixed (DAPI) | 0.548 ± 0.003 | 2.78 |
| Week 2, live cells | 0.519 ± 0.004 | 0.09 |
| Week 2, fixed (DAPI) | 0.516 ± 0.005 | 0.54 |

- Partitioning of total cytoplasm is close to symmetric (p ≈ 0.5), and live and fixed cells agree,
  so fixation does not distort the measurement.
- Added size grows weakly with birth size (slope m ≈ 0.2): an **adder-like** size control.
- α = 1.45 ± 0.02 (week 1) and 1.53 ± 0.03 (week 2); growth accelerates through the cycle,
  β(G1) < β(S) < β(G2).
- Main limitation: forward scatter is a proxy of cell size, not a direct volume measurement.

## Validation on synthetic data

The raw measurements belong to the host laboratory and are not included. `scripts/make_synthetic_data.py`
generates a time course with the same structure and a known ground truth (binomial dye partitioning
with a chosen p, a G1/S/G2 mixture, debris and doublets), so the whole pipeline can be run and checked:

| Ground truth | Recovered |
|---|---|
| p = 0.53 | 0.537 ± 0.002 |
| p = 0.50 | 0.511 ± 0.003 |
| generation labels at each time point | all correct |

The small upward bias in p (about 0.01) comes from measurement noise and peak overlap, which add
variance that the model attributes to asymmetry: values of p just above 0.5 should be read with this in mind.

## Running it

```bash
pip install -r requirements.txt
python scripts/make_synthetic_data.py          # writes data/synthetic/
python run_pipeline.py config/synthetic.toml    # tables and figures in results/synthetic/
pytest                                         # model sanity checks
```

For real data, copy `config/synthetic.toml`, point it to an events table (one row per event, with
an `Exp` column) or adapt `jurkat/io.py` to read the `.fcs` files, and set the gate values after
inspecting `figures/gating.png`. Generations can be listed per sample in `[generations.per_exp]`.

## Repository layout

```
jurkat/               analysis package (I/O, gating, generations, partitioning, cell cycle, growth)
run_pipeline.py       end-to-end analysis driven by a TOML config
config/               configurations (synthetic demo)
scripts/              synthetic data generator
tests/                sanity checks with known answers
presentation/         slides (HTML)
```

## Credits

During the internship, generation deconvolution was performed with the *CellDivision* tool developed
by the host group. This repository is my independent reimplementation
of the full analysis; it does not contain code from that tool.
