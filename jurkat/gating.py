"""Gates that isolate single, intact cells before any downstream analysis.

All gates return a boolean mask of the events to KEEP, so they can be combined
with `&` and their effect reported event by event.
"""

import numpy as np
from matplotlib.path import Path as MplPath


def polygon_gate(df, x, y, vertices):
    """Keep events inside a polygon in the (x, y) plane, e.g. FSC-A vs SSC-A.

    vertices: list of (x, y) points; the polygon is closed automatically.
    """
    path = MplPath(np.asarray(vertices, dtype=float))
    return path.contains_points(df[[x, y]].to_numpy())


def range_gate(df, channel, low=None, high=None):
    """Keep events with low <= channel <= high (either bound may be None)."""
    keep = np.ones(len(df), dtype=bool)
    if low is not None:
        keep &= df[channel].to_numpy() >= low
    if high is not None:
        keep &= df[channel].to_numpy() <= high
    return keep


def below_line_gate(df, x, y, slope, x0, y0):
    """Keep events on or below the line y = slope * (x - x0) + y0.

    Used to remove aggregates, which sit above the diagonal of a
    DNA-content (V450-A) vs size (FSC-A) plot.
    """
    return df[y].to_numpy() <= slope * (df[x].to_numpy() - x0) + y0


def dapi_gate(df, cfg, fsc='FSC-A', dapi_area='V450-A', dapi_width='V450-W'):
    """DNA-content gate used for the cell-cycle analysis.

    cfg keys: slope, x0, y0 (aggregate diagonal), area_low, area_high (noise),
              width_low, width_high (doublets, from the pulse width).
    """
    keep = below_line_gate(df, fsc, dapi_area, cfg['slope'], cfg['x0'], cfg['y0'])
    keep &= range_gate(df, dapi_area, cfg.get('area_low'), cfg.get('area_high'))
    keep &= range_gate(df, dapi_width, cfg.get('width_low'), cfg.get('width_high'))
    return keep


def report(name, keep):
    n, k = len(keep), int(np.sum(keep))
    print(f"  {name}: kept {k}/{n} events ({100 * k / max(n, 1):.1f}%)")
