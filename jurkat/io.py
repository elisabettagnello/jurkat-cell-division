"""Loading flow cytometry events and acquisition times."""

from pathlib import Path

import numpy as np
import pandas as pd


def read_fcs(path, channels=None):
    """Read one .fcs file into a DataFrame (optionally keeping only `channels`).

    Returns (events, metadata). Requires the `fcsparser` package.
    """
    import fcsparser  # optional dependency, only needed for raw .fcs input

    meta, data = fcsparser.parse(str(path), reformat_meta=True)
    if channels is not None:
        missing = set(channels) - set(data.columns)
        if missing:
            raise KeyError(f"{path}: channels not found: {sorted(missing)}")
        data = data[list(channels)]
    return data, meta


def hms_to_minutes(hms):
    h, m, s = (int(float(x)) for x in str(hms).split(':')[:3])
    return 60 * h + m + s / 60


def load_fcs_folder(folder, samples, channels, time_keyword='$ETIM'):
    """Load a time course stored as one .fcs file per time point.

    samples: list of dicts with keys `file`, `exp` (label used downstream),
             `day_offset_h` (hours to add to the clock time, e.g. 24 for day 1).
    Returns (events, times) where `events` has an `Exp` column and `times`
    maps each Exp to minutes elapsed since the first sample.
    """
    folder = Path(folder)
    frames, clock = [], {}
    for s in samples:
        data, meta = read_fcs(folder / s['file'], channels)
        data = data.assign(Exp=s['exp'])
        frames.append(data)
        clock[s['exp']] = hms_to_minutes(meta.get(time_keyword, '00:00:00')) + 60 * s.get('day_offset_h', 0)
    t0 = min(clock.values())
    times = {exp: t - t0 for exp, t in clock.items()}
    return pd.concat(frames, ignore_index=True), times


def load_events_table(path):
    """Load events already exported to a text/CSV table with an `Exp` column."""
    path = Path(path)
    sep = ',' if path.suffix == '.csv' else r'\s+'
    df = pd.read_csv(path, sep=sep, engine='python')
    if 'Exp' not in df.columns:
        raise KeyError(f"{path} needs an 'Exp' column identifying the time point")
    return df


def exp_order(df):
    """Exp labels sorted numerically (Exp0, Exp1, ..., Exp10)."""
    labels = df['Exp'].unique()
    try:
        return sorted(labels, key=lambda e: int(''.join(ch for ch in e if ch.isdigit())))
    except ValueError:
        return sorted(labels)


def safe_log2(x, floor=1.0):
    return np.log2(np.clip(np.asarray(x, dtype=float), floor, None))
