"""Time-aware data splitting utilities to prevent temporal data leakage."""

from __future__ import annotations

import pandas as pd


def time_based_split(
    df: pd.DataFrame,
    date_col: str,
    test_fraction: float = 0.2,
    cutoff_date: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe strictly chronologically by `date_col`."""
    ordered = df.sort_values(date_col).reset_index(drop=True)
    ordered[date_col] = pd.to_datetime(ordered[date_col])

    if cutoff_date is not None:
        cutoff = pd.Timestamp(cutoff_date)
        train_df = ordered[ordered[date_col] < cutoff].copy()
        test_df = ordered[ordered[date_col] >= cutoff].copy()
        return train_df, test_df

    split_idx = int(len(ordered) * (1.0 - test_fraction))
    split_idx = max(1, min(split_idx, len(ordered) - 1))
    return ordered.iloc[:split_idx].copy(), ordered.iloc[split_idx:].copy()
