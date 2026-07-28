"""CO2 price pathway extraction.

The transform is name-agnostic: it takes an already-loaded frame with canonical columns
``[region, year, value]`` (the loader/Coupler handles the GDX symbol + renames). Unit
conversion happens at the load seam and currency scaling in ``costs.apply_currency_factor``,
so nothing here changes a magnitude.
"""

from collections.abc import Sequence

import pandas as pd


def extract_co2_prices(
    raw: pd.DataFrame,
    regions: Sequence[str] | None = None,
    years: Sequence[int] | None = None,
    *,
    region_col: str = "region",
    year_col: str = "year",
    value_col: str = "value",
) -> pd.DataFrame:
    """Extract the per-(region, year) CO2 price pathway, filtered and reindexed.

    Args:
        raw: Long frame carrying at least ``region_col``, ``year_col``, ``value_col``.
        regions: Regions to keep. ``None`` keeps every region present in ``raw``.
        years: If given, reindex to the full ``regions × years`` grid, filling missing
            entries with 0. ``None`` leaves the row set as-is.
        region_col: Region column name.
        year_col: Year column name.
        value_col: Price column name.

    Returns:
        ``[region_col, year_col, value_col]``, sorted by region then year.
    """
    df = raw[[region_col, year_col, value_col]].copy()
    df[year_col] = df[year_col].astype(int)
    if regions is not None:
        df = df[df[region_col].isin(set(regions))]
    if years is not None:
        grid = pd.MultiIndex.from_product(
            [sorted(set(df[region_col]) if regions is None else set(regions)),
             [int(y) for y in years]],
            names=[region_col, year_col],
        )
        df = (
            df.set_index([region_col, year_col])[value_col]
            .reindex(grid, fill_value=0.0)
            .reset_index()
        )
    return df.sort_values([region_col, year_col]).reset_index(drop=True)


