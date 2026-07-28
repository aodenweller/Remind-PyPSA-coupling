"""REMIND symbol readers for GDX containers (via ``gamspy``).

Provides symbol/scalar readers and a symbol-name listing, with the opened container cached
per path so repeated reads against the same GDX file don't re-open it.
"""

import functools
from collections.abc import Mapping
from os import PathLike

import pandas as pd


@functools.lru_cache
def _open_container(path: str):
    """Open (and cache) a GDX container."""
    from gamspy import Container

    return Container(load_from=path)


def read_gdx_symbol(
    path: str | PathLike,
    symbol: str,
    rename_columns: Mapping[str, str] | None = None,
    error_on_empty: bool = True,
) -> pd.DataFrame | None:
    """Read one GDX symbol into a long DataFrame.

    Args:
        path: GDX file to read from.
        symbol: GDX symbol name.
        rename_columns: Column renames to apply on read.
        error_on_empty: Raise if the symbol has no records, rather than returning ``None``.

    Returns:
        The symbol's records as a long DataFrame, or ``None`` if empty and
        ``error_on_empty`` is ``False``.

    Raises:
        ValueError: If the symbol is empty and ``error_on_empty`` is ``True``.
    """
    data = _open_container(str(path))[symbol]
    df = data.records
    if error_on_empty and (df is None or df.empty):
        raise ValueError(f"{symbol} is empty. In: {path}")
    if df is None:
        return df
    if rename_columns:
        # .rename() ignores keys absent from df's columns by default — needed because candidate
        # fallbacks may have different columns (e.g. the v32 variable has 'level', the p32
        # parameter has 'value').
        df = df.rename(columns=dict(rename_columns))
    return df


def read_gdx_scalar(path: str | PathLike, symbol: str) -> float | str:
    """Read a scalar/string GDX symbol (e.g. model version, run name).

    Args:
        path: GDX file to read from.
        symbol: GDX symbol name.

    Returns:
        The scalar value: a single ``value`` for a scalar parameter, or the first element
        label for a string set.

    Raises:
        ValueError: If the symbol has no records.
    """
    df = read_gdx_symbol(path, symbol, error_on_empty=False)
    if df is None or df.empty:
        raise ValueError(f"{symbol} has no records in {path}")
    # scalar parameter -> single 'value'; string set -> first element label
    if "value" in df.columns and len(df) == 1:
        return df["value"].iloc[0]
    return df.iloc[0, 0]


def list_gdx_symbols(path: str | PathLike) -> list[str]:
    """List the symbol names available in a GDX container."""
    return sorted(_open_container(str(path)).data.keys())
