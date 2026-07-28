"""Pure steps for turning IAM installed capacities into PyPSA targets (p_nom_min).

In pipeline order: consolidate variant tokens (:func:`apply_consolidation`), put link-like
technologies on an input-capacity basis (:func:`convert_capacities_to_input_capacity_basis`), then map
model tech tokens to PyPSA carriers and sum (:func:`aggregate_capacities_to_carriers`).

Reading the symbols and sequencing these is the Coupler's job — see
``Coupler.prepare_capacities`` / ``Coupler.build_capacity_targets``.
"""

import logging
from collections.abc import Sequence

import pandas as pd

logger = logging.getLogger(__name__)


def convert_capacities_to_input_capacity_basis(
    capacities: pd.DataFrame,
    efficiencies: pd.DataFrame,
    link_techs: set[str],
    *,
    on: Sequence[str] = ("year", "region", "technology"),
    tech_col: str = "technology",
    value_col: str = "value",
    eff_col: str = "efficiency",
) -> pd.DataFrame:
    """Divide output-based capacities by efficiency for link-like techs (→ input basis).

    PyPSA links are defined by input capacity, but IAMs report output capacity.

    Args:
        capacities: Long frame with a technology and a value column.
        efficiencies: Long frame with the same key columns plus ``eff_col``.
        link_techs: Technology tokens to convert; other rows pass through unchanged.
        on: Columns to join ``capacities`` and ``efficiencies`` on.
        tech_col: Column identifying the technology.
        value_col: Column to divide.
        eff_col: Efficiency column in ``efficiencies``.

    Returns:
        ``capacities`` with ``value_col`` divided by efficiency for ``link_techs`` rows. Rows
        with missing or zero efficiency are left unchanged (with a warning).
    """
    merged = capacities.merge(efficiencies, on=list(on), how="left")
    is_link = merged[tech_col].isin(link_techs)
    missing = is_link & merged[eff_col].isna()
    zero = is_link & (merged[eff_col] == 0)
    if missing.any():
        logger.warning("Missing efficiency for %d link rows; keeping originals.", int(missing.sum()))
    if zero.any():
        logger.warning("Zero efficiency for %d link rows; keeping originals.", int(zero.sum()))
    valid = is_link & merged[eff_col].notna() & (merged[eff_col] != 0)
    merged.loc[valid, value_col] = merged.loc[valid, value_col] / merged.loc[valid, eff_col]
    return merged.drop(columns=[eff_col])


def aggregate_capacities_to_carriers(
    capacities: pd.DataFrame,
    tech_to_carrier: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("year", "region"),
    tech_col: str = "technology",
    map_tech_col: str,
    map_carrier_col: str,
    value_col: str = "value",
    unit: str = "MW",
    min_value: float = 0.0,
    round_digits: int = 2,
) -> pd.DataFrame:
    """Map model tech tokens to target carrier names, sum per (group, carrier).

    ``tech_to_carrier`` may be many-to-many: several tokens sharing a carrier are summed; one
    token feeding several carriers is preserved (not deduped), each carrier getting the full
    value.

    Args:
        capacities: Long frame with a technology and a value column.
        tech_to_carrier: Model tech → target carrier mapping table.
        group_cols: Columns to group by, alongside the mapped carrier.
        tech_col: Column identifying the technology in ``capacities``.
        map_tech_col: Column in ``tech_to_carrier`` holding the technology token.
        map_carrier_col: Column in ``tech_to_carrier`` holding the target carrier.
        value_col: Column to sum.
        unit: Unit label stamped onto the output.
        min_value: Rows at or below this value, after summing, are dropped.
        round_digits: Decimal places to round the summed value to.

    Returns:
        ``[*group_cols, carrier, value, unit]``. Rows whose tech token is absent from
        ``tech_to_carrier`` are dropped with a warning.
    """
    carrier_map = tech_to_carrier[[map_tech_col, map_carrier_col]].drop_duplicates()
    mapped = capacities.merge(carrier_map, left_on=tech_col, right_on=map_tech_col, how="left")
    unmapped = mapped[map_carrier_col].isna().sum()
    if unmapped:
        logger.warning("Dropping %d rows with unmapped technologies.", int(unmapped))
    mapped = mapped.dropna(subset=[map_carrier_col]).rename(columns={map_carrier_col: "carrier"})

    grouped = (
        mapped.groupby([*group_cols, "carrier"], as_index=False, observed=True)[value_col]
        .sum()
        .round(round_digits)
    )
    grouped = grouped[grouped[value_col] > min_value]
    grouped["unit"] = unit
    return grouped.sort_values([*group_cols, "carrier"]).reset_index(drop=True)


def apply_consolidation(
    caps: pd.DataFrame,
    *,
    vre_to_primary: dict[str, str] | None = None,
    battery_scaling: dict[str, float] | None = None,
    tech_col: str = "technology",
    value_col: str = "value",
) -> pd.DataFrame:
    """Apply the optional ``consolidation`` block from the capacity symbol spec.

    Two steps, both driven by config — no-op when params are absent (e.g. IAMC configs
    that have no ``consolidation`` block):

    1. **Token rename**: rename coupled variant tokens to their primary token via
       ``vre_to_primary`` (e.g. ``elh2VRE`` → ``elh2``; also used for battery-scaling targets
       below, e.g. ``storspv`` → ``btin``).
    2. **Battery scaling**: multiply each ``battery_scaling`` source row by its scaling factor
       before it's renamed to its ``vre_to_primary`` target. If that target already carries a
       positive value on its own, the source rows are dropped instead of scaled (bidirectional-
       coupling guard).

    Args:
        caps: Long frame with a technology and a value column.
        vre_to_primary: Coupled variant token → primary token renames.
        battery_scaling: Source token → scale factor, applied before the rename.
        tech_col: Column identifying the technology.
        value_col: Column battery scaling multiplies.

    Returns:
        ``caps`` with battery-scaling applied and technology tokens renamed.
    """
    caps = caps.copy()
    vre_to_primary = vre_to_primary or {}
    battery_scaling = battery_scaling or {}

    tech = caps[tech_col].astype(str)

    if battery_scaling:
        targets = {src: vre_to_primary.get(src, src) for src in battery_scaling}
        is_target_present = tech.isin(set(targets.values())) & (caps[value_col] > 0)
        is_stor = tech.isin(battery_scaling)
        if is_target_present.any():
            caps = caps[~is_stor].copy()
            tech = caps[tech_col].astype(str)
        else:
            scale = tech.map(battery_scaling)
            caps.loc[scale.notna(), value_col] *= scale[scale.notna()]

    caps[tech_col] = tech.map(lambda t: vre_to_primary.get(t, t))
    return caps


