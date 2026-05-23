"""
Utility functions for file I/O, catalog search, results display, and local
signal validation.


These helpers are imported directly in notebooks — no API calls here.

Usage:
    from wq.utils import save_result, load_results, search_fields, top_alphas
    from wq.utils import local_signal_test
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_PATH = DATA_DIR / "results.csv"
FIELDS_PATH = DATA_DIR / "datafields.parquet"


# ---------------------------------------------------------------------------
# Results log (data/results.csv)
# ---------------------------------------------------------------------------

def save_result(result: dict, config=None) -> None:
    """
    Append a Brain simulation result to data/results.csv.

    Extracts the key metrics from the raw API response and adds a fingerprint
    column so duplicate submissions can be detected.

    Args:
        result: raw dict returned by simulate() / poll_result()
        config: AlphaConfig that produced this result (for fingerprint + tags)
    """
    DATA_DIR.mkdir(exist_ok=True)

    is_m = result.get("is", {}) or {}
    os_m = result.get("os", {}) or {}

    expression = ""
    if config is not None:
        expression = config.expression
    elif isinstance(result.get("alpha"), dict):
        expression = result["alpha"].get("regular", "")

    from wq.alpha import compute_fitness
    sharpe = is_m.get("sharpe")
    returns = is_m.get("returns")
    turnover = is_m.get("turnover")
    fitness = is_m.get("fitness") or (compute_fitness(sharpe, returns, turnover) if all(v is not None for v in [sharpe, returns, turnover]) else None)

    row = {
        "sim_id": result.get("id", ""),
        "status": result.get("status", ""),
        "expression": expression,
        "fingerprint": config.fingerprint() if config else "",
        "tags": json.dumps(config.tags if config else []),
        "universe": config.universe if config else "",
        "neutralization": config.neutralization if config else "",
        "delay": config.delay if config else "",
        "decay": config.decay if config else "",
        # In-sample metrics
        "is_sharpe": sharpe,
        "is_fitness": fitness,
        "is_returns": returns,
        "is_turnover": turnover,
        "is_drawdown": is_m.get("drawdown"),
        # Out-of-sample metrics (may be empty for new submissions)
        "os_sharpe": os_m.get("sharpe"),
        "os_fitness": os_m.get("fitness"),
        "os_returns": os_m.get("returns"),
        "os_turnover": os_m.get("turnover"),
        "notes": config.notes if config else "",
    }

    df_new = pd.DataFrame([row])

    if RESULTS_PATH.exists():
        df_new.to_csv(RESULTS_PATH, mode="a", header=False, index=False)
    else:
        df_new.to_csv(RESULTS_PATH, index=False)


def load_results(path: str | Path = None) -> pd.DataFrame:
    """
    Load all simulation results from data/results.csv.

    Returns an empty DataFrame (with correct columns) if no results exist yet.
    """
    path = Path(path) if path else RESULTS_PATH
    if not path.exists():
        return pd.DataFrame(columns=[
            "sim_id", "status", "expression", "fingerprint", "tags",
            "universe", "neutralization", "delay", "decay",
            "is_sharpe", "is_fitness", "is_returns", "is_turnover", "is_drawdown",
            "os_sharpe", "os_fitness", "os_returns", "os_turnover", "notes",
        ])
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Data field catalog helpers
# ---------------------------------------------------------------------------

def load_datafields(path: str | Path = None) -> pd.DataFrame:
    """
    Load the cached data field catalog from data/datafields.parquet.

    Raises FileNotFoundError if the catalog hasn't been downloaded yet.
    Run notebook 01_datafields.ipynb or call fetch_datafields() first.
    """
    path = Path(path) if path else FIELDS_PATH
    if not path.exists():
        raise FileNotFoundError(
            "Data field catalog not found. "
            "Run notebook 01_datafields.ipynb to download it first."
        )
    return pd.read_parquet(path)


def search_fields(df: pd.DataFrame, keyword: str = "", category: str = None, field_type: str = None) -> pd.DataFrame:
    """
    Search the data field catalog by keyword, category, or type.

    Brain fields have these key columns after normalization:
      id              — the field identifier used in alpha expressions (e.g. "close")
      description     — human-readable description
      category_name   — high-level category (e.g. "Fundamental", "Price Volume")
      dataset_name    — dataset the field belongs to
      alphacount      — how many alphas on Brain use this field

    Args:
        df:         DataFrame from load_datafields()
        keyword:    search in field id and description (case-insensitive)
        category:   filter by category name (e.g. "fundamental", "price")
        field_type: filter by type column (e.g. "MATRIX", "VECTOR")

    Returns:
        Filtered DataFrame sorted by alphacount descending (most-used fields first).
    """
    mask = pd.Series([True] * len(df), index=df.index)

    if keyword:
        kw = keyword.lower()
        # Search specifically in id and description — avoid false matches on
        # dataset_name / category_name columns
        id_col = "id" if "id" in df.columns else None
        desc_col = next((c for c in df.columns if c.lower() == "description"), None)
        text_mask = pd.Series([False] * len(df), index=df.index)
        if id_col:
            text_mask |= df[id_col].astype(str).str.lower().str.contains(kw, na=False)
        if desc_col:
            text_mask |= df[desc_col].astype(str).str.lower().str.contains(kw, na=False)
        mask &= text_mask

    if category:
        # category_name is the human-readable label (e.g. "Fundamental")
        cat_col = next(
            (c for c in df.columns if c.lower() in ("category_name", "category_id")),
            next((c for c in df.columns if "category" in c.lower() and "sub" not in c.lower()), None),
        )
        if cat_col:
            mask &= df[cat_col].astype(str).str.lower().str.contains(category.lower(), na=False)

    if field_type:
        type_col = next((c for c in df.columns if c.lower() == "type"), None)
        if type_col:
            mask &= df[type_col].astype(str).str.lower().str.contains(field_type.lower(), na=False)

    result = df[mask].copy()

    # Sort by alphaCount so the most-used fields appear first
    count_col = next((c for c in df.columns if "alphacount" in c.lower() or "alpha_count" in c.lower()), None)
    if count_col:
        result = result.sort_values(count_col, ascending=False)

    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Results analysis helpers
# ---------------------------------------------------------------------------

def top_alphas(df: pd.DataFrame = None, n: int = 20, metric: str = "is_fitness") -> pd.DataFrame:
    """
    Return the top-n alphas from the results log, sorted by metric.

    Args:
        df:     DataFrame from load_results() — loads from disk if None
        n:      number of rows to return
        metric: column to sort by (default "is_fitness")
    """
    if df is None:
        df = load_results()
    if df.empty:
        print("No results yet — run some simulations first.")
        return df

    completed = df[df["status"].str.upper().isin(["COMPLETE", "WARNING"])].copy()
    if completed.empty:
        print("No completed simulations found.")
        return completed

    return completed.sort_values(metric, ascending=False).head(n).reset_index(drop=True)


def parse_result(result: dict) -> dict:
    """
    Extract key metrics from a raw Brain API result dict into a flat dict.

    Useful for quick inspection in notebooks.
    """
    is_m = result.get("is", {}) or {}
    os_m = result.get("os", {}) or {}
    return {
        "status": result.get("status"),
        "is_sharpe": is_m.get("sharpe"),
        "is_fitness": is_m.get("fitness"),
        "is_returns": is_m.get("returns"),
        "is_turnover": is_m.get("turnover"),
        "is_drawdown": is_m.get("drawdown"),
        "os_sharpe": os_m.get("sharpe"),
        "os_fitness": os_m.get("fitness"),
    }


# ---------------------------------------------------------------------------
# Local signal test (notebook 07 helper)
# ---------------------------------------------------------------------------

def local_signal_test(signal: pd.DataFrame, returns: pd.DataFrame, forward_days: int = 1) -> dict:
    """
    Compute basic signal quality metrics locally — a 2-minute sanity check
    before spending a Brain simulation slot.

    Args:
        signal:       DataFrame of shape (dates × stocks) — your cross-sectional signal
        returns:      DataFrame of shape (dates × stocks) — daily stock returns
                      (must share the same columns/index as signal)
        forward_days: how many days ahead to test predictability (default 1)

    Returns:
        dict with keys:
            mean_ic     — average rank correlation between signal and forward return
            ic_std      — standard deviation of daily IC
            ic_tstat    — t-statistic of IC (> 2 is a green light)
            ic_ir       — IC / IC_std (information ratio of the signal itself)
            turnover    — average daily fraction of portfolio that rebalances
            ls_sharpe   — annualised Sharpe of a long-top-decile / short-bottom-decile strategy
            verdict     — "SUBMIT" if ic_tstat > 2 and ls_sharpe > 0.5, else "SKIP"

    Interpretation:
        ic_tstat > 2  → signal is statistically significant
        ls_sharpe > 0.5 → long/short strategy is directionally profitable
        verdict == "SUBMIT" → worth sending to Brain
    """
    # Align
    common_cols = signal.columns.intersection(returns.columns)
    common_idx = signal.index.intersection(returns.index)
    sig = signal.loc[common_idx, common_cols]
    fwd = returns.loc[common_idx, common_cols].shift(-forward_days)

    # Daily IC (rank correlation)
    ics = []
    for date in sig.index[:-forward_days]:
        s_row = sig.loc[date].dropna()
        r_row = fwd.loc[date].dropna()
        common = s_row.index.intersection(r_row.index)
        if len(common) < 20:
            continue
        ic = s_row[common].rank().corr(r_row[common].rank(), method="spearman")
        ics.append(ic)

    ics = np.array(ics)
    mean_ic = float(np.nanmean(ics))
    ic_std = float(np.nanstd(ics))
    ic_tstat = (mean_ic / (ic_std / math.sqrt(len(ics)))) if ic_std > 0 else 0.0
    ic_ir = (mean_ic / ic_std) if ic_std > 0 else 0.0

    # Turnover (fraction of top-decile that changes day over day)
    ranks = sig.rank(axis=1, pct=True)
    long_mask = ranks > 0.9
    turnovers = []
    for i in range(1, len(ranks)):
        prev = long_mask.iloc[i - 1]
        curr = long_mask.iloc[i]
        # stocks that entered or exited the long basket
        changed = (prev != curr).sum()
        total = curr.sum() + prev.sum()
        if total > 0:
            turnovers.append(changed / total)
    turnover = float(np.nanmean(turnovers)) if turnovers else float("nan")

    # Long/short daily returns
    ls_returns = []
    for date in sig.index[:-forward_days]:
        r = fwd.loc[date].dropna()
        s = sig.loc[date].dropna()
        common = r.index.intersection(s.index)
        if len(common) < 20:
            continue
        rank_pct = s[common].rank(pct=True)
        long_ret = r[common][rank_pct > 0.9].mean()
        short_ret = r[common][rank_pct < 0.1].mean()
        ls_returns.append(long_ret - short_ret)

    ls_returns = np.array(ls_returns)
    ls_sharpe = float(np.nanmean(ls_returns) / np.nanstd(ls_returns) * math.sqrt(252)) if np.nanstd(ls_returns) > 0 else 0.0

    verdict = "SUBMIT" if (ic_tstat > 2 and ls_sharpe > 0.5) else "SKIP"

    return {
        "mean_ic": round(mean_ic, 4),
        "ic_std": round(ic_std, 4),
        "ic_tstat": round(ic_tstat, 2),
        "ic_ir": round(ic_ir, 3),
        "turnover": round(turnover, 3),
        "ls_sharpe": round(ls_sharpe, 2),
        "verdict": verdict,
    }
