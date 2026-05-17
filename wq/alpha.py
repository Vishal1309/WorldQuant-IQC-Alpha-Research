"""
Alpha expression model and fitness utilities.

An AlphaConfig describes a complete alpha: the expression string plus all the
simulation settings (universe, neutralization, decay, etc.). It also knows how
to serialize itself into the payload shape the Brain API expects.

Usage:
    config = AlphaConfig(
        expression="ts_rank(close, 20)",
        tags=["momentum", "price-volume"],
        notes="Simple price momentum — baseline",
    )
    payload = config.to_payload()
    fitness = compute_fitness(sharpe=1.4, returns=0.12, turnover=0.45)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# AlphaConfig
# ---------------------------------------------------------------------------

@dataclass
class AlphaConfig:
    """
    A complete alpha specification ready for Brain simulation.

    expression:     Brain operator expression string
    region:         "USA" | "CHN" | "EUR" | "JPN" (default "USA")
    universe:       "TOP3000" | "TOP2000" | "TOP1000" | "TOP500" (default "TOP3000")
    neutralization: "Market" | "Industry" | "Subindustry" | "None" (default "Subindustry")
    delay:          0 or 1 — use 1 (daily delay) to start; 0 is intraday
    decay:          0–33 — signal half-life smoothing; 0 = no decay
    truncation:     0.01–0.10 — max weight per stock; 0.08 is a safe default
    pasteurization: "On" | "Off" — removes outlier stocks (keep "On")
    tags:           list of strings for your own categorisation
    notes:          free-text explanation of the alpha thesis
    """
    expression: str
    region: str = "USA"
    universe: str = "TOP3000"
    neutralization: str = "Subindustry"
    delay: int = 1
    decay: int = 0
    truncation: float = 0.08
    pasteurization: str = "On"
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    def to_payload(self) -> dict:
        """Serialize to the JSON payload shape the Brain /simulations endpoint expects."""
        return {
            "type": "REGULAR",
            "settings": {
                "instrumentType": "EQUITY",
                "region": self.region.upper(),
                "universe": self.universe.upper(),
                "delay": self.delay,
                "decay": self.decay,
                "neutralization": self.neutralization.upper(),
                "truncation": self.truncation,
                "pasteurization": self.pasteurization.upper(),
                "unitHandling": "VERIFY",
                "nanHandling": "ON",
                "language": "FASTEXPR",
                "visualization": False,
            },
            "regular": self.expression,
        }

    def fingerprint(self) -> str:
        """
        Stable SHA256 hash of this alpha — used to detect duplicates before
        re-submitting to Brain (saves simulation quota).
        """
        canonical = json.dumps(self.to_payload(), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def __str__(self) -> str:
        return (
            f"AlphaConfig({self.expression!r}, "
            f"universe={self.universe}, neutralization={self.neutralization}, "
            f"delay={self.delay}, decay={self.decay})"
        )


# ---------------------------------------------------------------------------
# Fitness formula
# ---------------------------------------------------------------------------

def compute_fitness(sharpe: float, returns: float, turnover: float) -> float:
    """
    WorldQuant Brain fitness formula.

    fitness = sqrt(abs(returns) / max(turnover, 0.125)) * sharpe

    Higher fitness = better alpha. A good target is fitness >= 1.0 with
    Sharpe >= 1.25.
    """
    import math
    if sharpe is None or returns is None or turnover is None:
        return float("nan")
    denominator = max(abs(turnover), 0.125)
    return math.sqrt(abs(returns) / denominator) * sharpe


# ---------------------------------------------------------------------------
# Submittability check
# ---------------------------------------------------------------------------

def is_submittable(result: dict) -> bool:
    """
    Return True if a Brain simulation result meets the minimum bar for
    submission to the IQC leaderboard.

    Criteria:
    - Simulation completed without error
    - In-sample Sharpe >= 1.25
    - Fitness > 0 (positive returns direction)
    """
    if result.get("status", "").upper() not in ("COMPLETE", "WARNING"):
        return False

    is_metrics = result.get("is", {})
    sharpe = is_metrics.get("sharpe")
    fitness = is_metrics.get("fitness")

    if sharpe is None or fitness is None:
        return False

    return sharpe >= 1.25 and fitness > 0


# ---------------------------------------------------------------------------
# Alpha template helpers
# ---------------------------------------------------------------------------

# Momentum templates — parameterised by {field} and {window}
MOMENTUM_TEMPLATES = [
    "ts_rank({field}, {window})",
    "ts_zscore({field}, {window})",
    "-1 * ts_corr(rank({field}), rank(volume), {window})",
    "ts_rank({field}, {window}) - ts_rank({field}, {long_window})",
]

# Mean-reversion templates
REVERSION_TEMPLATES = [
    "-1 * ts_zscore({field}, {window})",
    "ts_mean({field}, {window}) - {field}",
    "-1 * ts_rank({field}, {window})",
    "ts_mean({field}, {long_window}) - ts_mean({field}, {window})",
]

# Fundamental / cross-sectional templates
FUNDAMENTAL_TEMPLATES = [
    "group_rank({field}, industry)",
    "group_rank({field}, subindustry)",
    "rank({field})",
    "ts_delta({field}, {window})",
    "group_rank(ts_delta({field}, {window}), subindustry)",
]


def expand_templates(templates: list[str], field: str, window: int = 20, long_window: int = 60) -> list[str]:
    """
    Instantiate a list of template strings for a given field and window.

    Returns only templates where all placeholders can be filled.
    """
    results = []
    for tmpl in templates:
        try:
            expr = tmpl.format(field=field, window=window, long_window=long_window)
            results.append(expr)
        except KeyError:
            pass
    return results


def momentum_alphas(
    field: str,
    windows: list[int] = None,
    long_window: int = 60,
    **kwargs,
) -> list["AlphaConfig"]:
    """
    Generate a list of momentum AlphaConfig objects for a given field.

    Args:
        field:       Brain data field id (e.g. "close", "volume")
        windows:     list of lookback windows to test (default [5, 10, 20])
        long_window: long window for dual-window templates (default 60)
        **kwargs:    passed through to AlphaConfig (universe, neutralization, etc.)
    """
    if windows is None:
        windows = [5, 10, 20]
    configs = []
    for w in windows:
        for expr in expand_templates(MOMENTUM_TEMPLATES, field, window=w, long_window=long_window):
            configs.append(AlphaConfig(expression=expr, tags=["momentum", field], **kwargs))
    return configs


def reversion_alphas(
    field: str,
    windows: list[int] = None,
    long_window: int = 60,
    **kwargs,
) -> list["AlphaConfig"]:
    """Generate mean-reversion AlphaConfig objects for a given field."""
    if windows is None:
        windows = [3, 5, 10]
    configs = []
    for w in windows:
        for expr in expand_templates(REVERSION_TEMPLATES, field, window=w, long_window=long_window):
            configs.append(AlphaConfig(expression=expr, tags=["reversion", field], **kwargs))
    return configs


def fundamental_alphas(field: str, windows: list[int] = None, **kwargs) -> list["AlphaConfig"]:
    """Generate cross-sectional fundamental AlphaConfig objects for a given field."""
    if windows is None:
        windows = [4, 8]
    configs = []
    for w in windows:
        for expr in expand_templates(FUNDAMENTAL_TEMPLATES, field, window=w):
            configs.append(AlphaConfig(expression=expr, tags=["fundamental", field], **kwargs))
    return configs
