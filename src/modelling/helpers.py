"""
modelling/helpers.py
--------------------
Shared utilities for the modelling sub-package:
formatting helpers and custom metrics.
"""

import numpy as np


# -- Pretty-print separators ------------------------------------------------
def section(title: str) -> None:
    """Print a major section header."""
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def subsection(title: str) -> None:
    """Print a minor sub-section header."""
    print(f"\n-- {title} " + "-" * max(0, 55 - len(title)))


# -- Custom metrics ---------------------------------------------------------
def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error, safe for zero targets."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)