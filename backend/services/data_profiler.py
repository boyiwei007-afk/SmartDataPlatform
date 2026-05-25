"""
Data Profiler — comprehensive data profiling for the "data overview" stage.

Returns column-by-column profiles including distribution statistics,
sparkline bins, outlier counts, and overall data quality metrics.
Reuses ``_infer_dtype`` and ``_safe_json_value`` from data_engine.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_engine import _infer_dtype, _sanitise_dataframe, _read_csv, _read_excel

logger = logging.getLogger(__name__)

_SPARKLINE_BINS = 20
_MAX_SAMPLE = 50_000


def _safe(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        if np.isnan(v) or np.isinf(v):
            return None
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if pd.isna(v):
        return None
    return v


def profile_data(file_path: str) -> dict[str, Any]:
    """Generate a comprehensive data profile for a previously-uploaded file.

    Returns:
        {
            "rows": int, "columns": int,
            "completeness": float,        # overall non-null rate (0-1)
            "duplicate_rows": int,
            "memory_mb": float,
            "column_profiles": [...],
            "quality_issues": {...}        # summary of detected issues
        }
    """
    path = Path(file_path)
    csv_path = path.with_suffix(".csv")

    if csv_path.exists():
        df = _read_csv(csv_path)
    elif path.suffix.lower() in (".xls", ".xlsx"):
        df = _read_excel(path)
    else:
        df = _read_csv(path)

    df = _sanitise_dataframe(df)
    n_rows, n_cols = df.shape

    # File size
    target = csv_path if csv_path.exists() else path
    memory_mb = round(os.path.getsize(target) / (1024 * 1024), 2)

    # Overall completeness
    total_cells = n_rows * n_cols
    non_null = int(df.notna().sum().sum())
    completeness = round(non_null / total_cells, 4) if total_cells > 0 else 0.0

    # Duplicates
    dup_rows = int(df.duplicated().sum())

    # Column profiles
    column_profiles = []
    quality_issues = {
        "high_missing": [],       # >30% missing
        "constant_columns": [],   # nunique <= 1
        "potential_outliers": [],  # columns with outlier_count > 0
        "type_suggestions": [],   # columns where dtype might be wrong
    }

    for col in df.columns:
        col_str = str(col)
        series = df[col]
        dtype = _infer_dtype(series)

        missing_count = int(series.isna().sum())
        missing_pct = round(missing_count / n_rows, 4) if n_rows > 0 else 0.0
        unique_count = int(series.nunique(dropna=True))

        profile = {
            "name": col_str,
            "dtype": dtype,
            "missing_pct": missing_pct,
            "missing_count": missing_count,
            "unique_count": unique_count,
        }

        # Numeric columns: distribution stats + sparkline + outlier detection
        if dtype == "numeric":
            s = series.dropna()
            if len(s) >= 5:
                q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outliers = s[(s < lower) | (s > upper)]
                outlier_count = int(len(outliers))

                profile["distribution"] = {
                    "min": _safe(s.min()),
                    "q25": round(q1, 4),
                    "median": round(float(s.quantile(0.50)), 4),
                    "q75": round(q3, 4),
                    "max": _safe(s.max()),
                    "mean": round(float(s.mean()), 4),
                    "std": round(float(s.std()), 4),
                }
                profile["outlier_count"] = outlier_count
                if outlier_count > 0:
                    quality_issues["potential_outliers"].append({
                        "column": col_str,
                        "count": outlier_count,
                        "pct": round(outlier_count / len(s) * 100, 2),
                    })

                # Sparkline bins (20 bins)
                try:
                    counts_sp, bins_sp = np.histogram(s.values, bins=_SPARKLINE_BINS)
                    profile["sparkline"] = [int(c) for c in counts_sp]
                except Exception:
                    profile["sparkline"] = []

        # Categorical / text columns: top values
        if dtype in ("categorical", "text", "boolean"):
            try:
                top = series.dropna().value_counts().head(5)
                profile["top_values"] = [
                    {"name": str(k), "count": int(v)}
                    for k, v in top.items()
                ]
            except Exception:
                profile["top_values"] = []

        # High missing flag
        if missing_pct > 0.3:
            quality_issues["high_missing"].append({
                "column": col_str,
                "pct": round(missing_pct * 100, 1),
            })

        # Constant column
        if unique_count <= 1:
            quality_issues["constant_columns"].append(col_str)

        # Type suggestion (e.g. date stored as text)
        if dtype == "text" and unique_count > 3:
            try:
                converted = pd.to_datetime(series.dropna().head(20), errors="raise")
                if len(converted) >= 5:
                    quality_issues["type_suggestions"].append({
                        "column": col_str,
                        "current": "text",
                        "suggested": "datetime",
                    })
            except (ValueError, TypeError):
                pass

        column_profiles.append(profile)

    return {
        "rows": n_rows,
        "columns": n_cols,
        "completeness": completeness,
        "duplicate_rows": dup_rows,
        "memory_mb": memory_mb,
        "column_profiles": column_profiles,
        "quality_issues": quality_issues,
    }
