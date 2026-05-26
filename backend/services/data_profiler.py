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
        "high_missing": [],        # >30% missing
        "constant_columns": [],    # nunique <= 1
        "low_variance": [],        # nunique/total < 0.1% (near-constant)
        "potential_outliers": [],  # columns with outlier_count > 0
        "type_suggestions": [],    # columns where dtype might be wrong
        "missing_patterns": [],    # column missingness correlated with another column
        "duplicate_columns": [],   # numeric column pairs with r > 0.999
    }

    for col in df.columns:
        col_str = str(col)
        series = df[col]
        dtype = _infer_dtype(series, col_str)

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
        if dtype in ("categorical", "text", "boolean", "identifier"):
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

        # Type suggestion: text that looks like boolean
        if dtype == "text" and 1 < unique_count <= 4:
            s_lower = series.dropna().astype(str).str.lower()
            boolean_sets = [
                {"是", "否"}, {"yes", "no"}, {"y", "n"},
                {"true", "false"}, {"t", "f"}, {"1", "0"},
            ]
            vals = set(s_lower.unique())
            if any(vals.issubset(bs) for bs in boolean_sets):
                quality_issues["type_suggestions"].append({
                    "column": col_str,
                    "current": "text",
                    "suggested": "boolean",
                })

        # Type suggestion: low-cardinality numeric → categorical
        if dtype == "numeric" and 2 <= unique_count <= 5:
            quality_issues["type_suggestions"].append({
                "column": col_str,
                "current": "numeric",
                "suggested": "categorical",
            })

        # Low-variance column (near-constant)
        if n_rows > 100 and unique_count > 1 and (unique_count / n_rows) < 0.001:
            quality_issues["low_variance"].append({
                "column": col_str,
                "pct": round(unique_count / n_rows * 100, 3),
            })

        column_profiles.append(profile)

    # ---- Post-loop: missingness patterns ------------------------------------
    if n_rows >= 20:
        miss_matrix = df.isna()
        miss_cols = [c for c in df.columns if 0 < miss_matrix[str(c)].mean() < 0.95]
        if len(miss_cols) >= 2:
            seen_pairs: set[tuple[str, str]] = set()
            for i, col_a in enumerate(miss_cols[:15]):
                miss_a = miss_matrix[str(col_a)]
                for col_b in miss_cols[i + 1:]:
                    key = (str(col_a), str(col_b))
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    miss_b = miss_matrix[str(col_b)]
                    both = (miss_a & miss_b).sum()
                    if both == 0:
                        continue
                    a_given_b = both / miss_b.sum() if miss_b.sum() > 0 else 0
                    b_given_a = both / miss_a.sum() if miss_a.sum() > 0 else 0
                    if a_given_b > 0.8 or b_given_a > 0.8:
                        quality_issues["missing_patterns"].append({
                            "column_a": key[0],
                            "column_b": key[1],
                            "joint_missing": int(both),
                            "a_when_b_missing_rate": round(a_given_b, 3),
                            "b_when_a_missing_rate": round(b_given_a, 3),
                        })

    # ---- Post-loop: duplicate numeric columns (r > 0.999) -------------------
    num_cols = [p["name"] for p in column_profiles if p["dtype"] == "numeric"]
    if len(num_cols) >= 2:
        sample_n = min(n_rows, _MAX_SAMPLE)
        df_num = df[num_cols].head(sample_n).select_dtypes(include=[np.number])
        if df_num.shape[1] >= 2:
            corr = df_num.corr().abs()
            seen: set[tuple[str, str]] = set()
            for i, c1 in enumerate(corr.columns):
                for c2 in corr.columns[i + 1:]:
                    if c1 == c2:
                        continue
                    key = (str(c1), str(c2))
                    if key in seen:
                        continue
                    val = corr.loc[c1, c2]
                    if isinstance(val, (int, float)) and val > 0.999:
                        seen.add(key)
                        quality_issues["duplicate_columns"].append({
                            "column_a": key[0],
                            "column_b": key[1],
                            "correlation": round(float(val), 4),
                        })

    return {
        "rows": n_rows,
        "columns": n_cols,
        "completeness": completeness,
        "duplicate_rows": dup_rows,
        "memory_mb": memory_mb,
        "column_profiles": column_profiles,
        "quality_issues": quality_issues,
    }
