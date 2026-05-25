"""
Data Cleaner — executes user-confirmed cleaning operations on uploaded data.

Each operation type is a discrete function.  ``apply_cleaning()`` takes a
list of operation dicts, applies them sequentially, saves the cleaned
result as a new CSV file, and returns a summary.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from datetime import datetime

import numpy as np
import pandas as pd

from .data_engine import _sanitise_dataframe, _read_csv, _read_excel

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"

# ---------------------------------------------------------------------------
# Individual cleaning operations
# ---------------------------------------------------------------------------


def _drop_columns(df: pd.DataFrame, columns: list[str], logs: list[str]) -> pd.DataFrame:
    existing = [c for c in columns if c in df.columns]
    if existing:
        df = df.drop(columns=existing)
        logs.append(f"删除 {len(existing)} 列: {existing}")
    return df


def _fill_missing(df: pd.DataFrame, column: str, method: str,
                  fill_value: Any = None, logs: list[str] | None = None) -> pd.DataFrame:
    """Fill missing values in a single column."""
    if column not in df.columns:
        if logs is not None:
            logs.append(f"跳过: 列 '{column}' 不存在")
        return df
    if logs is None:
        logs = []

    before = int(df[column].isna().sum())
    if before == 0:
        return df

    if method == "mean" and pd.api.types.is_numeric_dtype(df[column]):
        df[column] = df[column].fillna(df[column].mean())
    elif method == "median" and pd.api.types.is_numeric_dtype(df[column]):
        df[column] = df[column].fillna(df[column].median())
    elif method == "mode":
        mode_val = df[column].mode()
        if len(mode_val) > 0:
            df[column] = df[column].fillna(mode_val.iloc[0])
    elif method == "forward":
        df[column] = df[column].ffill()
    elif method == "backward":
        df[column] = df[column].bfill()
    elif method == "custom" and fill_value is not None:
        df[column] = df[column].fillna(fill_value)
    else:
        # Fallback: if numeric use median, else mode
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].median())
        else:
            mode_val = df[column].mode()
            if len(mode_val) > 0:
                df[column] = df[column].fillna(mode_val.iloc[0])

    after = int(df[column].isna().sum())
    logs.append(f"填充 '{column}': {before} -> {after} NaN (方法: {method})")
    return df


def _handle_outliers(df: pd.DataFrame, column: str, method: str,
                     logs: list[str]) -> pd.DataFrame:
    """Cap or remove outliers in a numeric column."""
    if column not in df.columns:
        return df

    s = df[column].dropna()
    if len(s) < 5:
        return df

    q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    if method == "cap":
        before = int(((df[column] < lower) | (df[column] > upper)).sum())
        df[column] = df[column].clip(lower=lower, upper=upper)
        logs.append(f"截断 '{column}' 异常值: {before} 个值被限制到 [{lower:.2f}, {upper:.2f}]")
    elif method == "remove":
        before = len(df)
        df = df[(df[column] >= lower) & (df[column] <= upper)]
        removed = before - len(df)
        logs.append(f"剔除 '{column}' 异常值: {removed} 行被删除")
    # "keep" method: no-op

    return df


def _drop_duplicates_op(df: pd.DataFrame, keep: str,
                        logs: list[str]) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(keep=keep)
    removed = before - len(df)
    logs.append(f"删除重复行: {removed} 行 (保留: {keep})")
    return df


def _drop_constant_columns(df: pd.DataFrame, columns: list[str],
                           logs: list[str]) -> pd.DataFrame:
    existing = [c for c in columns if c in df.columns]
    constant = [c for c in existing if df[c].nunique(dropna=True) <= 1]
    if constant:
        df = df.drop(columns=constant)
        logs.append(f"删除 {len(constant)} 个常量列: {constant}")
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def apply_cleaning(filename: str, operations: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply a sequence of cleaning operations and save the result.

    Args:
        filename: The uploaded file to clean (relative to UPLOAD_DIR).
        operations: List of operation dicts, each with a ``type`` key:
            - {"type": "drop_columns", "columns": [str, ...]}
            - {"type": "fill_missing", "column": str, "method": str, "fill_value": Any}
            - {"type": "handle_outliers", "column": str, "method": "cap|remove|keep"}
            - {"type": "drop_duplicates", "keep": "first"|"last"}
            - {"type": "drop_constant_columns", "columns": [str, ...]}

    Returns:
        {"new_filename": str, "summary": {...}, "schema": {...}}
    """
    path = UPLOAD_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"文件 '{filename}' 不存在")

    # Load
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        df = _read_csv(csv_path)
    elif path.suffix.lower() in (".xls", ".xlsx"):
        df = _read_excel(path)
    else:
        df = _read_csv(path)

    df = _sanitise_dataframe(df)
    rows_before, cols_before = df.shape
    logs: list[str] = []
    total_filled = 0
    total_removed = 0

    for op in operations:
        op_type = op.get("type", "")

        if op_type == "drop_columns":
            df = _drop_columns(df, op.get("columns", []), logs)
        elif op_type == "fill_missing":
            df = _fill_missing(df, op.get("column", ""),
                               op.get("method", "median"),
                               op.get("fill_value"), logs)
            total_filled += 1
        elif op_type == "handle_outliers":
            before_rows = len(df)
            df = _handle_outliers(df, op.get("column", ""),
                                  op.get("method", "keep"), logs)
            total_removed += before_rows - len(df)
        elif op_type == "drop_duplicates":
            before_rows = len(df)
            df = _drop_duplicates_op(df, op.get("keep", "first"), logs)
            total_removed += before_rows - len(df)
        elif op_type == "drop_constant_columns":
            df = _drop_constant_columns(df, op.get("columns", []), logs)
        else:
            logger.warning("Unknown cleaning operation: %s", op_type)

    rows_after, cols_after = df.shape

    # Save cleaned file
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(filename).stem
    new_filename = f"cleaned_{stem}_{ts}.csv"
    new_path = UPLOAD_DIR / new_filename
    df.to_csv(new_path, index=False)
    logger.info("Cleaned data saved: %s (%d rows x %d cols)", new_filename,
                rows_after, cols_after)

    # Generate schema for cleaned data
    from .data_engine import process_dataframe
    schema = process_dataframe(str(new_path))

    return {
        "new_filename": new_filename,
        "summary": {
            "rows_before": rows_before,
            "rows_after": rows_after,
            "cols_before": cols_before,
            "cols_after": cols_after,
            "operations_applied": len(operations),
            "rows_removed": rows_before - rows_after,
            "values_filled": total_filled,
            "log": logs,
        },
        "schema": schema,
    }
