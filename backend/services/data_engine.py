"""
Data Engine — Universal Table Reader & Schema Extractor

Provides `process_dataframe()` which reads any CSV or Excel file,
dynamically infers column types and structure, and returns a
summary JSON payload suitable for the frontend Schema panel.

Design principles:
- Zero hard-coded column names — fully data-driven.
- Aggressive encoding fallback for real-world messy files.
- Graceful degradation: problematic columns are flagged, never fatal.
学生更改部分：增强数据鲁棒性，因为大模型最开始把行列的名字给固定死了
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def process_dataframe(file_path: str) -> dict[str, Any]:
    """Read a CSV / Excel file and return a structural summary.

    Args:
        file_path: Absolute path to the uploaded file.

    Returns:
        A dict with keys:
        - filename (str)
        - total_rows (int)
        - total_columns (int)
        - columns (list[dict]): per-column metadata (name, dtype, missing_rate, …)
        - sample_data (list[dict]): first 3 rows as list of key-value pairs.

    Raises:
        ValueError: If the file format is unsupported or the content is
                    unparseable even after fallback attempts.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    suffix = path.suffix.lower()
    if suffix in (".xls", ".xlsx"):
        df = _read_excel(path)
    elif suffix == ".csv":
        df = _read_csv(path)
    else:
        raise ValueError(f"不支持的文件格式: '{suffix}'。请上传 CSV 或 Excel 文件。")

    if df.empty:
        raise ValueError("文件内容为空（0 行数据），请检查文件。")

    # Basic cleanup ----------------------------------------------------------
    df = _sanitise_dataframe(df)

    total_rows, total_columns = df.shape

    # Per-column metadata ----------------------------------------------------
    columns_meta: list[dict[str, Any]] = []
    for col in df.columns:
        col_str = str(col)
        series = df[col]
        missing_count = int(series.isna().sum())
        missing_rate = round(missing_count / total_rows, 4) if total_rows else 0.0
        dtype_name = _infer_dtype(series)

        columns_meta.append({
            "name": col_str,
            "dtype": dtype_name,
            "missing_count": missing_count,
            "missing_rate": missing_rate,
            "unique_values": int(series.nunique(dropna=True)),
            # a few descriptive stats for numeric columns
            "stats": _column_stats(series, dtype_name),
        })

    # Sample rows (first 3) --------------------------------------------------
    sample_data: list[dict[str, Any]] = []
    for _, row in df.head(3).iterrows():
        sample_data.append({
            str(k): _safe_json_value(v) for k, v in row.items()
        })

    return {
        "filename": path.name,
        "total_rows": total_rows,
        "total_columns": total_columns,
        "columns": columns_meta,
        "sample_data": sample_data,
    }


# ---------------------------------------------------------------------------
# Internal helpers — file I/O
# ---------------------------------------------------------------------------

# Ordered list of encodings to try when reading CSV files.
_CSV_ENCODINGS = ["utf-8", "utf-8-sig", "gbk", "gb18030", "latin1", "iso-8859-1"]


def _read_csv(path: Path) -> pd.DataFrame:
    """Read CSV with automatic encoding fallback.

    Tries each encoding in *_CSV_ENCODINGS*; the first one that does not
    raise a UnicodeDecodeError wins.  If all fail the last exception is
    re-raised as a ValueError.
    """
    last_err: Exception | None = None
    for enc in _CSV_ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=enc)
            logger.debug("CSV read successfully with encoding=%s", enc)
            return df
        except (UnicodeDecodeError, UnicodeError) as exc:
            last_err = exc
            continue
        except Exception as exc:
            # Other errors (e.g. parser errors) should surface immediately
            raise ValueError(f"CSV 文件解析失败: {exc}") from exc

    raise ValueError(
        f"无法识别文件编码，已尝试 {_CSV_ENCODINGS}。"
        f"最后错误: {last_err}"
    )


def _read_excel(path: Path) -> pd.DataFrame:
    """Read Excel (.xls / .xlsx), falling back to the older ``xlrd`` engine
    if openpyxl fails (common with legacy .xls files)."""
    try:
        return pd.read_excel(path, engine="openpyxl")
    except Exception:
        logger.debug("openpyxl failed, trying xlrd for %s", path.name)
        try:
            return pd.read_excel(path, engine="xlrd")
        except Exception as exc:
            raise ValueError(f"Excel 文件解析失败: {exc}") from exc


# ---------------------------------------------------------------------------
# Internal helpers — data quality
# ---------------------------------------------------------------------------


def _sanitise_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a freshly-loaded DataFrame in-place.

    Operations:
    - Strip whitespace from column names.
    - Drop fully-empty rows and columns.
    - Convert column names to string.
    """
    # Drop completely empty rows & columns
    df = df.dropna(how="all").dropna(axis=1, how="all")

    # Normalise column names
    df.columns = [str(c).strip() for c in df.columns]

    # Replace the literal string "NaN"/"nan"/"N/A" with actual NaN
    df = df.replace(["NaN", "nan", "N/A", "NA", "null", "NULL", "None"], np.nan)

    return df


# ---------------------------------------------------------------------------
# Internal helpers — type inference & stats
# ---------------------------------------------------------------------------


def _infer_dtype(series: pd.Series) -> str:
    """Heuristic column-type classifier.

    Returns one of:
        ``"numeric"``, ``"categorical"``, ``"datetime"``, ``"text"``,
        ``"boolean"``, ``"unknown"``.
    """
    # Drop NaN for cleaner inference
    s = series.dropna()
    if len(s) == 0:
        return "unknown"

    # Boolean (including 0/1 binary)
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if s.nunique() <= 2 and set(s.unique()).issubset({0, 1, True, False, "True", "False", "true", "false", 0.0, 1.0}):
        return "boolean"

    # Numeric
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"

    # Datetime
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    # Try to coerce to datetime for object columns (suppress dateutil noise)
    if s.dtype == object:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pd.to_datetime(s, errors="raise")
            return "datetime"
        except (ValueError, TypeError):
            pass

    # String / object
    if s.dtype == object:
        nunique = s.nunique()
        total = len(s)
        # If cardinality is low relative to sample → categorical
        if nunique <= 50 or (nunique / total) < 0.2:
            return "categorical"
        return "text"

    return "unknown"


def _column_stats(series: pd.Series, dtype_name: str) -> dict[str, Any]:
    """Compute lightweight descriptive stats for a column.

    For numeric columns we return min / max / mean / std;
    for others we return the top-5 most frequent values.
    """
    stats: dict[str, Any] = {}

    if dtype_name == "numeric":
        s = series.dropna()
        if len(s) > 0:
            stats["min"] = _safe_json_value(s.min())
            stats["max"] = _safe_json_value(s.max())
            stats["mean"] = round(float(s.mean()), 4)
            stats["std"] = round(float(s.std()), 4)
    elif dtype_name in ("categorical", "boolean"):
        counts = series.dropna().value_counts().head(5)
        stats["top_values"] = {
            str(k): int(v) for k, v in counts.items()
        }

    return stats


# ---------------------------------------------------------------------------
# Advanced Analysis (Tab 2 — automated EDA)
# ---------------------------------------------------------------------------

# Maximum rows used for heavy computations (correlation, histograms).
_MAX_ANALYSIS_SAMPLE = 50_000
# Maximum numeric columns in the correlation matrix (avoid huge heatmaps).
_MAX_CORR_COLUMNS = 10


def get_advanced_analysis(file_path: str) -> dict[str, Any]:
    """Run automated exploratory data analysis on the uploaded file.

    Computes four categories of analysis and returns chart-ready JSON:

    * **exploratory** — per-column stats (mean, std, quartiles, skewness, kurtosis)
    * **histograms** — 30-bin histograms for the first 4 numeric columns
    * **correlation** — correlation matrix (sampled, capped at 10 columns)
    * **categorical** — top-5 distributions for the first 4 categorical columns
    * **timeseries** — monthly aggregation of the first numeric metric by the
      first datetime column (if both exist)

    Args:
        file_path: Absolute path to the uploaded file.

    Returns:
        A dict with keys ``exploratory``, ``histograms``, ``correlation``,
        ``categorical``, and ``timeseries`` (the latter may be ``null``).
    """
    path = Path(file_path)
    # Prefer CSV cache for speed
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        df = _read_csv(csv_path)
    else:
        suffix = path.suffix.lower()
        if suffix in (".xls", ".xlsx"):
            df = _read_excel(path)
        else:
            df = _read_csv(path)

    df = _sanitise_dataframe(df)

    n_total = len(df)
    sample_n = min(n_total, _MAX_ANALYSIS_SAMPLE)

    # Work on a sampled copy for heavy computation (corr, histograms, stats)
    df_sample = df.sample(sample_n, random_state=42) if n_total > sample_n else df.copy()

    # --- classify columns --------------------------------------------------
    numeric_cols: list[str] = []
    datetime_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in df.columns:
        try:
            dt = _infer_dtype(df[col])
        except Exception:
            continue
        if dt == "numeric":
            numeric_cols.append(str(col))
        elif dt == "datetime":
            datetime_cols.append(str(col))
        elif dt == "categorical":
            categorical_cols.append(str(col))

    logger.info(
        "Advanced analysis: %d rows → sample %d | numeric=%d datetime=%d cat=%d",
        n_total, sample_n, len(numeric_cols), len(datetime_cols), len(categorical_cols),
    )

    # ---- 1. Exploratory stats (all numeric cols) -------------------------
    exploratory: list[dict[str, Any]] = []
    if numeric_cols and len(df_sample) >= 2:
        desc = df_sample[numeric_cols].describe(percentiles=[0.25, 0.5, 0.75])
        skew_vals = df_sample[numeric_cols].skew()
        kurt_vals = df_sample[numeric_cols].kurtosis()
        for col in numeric_cols:
            exploratory.append({
                "column": col,
                "count": int(desc.loc["count", col]) if "count" in desc.index else len(df_sample),
                "mean": round(float(desc.loc["mean", col]), 4),
                "std": round(float(desc.loc["std", col]), 4),
                "min": round(float(desc.loc["min", col]), 4),
                "q25": round(float(desc.loc["25%", col]), 4),
                "q50": round(float(desc.loc["50%", col]), 4),
                "q75": round(float(desc.loc["75%", col]), 4),
                "max": round(float(desc.loc["max", col]), 4),
                "skewness": round(float(skew_vals[col]), 4),
                "kurtosis": round(float(kurt_vals[col]), 4),
            })

    # ---- 2. Histograms (first 4 numeric cols, 30 bins) -------------------
    histograms: dict[str, dict[str, list[float]]] = {}
    for col in numeric_cols[:4]:
        s = df_sample[col].dropna()
        if len(s) < 5:
            continue
        counts, bins = np.histogram(s, bins=30)
        histograms[col] = {
            "bins": [round(float(b), 4) for b in bins[:-1]],
            "counts": [int(c) for c in counts],
        }

    # ---- 3. Correlation matrix (capped columns, use sample) --------------
    corr_columns: list[str] = numeric_cols[:_MAX_CORR_COLUMNS]
    if len(corr_columns) >= 2:
        corr_df = df_sample[corr_columns].corr()
        correlation = {
            "columns": corr_columns,
            "matrix": [[round(float(v), 4) for v in row] for row in corr_df.values.tolist()],
        }
    else:
        correlation = {"columns": [], "matrix": []}

    # ---- 4. Categorical distributions (first 4 cols, top 5 values) -------
    categorical: list[dict[str, Any]] = []
    for col in categorical_cols[:4]:
        vc = df[col].value_counts().head(5)
        if len(vc) == 0:
            continue
        total = int(vc.sum())
        categorical.append({
            "column": col,
            "data": [
                {"name": str(k), "value": int(v), "pct": round(float(v) / total * 100, 1)}
                for k, v in vc.items()
            ],
        })

    # ---- 5. Time-series (first datetime × first numeric) -----------------
    timeseries: dict[str, Any] | None = None
    if datetime_cols and numeric_cols:
        date_col = datetime_cols[0]
        metric_col = numeric_cols[0]
        try:
            ts_df = df[[date_col, metric_col]].dropna().copy()
            ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors="coerce")
            ts_df = ts_df.dropna(subset=[date_col])
            if len(ts_df) > 10:
                ts_df = ts_df.set_index(date_col)
                freq = "M"
                if len(ts_df) < 30:
                    freq = "D"
                elif len(ts_df) > 100_000:
                    freq = "QE"
                agg = ts_df.resample(freq)[metric_col].sum().reset_index()
                timeseries = {
                    "date_col": date_col,
                    "metric_col": metric_col,
                    "data": [
                        {"date": str(r[date_col])[:10], "value": round(float(r[metric_col]), 2)}
                        for _, r in agg.iterrows()
                    ],
                }
        except Exception:
            logger.warning("Time-series aggregation failed", exc_info=True)

    return {
        "exploratory": exploratory,
        "histograms": histograms,
        "correlation": correlation,
        "categorical": categorical,
        "timeseries": timeseries,
    }


def _safe_json_value(value: Any) -> Any:
    """Convert numpy / pandas scalar to a plain Python type for JSON
    serialisation."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value
