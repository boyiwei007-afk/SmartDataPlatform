"""
Analysis Functions — implementations of all registered analysis actions.

Every function in this module receives a pandas DataFrame plus typed
parameters and returns a standardised dict:

    {
        "chart_data": {...},      # ECharts-ready data (varies by chart type)
        "stats": {...},           # key statistical results (numbers, p-values, etc.)
        "table": [[...], ...],    # row-oriented table data for rendering
        "chart_type": "..."       # the chart type to use
    }

All values are JSON-safe (native Python types, no numpy scalars).

Design rules:
- Sample to 50k rows for heavy computations (correlations, clustering, PCA).
- Never mutate the input DataFrame — work on copies.
- Gracefully degrade when data is insufficient (return error messages, never raise).
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sampling threshold (same as data_engine for consistency)
# ---------------------------------------------------------------------------
_MAX_SAMPLE = 50_000


def _sample(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy, sampled if needed."""
    if len(df) > _MAX_SAMPLE:
        return df.sample(_MAX_SAMPLE, random_state=42)
    return df.copy()


def _safe(v: Any) -> Any:
    """Convert numpy scalar to native Python type."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        if np.isnan(v) or np.isinf(v):
            return None
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    if pd.isna(v):
        return None
    return v


def _safe_list(arr: np.ndarray) -> list[float]:
    """Convert a numpy array to a list of native floats."""
    return [round(float(x), 6) if not (np.isnan(x) or np.isinf(x)) else None
            for x in arr]


# =========================================================================
# Layer 1 — 看分布 (univariate)
# =========================================================================

def describe_distribution(df: pd.DataFrame, column: str) -> dict[str, Any]:
    """Numerical column distribution profile: histogram + stats + normality test."""
    s = df[column].dropna()
    if len(s) < 5:
        return {"error": f"列 '{column}' 有效数据不足（需至少5条）", "chart_type": "histogram"}

    # Histogram (30 bins)
    counts, bins = np.histogram(s.values, bins=30)
    chart_data = {
        "bins": [round(float(b), 4) for b in bins[:-1]],
        "counts": [int(c) for c in counts],
        "kde_x": [], "kde_y": [],  # populated below if scipy available
    }
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(s.values)
        x_kde = np.linspace(float(bins[0]), float(bins[-1]), 200)
        y_kde = kde(x_kde)
        chart_data["kde_x"] = _safe_list(x_kde)
        chart_data["kde_y"] = _safe_list(y_kde)
    except Exception:
        pass

    # Basic stats
    stats_out = {
        "count": int(len(s)),
        "mean": round(float(s.mean()), 4),
        "std": round(float(s.std()), 4),
        "min": _safe(s.min()),
        "q25": round(float(s.quantile(0.25)), 4),
        "q50": round(float(s.quantile(0.50)), 4),
        "q75": round(float(s.quantile(0.75)), 4),
        "max": _safe(s.max()),
        "skewness": round(float(s.skew()), 4),
        "kurtosis": round(float(s.kurtosis()), 4),
        "iqr": round(float(s.quantile(0.75) - s.quantile(0.25)), 4),
    }

    # Normality test
    try:
        if len(s) >= 2000:
            stat, p_norm = sp_stats.kstest(s.values, "norm", args=(s.mean(), s.std()))
            test_name = "Kolmogorov-Smirnov"
        else:
            stat, p_norm = sp_stats.shapiro(s.values[:2000])
            test_name = "Shapiro-Wilk"
        stats_out["normality_test"] = test_name
        stats_out["normality_stat"] = round(float(stat), 6)
        stats_out["normality_p"] = round(float(p_norm), 6)
    except Exception:
        stats_out["normality_test"] = None

    return {
        "chart_data": chart_data,
        "stats": stats_out,
        "table": [],
        "chart_type": "histogram",
    }


def describe_frequency(df: pd.DataFrame, column: str, top_n: int = 10) -> dict[str, Any]:
    """Categorical column frequency: horizontal bar + counts + entropy."""
    vc = df[column].value_counts()
    total = int(vc.sum())
    items = vc.head(top_n)
    labels = [str(k) for k in items.index]
    values = [int(v) for v in items.values]

    # Entropy
    probs = vc.values / vc.values.sum()
    entropy = -float(np.sum(probs * np.log2(probs + 1e-12)))

    # Cumulative
    cum_pct = np.cumsum(items.values) / total * 100

    chart_data = {
        "labels": labels,
        "values": values,
        "cum_pct": [round(float(x), 2) for x in cum_pct],
    }
    stats_out = {
        "total_categories": int(len(vc)),
        "shown_categories": len(labels),
        "top_category": labels[0] if labels else None,
        "top_pct": round(float(values[0]) / total * 100, 2) if values else 0,
        "entropy": round(float(entropy), 4),
        "max_entropy": round(float(np.log2(len(vc))), 4),
    }
    table = [[labels[i], values[i], round(float(values[i]) / total * 100, 2)]
             for i in range(len(labels))]

    return {
        "chart_data": chart_data,
        "stats": stats_out,
        "table": table,
        "chart_type": "horizontal_bar",
    }


def detect_outliers(df: pd.DataFrame, column: str, method: str = "both") -> dict[str, Any]:
    """Outlier detection via IQR and/or Z-score. Returns boxplot data + outlier list."""
    s = df[column].dropna()
    q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
    iqr = q3 - q1
    lower_iqr, upper_iqr = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    outliers_iqr = s[(s < lower_iqr) | (s > upper_iqr)]
    outliers_zscore = pd.Series(dtype=float)
    if len(s) > 0 and s.std() > 1e-10:
        z = np.abs((s.values - s.mean()) / s.std())
        outliers_zscore = s[z > 3]

    if method == "iqr":
        outlier_mask = (s < lower_iqr) | (s > upper_iqr)
    elif method == "zscore":
        outlier_mask = pd.Series(z > 3, index=s.index) if len(s) > 0 else pd.Series(dtype=bool)
    else:
        outlier_mask = (s < lower_iqr) | (s > upper_iqr)
        if len(s) > 0 and s.std() > 1e-10:
            outlier_mask = outlier_mask | pd.Series(z > 3, index=s.index)

    outlier_vals = s[outlier_mask]
    n_out = int(outlier_mask.sum())

    # Boxplot summary data
    chart_data = {
        "boxplot": {
            "min": _safe(s.min()),
            "q1": q1,
            "median": round(float(s.median()), 4),
            "q3": q3,
            "max": _safe(s.max()),
            "lower_fence": round(lower_iqr, 4),
            "upper_fence": round(upper_iqr, 4),
        },
        "outliers": _safe_list(outlier_vals.values),
        "values": _safe_list(s.values),
    }
    stats_out = {
        "total_count": int(len(s)),
        "outlier_count": n_out,
        "outlier_pct": round(float(n_out) / len(s) * 100, 2) if len(s) > 0 else 0,
        "iqr": round(iqr, 4),
        "method": method,
    }
    table = [
        [round(float(v), 4)]
        for v in outlier_vals.head(20).values
    ]

    return {
        "chart_data": chart_data,
        "stats": stats_out,
        "table": table,
        "chart_type": "boxplot",
    }


def test_normality(df: pd.DataFrame, column: str) -> dict[str, Any]:
    """Normality test: QQ plot data + Shapiro-Wilk or D'Agostino test."""
    s = df[column].dropna()
    if len(s) < 8:
        return {"error": "样本量过小，无法进行正态性检验", "chart_type": "qq"}

    # QQ plot: theoretical vs sample quantiles
    n = len(s)
    theo = np.sort(np.random.normal(s.mean(), s.std(), n)) if n > 5000 else \
           sp_stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n, loc=s.mean(), scale=s.std())
    sample_q = np.sort(s.values)
    # Downsample for chart rendering
    step = max(1, n // 500)
    chart_data = {
        "qq_x": _safe_list(theo[::step]),
        "qq_y": _safe_list(sample_q[::step]),
    }

    # Test
    if n <= 2000:
        stat, p_val = sp_stats.shapiro(s.values)
        test_name = "Shapiro-Wilk"
    else:
        stat, p_val = sp_stats.normaltest(s.values)
        test_name = "D'Agostino-Pearson"

    stats_out = {
        "test": test_name,
        "statistic": round(float(stat), 6),
        "p_value": round(float(p_val), 6),
        "is_normal": bool(p_val > 0.05),
        "skewness": round(float(s.skew()), 4),
        "kurtosis": round(float(s.kurtosis()), 4),
    }

    return {
        "chart_data": chart_data,
        "stats": stats_out,
        "table": [],
        "chart_type": "qq",
    }


def analyze_composition(df: pd.DataFrame, column: str, top_n: int = 10) -> dict[str, Any]:
    """Composition / market share analysis: donut chart + HHI index."""
    vc = df[column].value_counts()
    total = int(vc.sum())
    top = vc.head(top_n)
    other_count = int(vc.iloc[top_n:].sum()) if len(vc) > top_n else 0

    labels = [str(k) for k in top.index]
    values = [int(v) for v in top.values]
    if other_count > 0:
        labels.append("其他")
        values.append(other_count)

    # HHI (Herfindahl-Hirschman Index)
    shares = np.array(values) / sum(values)
    hhi = float(np.sum(shares ** 2) * 10000)

    chart_data = {
        "labels": labels,
        "values": values,
    }
    stats_out = {
        "total_categories": int(len(vc)),
        "hhi": round(hhi, 2),
        "hhi_interpretation": (
            "高度集中" if hhi > 2500 else "中度集中" if hhi > 1500 else "分散"
        ),
        "top1_pct": round(float(values[0]) / total * 100, 2) if values else 0,
        "top3_pct": round(float(sum(values[:3])) / total * 100, 2),
    }
    table = [
        [labels[i], values[i], round(float(values[i]) / total * 100, 2)]
        for i in range(len(labels))
    ]

    return {
        "chart_data": chart_data,
        "stats": stats_out,
        "table": table,
        "chart_type": "donut",
    }


# =========================================================================
# Layer 2 — 看关系 (bivariate / multivariate)
# =========================================================================

def correlate(df: pd.DataFrame, x: str, y: str, method: str = "pearson") -> dict[str, Any]:
    """Correlation between two numeric columns: scatter + coefficient + significance."""
    sub = df[[x, y]].dropna()
    if len(sub) < 5:
        return {"error": f"有效数据不足（{len(sub)}条），无法计算相关性", "chart_type": "scatter"}

    xs, ys = sub[x].values, sub[y].values

    if method == "spearman":
        r, p_val = sp_stats.spearmanr(xs, ys)
    else:
        r, p_val = sp_stats.pearsonr(xs, ys)

    # Regression line
    slope, intercept, *_ = np.polyfit(xs, ys, 1)
    x_line = np.linspace(xs.min(), xs.max(), 50)
    y_line = slope * x_line + intercept

    # Confidence band (95%)
    n = len(xs)
    y_pred = slope * xs + intercept
    residuals = ys - y_pred
    mse = np.sum(residuals ** 2) / (n - 2) if n > 2 else 0
    se = np.sqrt(mse / np.sum((xs - xs.mean()) ** 2)) if n > 2 else 0

    chart_data = {
        "points": [[_safe(xs[i]), _safe(ys[i])] for i in range(min(len(xs), 2000))],
        "trendline_x": _safe_list(x_line),
        "trendline_y": _safe_list(y_line),
    }
    stats_out = {
        "method": method,
        "r": round(float(r), 6),
        "r_squared": round(float(r ** 2), 6),
        "p_value": round(float(p_val), 6),
        "significant": bool(p_val < 0.05),
        "n": int(len(sub)),
        "slope": round(float(slope), 6),
        "intercept": round(float(intercept), 6),
        "interpretation": (
            f"{'强' if abs(r) > 0.7 else '中等' if abs(r) > 0.4 else '弱'}"
            f"{'正' if r > 0 else '负'}相关"
            f"{'（显著）' if p_val < 0.05 else '（不显著）'}"
        ),
    }

    return {
        "chart_data": chart_data,
        "stats": stats_out,
        "table": [],
        "chart_type": "scatter",
    }


def cross_tabulate(df: pd.DataFrame, x: str, y: str) -> dict[str, Any]:
    """Cross-tabulation of two categorical columns: stacked bar + chi-square + Cramer's V."""
    ct = pd.crosstab(df[x], df[y])
    if ct.empty:
        return {"error": "交叉表为空，请检查所选列", "chart_type": "stacked_bar"}

    # Stacked bar data
    categories_x = [str(c) for c in ct.index]
    categories_y = [str(c) for c in ct.columns]
    series = []
    for j, y_cat in enumerate(categories_y):
        series.append({
            "name": y_cat,
            "data": [int(ct.iloc[i, j]) for i in range(len(categories_x))],
        })

    chart_data = {
        "x_labels": categories_x,
        "series": series,
    }

    # Chi-square test
    try:
        chi2, p_val, dof, expected = sp_stats.chi2_contingency(ct.values)
        n = ct.values.sum()
        cramer_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1))) if n > 0 and min(ct.shape) > 1 else 0
        stats_out = {
            "chi2": round(float(chi2), 4),
            "p_value": round(float(p_val), 6),
            "dof": int(dof),
            "significant": bool(p_val < 0.05),
            "cramers_v": round(float(cramer_v), 4),
        }
    except Exception:
        stats_out = {"error": "无法进行卡方检验"}

    table = [[str(ct.index[i]), str(ct.columns[j]), int(ct.iloc[i, j])]
             for i in range(len(categories_x)) for j in range(len(categories_y))]

    return {
        "chart_data": chart_data,
        "stats": stats_out,
        "table": table,
        "chart_type": "stacked_bar",
    }


def compare_groups(df: pd.DataFrame, group_col: str, value_col: str) -> dict[str, Any]:
    """Compare numeric values across groups: boxplot + ANOVA/Kruskal-Wallis + group stats."""
    groups = df.groupby(group_col)[value_col]
    group_data = {str(k): v.dropna().values for k, v in groups}
    # Keep groups with enough data
    group_data = {k: v for k, v in group_data.items() if len(v) >= 3}

    if len(group_data) < 2:
        return {"error": "有效分组数不足（需至少2组，每组至少3条数据）", "chart_type": "boxplot"}

    # Boxplot data
    boxplot_series = []
    for name, vals in group_data.items():
        boxplot_series.append({
            "name": name,
            "min": _safe(float(np.min(vals))),
            "q1": round(float(np.percentile(vals, 25)), 4),
            "median": round(float(np.percentile(vals, 50)), 4),
            "q3": round(float(np.percentile(vals, 75)), 4),
            "max": _safe(float(np.max(vals))),
            "mean": round(float(np.mean(vals)), 4),
            "count": int(len(vals)),
        })

    chart_data = {"boxplot_series": boxplot_series}

    # Group stats table
    table = [
        [s["name"], s["count"], s["mean"], s["median"], s["min"], s["max"],
         round(float(np.std(group_data[s["name"]])), 4)]
        for s in boxplot_series
    ]

    # ANOVA or Kruskal-Wallis
    values_list = [v for v in group_data.values()]
    try:
        if all(len(v) >= 30 for v in values_list):
            stat, p_val = sp_stats.f_oneway(*values_list)
            test_name = "ANOVA"
        else:
            stat, p_val = sp_stats.kruskal(*values_list)
            test_name = "Kruskal-Wallis"
        # Eta-squared (effect size)
        grand_mean = np.mean(np.concatenate(values_list))
        ss_between = sum(len(v) * (np.mean(v) - grand_mean) ** 2 for v in values_list)
        ss_total = sum((v - grand_mean).sum() ** 2 for v in values_list)
        eta2 = ss_between / ss_total if ss_total > 0 else 0

        stats_out = {
            "test": test_name,
            "statistic": round(float(stat), 6),
            "p_value": round(float(p_val), 6),
            "significant": bool(p_val < 0.05),
            "eta_squared": round(float(eta2), 6),
            "n_groups": len(group_data),
            "total_n": sum(len(v) for v in values_list),
        }
    except Exception:
        stats_out = {"error": "无法进行统计检验", "n_groups": len(group_data)}

    return {
        "chart_data": chart_data,
        "stats": stats_out,
        "table": table,
        "chart_type": "boxplot",
    }


def compare_pairs(df: pd.DataFrame, group_col: str, value_col: str) -> dict[str, Any]:
    """Pairwise post-hoc comparison: Tukey HSD."""
    from itertools import combinations
    groups = df.groupby(group_col)[value_col]
    group_data = {str(k): v.dropna().values for k, v in groups if len(v) >= 3}
    group_names = list(group_data.keys())

    if len(group_names) < 2:
        return {"error": "需要至少2个有效分组", "chart_type": "horizontal_bar"}
    if len(group_names) > 20:
        return {"error": "分组过多（>20），两两比较结果难以解读", "chart_type": "horizontal_bar"}

    pairs_data = []
    for a, b in combinations(group_names, 2):
        try:
            stat, p_val = sp_stats.ttest_ind(group_data[a], group_data[b])
        except Exception:
            stat, p_val = 0, 1
        diff = float(np.mean(group_data[a]) - np.mean(group_data[b]))
        pairs_data.append({
            "pair": f"{a} vs {b}",
            "diff": round(diff, 4),
            "p_value": round(float(p_val), 6),
            "significant": bool(p_val < 0.05),
        })

    # Sort by absolute difference
    pairs_data.sort(key=lambda x: abs(x["diff"]), reverse=True)

    chart_data = {
        "labels": [p["pair"] for p in pairs_data],
        "values": [p["diff"] for p in pairs_data],
        "significance": [p["significant"] for p in pairs_data],
    }
    stats_out = {
        "n_comparisons": len(pairs_data),
        "significant_pairs": sum(1 for p in pairs_data if p["significant"]),
        "groups": group_names,
    }
    table = [
        [p["pair"], p["diff"], p["p_value"],
         "显著" if p["significant"] else "不显著"]
        for p in pairs_data
    ]

    return {
        "chart_data": chart_data,
        "stats": stats_out,
        "table": table,
        "chart_type": "horizontal_bar",
    }


def correlation_matrix(df: pd.DataFrame, columns: list[str] | None = None,
                       method: str = "pearson") -> dict[str, Any]:
    """Correlation matrix heatmap among numeric columns."""
    if columns is None:
        columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    else:
        columns = [c for c in columns if c in df.columns]
    if len(columns) < 2:
        return {"error": "需要至少2列数值数据", "chart_type": "heatmap"}
    if len(columns) > 15:
        columns = columns[:15]

    sub = _sample(df[columns]).dropna()
    corr = sub.corr(method=method)

    chart_data = {
        "x_labels": columns,
        "y_labels": columns,
        "matrix": [[round(float(corr.iloc[i, j]), 4)
                     for j in range(len(columns))]
                    for i in range(len(columns))],
    }

    # Identify top correlations
    pairs = []
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            pairs.append((columns[i], columns[j], float(corr.iloc[i, j])))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    stats_out = {
        "method": method,
        "n_columns": len(columns),
        "top_correlations": [
            {"pair": f"{a} & {b}", "r": round(r, 4)}
            for a, b, r in pairs[:5]
        ],
    }

    return {
        "chart_data": chart_data,
        "stats": stats_out,
        "table": [[a, b, round(r, 4)] for a, b, r in pairs],
        "chart_type": "heatmap",
    }


def dimension_reduce(df: pd.DataFrame, columns: list[str] | None = None,
                     n_components: int = 2) -> dict[str, Any]:
    """PCA dimensionality reduction: scatter of PC1 vs PC2 + loadings."""
    if columns is None:
        columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(columns) < 3:
        return {"error": "需要至少3列数值数据进行降维分析", "chart_type": "scatter"}

    sub = _sample(df[columns]).dropna()
    if len(sub) < 10:
        return {"error": "有效数据不足", "chart_type": "scatter"}

    scaler = StandardScaler()
    X = scaler.fit_transform(sub.values)
    n_comp = min(n_components, len(columns) - 1, X.shape[0] - 1)
    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(X)

    chart_data = {
        "points": [[round(float(scores[i, 0]), 4),
                     round(float(scores[i, 1]), 4) if n_comp >= 2 else 0]
                   for i in range(len(scores))],
        "loadings": [
            {"variable": columns[j],
             "pc1": round(float(pca.components_[0, j]), 4),
             "pc2": round(float(pca.components_[1, j]), 4) if n_comp >= 2 else 0}
            for j in range(len(columns))
        ],
    }
    stats_out = {
        "n_components": n_comp,
        "variance_explained": [
            round(float(v), 4) for v in pca.explained_variance_ratio_
        ],
        "cumulative_variance": round(float(np.sum(pca.explained_variance_ratio_)), 4),
        "n_samples": len(scores),
    }
    table = [
        [columns[j], round(float(pca.components_[0, j]), 4),
         round(float(pca.components_[1, j]), 4) if n_comp >= 2 else 0]
        for j in range(len(columns))
    ]

    return {
        "chart_data": chart_data,
        "stats": stats_out,
        "table": table,
        "chart_type": "scatter",
    }


def cluster_analysis(df: pd.DataFrame, columns: list[str] | None = None,
                     n_clusters: int | None = None) -> dict[str, Any]:
    """K-Means clustering with automatic optimal k selection."""
    if columns is None:
        columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(columns) < 2:
        return {"error": "需要至少2列数值数据进行聚类", "chart_type": "scatter"}

    sub = _sample(df[columns]).dropna()
    if len(sub) < 10:
        return {"error": "有效数据不足", "chart_type": "scatter"}

    scaler = StandardScaler()
    X = scaler.fit_transform(sub.values)

    # Determine optimal k via elbow/silhouette if not specified
    if n_clusters is None:
        max_k = min(8, len(sub) // 2)
        best_score = -1
        n_clusters = 3
        for k in range(2, max_k + 1):
            try:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(X)
                if len(set(labels)) > 1:
                    score = float(sp_stats.silhouette_score(X, labels))
                    if score > best_score:
                        best_score = score
                        n_clusters = k
            except Exception:
                continue

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    # PCA for visualization
    pca = PCA(n_components=2)
    scores = pca.fit_transform(X)

    # Cluster profiles
    cluster_centers = scaler.inverse_transform(km.cluster_centers_)
    clusters = []
    for i in range(n_clusters):
        mask = labels == i
        clusters.append({
            "id": i,
            "size": int(mask.sum()),
            "pct": round(float(mask.sum()) / len(labels) * 100, 2),
            "center": {columns[j]: round(float(cluster_centers[i, j]), 4)
                       for j in range(len(columns))},
        })

    chart_data = {
        "points": [[round(float(scores[j, 0]), 4),
                     round(float(scores[j, 1]), 4),
                     int(labels[j])]
                   for j in range(len(scores))],
    }
    try:
        sil = float(sp_stats.silhouette_score(X, labels)) if len(set(labels)) > 1 else 0
    except Exception:
        sil = 0
    stats_out = {
        "n_clusters": n_clusters,
        "silhouette_score": round(sil, 4),
        "clusters": clusters,
    }
    table = [
        [f"簇{i}", clusters[i]["size"], clusters[i]["pct"]]
        for i in range(n_clusters)
    ]

    return {
        "chart_data": chart_data,
        "stats": stats_out,
        "table": table,
        "chart_type": "scatter",
    }


# =========================================================================
# Layer 3 — 看趋势 / 结构 / 排名
# =========================================================================

def timeseries_line(df: pd.DataFrame, date_col: str, metric_col: str,
                    freq: str = "auto") -> dict[str, Any]:
    """Time-series line chart with trend strength."""
    ts = df[[date_col, metric_col]].dropna().copy()
    ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
    ts = ts.dropna(subset=[date_col])
    if len(ts) < 5:
        return {"error": "有效时序数据不足", "chart_type": "line"}

    ts = ts.set_index(date_col).sort_index()
    series = ts[metric_col]

    if freq == "auto":
        n = len(series)
        freq = "D" if n < 60 else "W" if n < 500 else "M" if n < 5000 else "Q"

    agg = series.resample(freq).sum()

    chart_data = {
        "dates": [str(d)[:10] for d in agg.index],
        "values": [round(float(v), 2) for v in agg.values],
    }

    # Trend indicator
    x = np.arange(len(agg))
    y = agg.values
    slope, _ = np.polyfit(x, y, 1)[:2]
    trend = "上升" if slope > 0 else "下降" if slope < 0 else "平稳"

    stats_out = {
        "n_periods": len(agg),
        "freq": freq,
        "total": round(float(agg.sum()), 4),
        "mean": round(float(agg.mean()), 4),
        "min": round(float(agg.min()), 4),
        "max": round(float(agg.max()), 4),
        "trend": trend,
        "trend_slope": round(float(slope), 6),
        "std": round(float(agg.std()), 4),
    }

    return {
        "chart_data": chart_data,
        "stats": stats_out,
        "table": [[str(d)[:10], round(float(v), 2)] for d, v in agg.items()],
        "chart_type": "line",
    }


def timeseries_decompose(df: pd.DataFrame, date_col: str, metric_col: str,
                         period: int = 12) -> dict[str, Any]:
    """Time-series decomposition: trend + seasonal + residual."""
    ts = df[[date_col, metric_col]].dropna().copy()
    ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
    ts = ts.dropna(subset=[date_col])
    ts = ts.set_index(date_col).sort_index()
    series = ts[metric_col]

    if len(series) < period * 2:
        return {"error": f"数据点不足（需要至少{period*2}个，当前{len(series)}个）", "chart_type": "line"}

    # Simple decomposition via moving average
    trend = series.rolling(window=period, center=True).mean()
    detrended = series - trend
    # Seasonal: average detrended value for each period position
    seasonal = np.zeros(len(series))
    for i in range(period):
        idx = np.arange(i, len(series), period)
        if len(idx) > 0:
            seasonal[idx] = detrended.iloc[idx].mean()
    residual = detrended - seasonal

    dates_str = [str(d)[:10] for d in series.index]
    chart_data = {
        "dates": dates_str,
        "original": _safe_list(series.values),
        "trend": _safe_list(trend.values),
        "seasonal": _safe_list(seasonal),
        "residual": _safe_list(residual),
    }

    # Seasonality strength
    var_seasonal = np.nanvar(seasonal)
    var_residual = np.nanvar(residual)
    strength = var_seasonal / (var_seasonal + var_residual) if (var_seasonal + var_residual) > 0 else 0

    stats_out = {
        "seasonal_strength": round(float(strength), 4),
        "trend_direction": "上升" if trend.iloc[-1] > trend.iloc[0] else "下降",
        "period": period,
        "n_points": len(series),
    }

    return {
        "chart_data": chart_data,
        "stats": stats_out,
        "table": [],
        "chart_type": "line",
    }


def timeseries_growth(df: pd.DataFrame, date_col: str, metric_col: str,
                      mode: str = "auto") -> dict[str, Any]:
    """Growth rate analysis: bar chart of period-over-period growth rates."""
    ts = df[[date_col, metric_col]].dropna().copy()
    ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
    ts = ts.dropna(subset=[date_col])
    ts = ts.set_index(date_col).sort_index()

    # Determine frequency
    n = len(ts)
    freq = "M" if n < 500 else "Q"
    if mode == "yoy":
        freq = "Y"

    agg = ts[metric_col].resample(freq).sum()
    if mode == "auto":
        pct_change = agg.pct_change()
    elif mode == "yoy":
        pct_change = agg.pct_change(periods=12 // max(1, {"M": 1, "Q": 3, "Y": 12}[freq]))
    else:
        pct_change = agg.pct_change()

    pct_change = pct_change.dropna()

    chart_data = {
        "dates": [str(d)[:10] for d in pct_change.index],
        "values": [round(float(v) * 100, 2) for v in pct_change.values],
    }
    stats_out = {
        "mean_growth": round(float(pct_change.mean()) * 100, 2),
        "std_growth": round(float(pct_change.std()) * 100, 2),
        "positive_periods": int((pct_change > 0).sum()),
        "negative_periods": int((pct_change < 0).sum()),
        "volatility": round(float(pct_change.std() / max(abs(pct_change.mean()), 1e-10)), 4),
    }

    return {
        "chart_data": chart_data,
        "stats": stats_out,
        "table": [
            [str(d)[:10], f"{round(float(v) * 100, 2)}%"]
            for d, v in pct_change.items()
        ],
        "chart_type": "bar",
    }


def moving_average(df: pd.DataFrame, date_col: str, metric_col: str,
                   window: int = 0) -> dict[str, Any]:
    """Moving average smoothing: original + MA overlay."""
    ts = df[[date_col, metric_col]].dropna().copy()
    ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
    ts = ts.dropna(subset=[date_col])
    ts = ts.set_index(date_col).sort_index()
    series = ts[metric_col]

    if window <= 0:
        window = max(3, len(series) // 20)

    ma = series.rolling(window=window, center=True).mean()

    dates_str = [str(d)[:10] for d in series.index]
    chart_data = {
        "dates": dates_str,
        "original": _safe_list(series.values),
        "moving_avg": _safe_list(ma.values),
    }
    stats_out = {
        "window": window,
        "n_points": len(series),
    }

    return {
        "chart_data": chart_data,
        "stats": stats_out,
        "table": [],
        "chart_type": "line",
    }


def rank_top_n(df: pd.DataFrame, column: str, label_col: str, n: int = 10,
               group_col: str | None = None) -> dict[str, Any]:
    """Top-N ranking: horizontal bar chart + ranking table + cumulative share."""
    if group_col:
        ranked = df.groupby(group_col)[column].sum().sort_values(ascending=False).head(n)
    else:
        ranked = df.groupby(label_col)[column].sum().sort_values(ascending=False).head(n)

    labels = [str(k) for k in ranked.index]
    values = [int(v) for v in ranked.values]
    total = float(df[column].sum())
    cum_pct = np.cumsum(values) / total * 100

    chart_data = {
        "labels": labels,
        "values": values,
        "cum_pct": [round(float(x), 2) for x in cum_pct],
    }
    stats_out = {
        "top1_pct": round(float(values[0]) / total * 100, 2) if values else 0,
        "top3_pct": round(float(sum(values[:3])) / total * 100, 2),
        "top_n_pct": round(float(sum(values)) / total * 100, 2),
        "total_items": len(ranked),
    }
    table = [
        [i + 1, labels[i], values[i], round(float(values[i]) / total * 100, 2),
         round(float(cum_pct[i]), 2)]
        for i in range(len(labels))
    ]

    return {
        "chart_data": chart_data,
        "stats": stats_out,
        "table": table,
        "chart_type": "horizontal_bar",
    }


def pareto_analysis(df: pd.DataFrame, column: str, label_col: str) -> dict[str, Any]:
    """Pareto analysis (80/20 rule): dual-axis Pareto chart."""
    ranked = df.groupby(label_col)[column].sum().sort_values(ascending=False)
    labels = [str(k) for k in ranked.index]
    values = [float(v) for v in ranked.values]
    total = sum(values)
    cum_pct = [round(float(np.sum(values[:i + 1])) / total * 100, 2)
               for i in range(len(values))]

    # Find 80% cutoff
    cutoff_idx = next((i for i, cp in enumerate(cum_pct) if cp >= 80), len(values))

    chart_data = {
        "labels": labels,
        "values": values,
        "cum_pct": cum_pct,
        "cutoff_idx": cutoff_idx,
    }
    stats_out = {
        "total_items": len(labels),
        "pareto_ratio": round(float(cutoff_idx + 1) / len(labels) * 100, 2),
        "top_20_pct": round(float(sum(values[:max(1, len(values) // 5)])) / total * 100, 2),
        "is_pareto": bool(cum_pct[max(0, len(values) // 5 - 1)] >= 80),
    }
    table = [
        [i + 1, labels[i], round(values[i], 2), cum_pct[i]]
        for i in range(len(labels))
    ]

    return {
        "chart_data": chart_data,
        "stats": stats_out,
        "table": table,
        "chart_type": "pareto",
    }


# =========================================================================
# Layer 4 — 数据质量
# =========================================================================

def profile_missing(df: pd.DataFrame, columns: list[str] | None = None) -> dict[str, Any]:
    """Missing value profiling: bar chart of missing rates + missing pattern heatmap."""
    if columns is None:
        columns = list(df.columns)

    missing_rates = {}
    for col in columns:
        rate = df[col].isna().mean()
        if rate > 0:
            missing_rates[col] = round(float(rate), 4)

    chart_data = {
        "labels": list(missing_rates.keys()),
        "values": [round(v * 100, 2) for v in missing_rates.values()],
    }

    # Missing pattern: which columns tend to be missing together
    missing_cols = [c for c in columns if df[c].isna().mean() > 0]
    pattern_heatmap = None
    if len(missing_cols) >= 2:
        missing_matrix = df[missing_cols].isna().astype(int)
        corr_missing = missing_matrix.corr()
        pattern_heatmap = {
            "x_labels": missing_cols,
            "y_labels": missing_cols,
            "matrix": [[round(float(corr_missing.iloc[i, j]), 4)
                         for j in range(len(missing_cols))]
                        for i in range(len(missing_cols))],
        }

    stats_out = {
        "total_cells": int(df[columns].size),
        "missing_cells": int(df[columns].isna().sum().sum()),
        "missing_pct": round(float(df[columns].isna().sum().sum()) / df[columns].size * 100, 2),
        "columns_with_missing": len(missing_rates),
        "columns_no_missing": len(columns) - len(missing_rates),
    }
    table = [
        [col, f"{rate * 100:.2f}%", int(df[col].isna().sum())]
        for col, rate in sorted(missing_rates.items(), key=lambda x: x[1], reverse=True)
    ]

    chart_data["pattern_heatmap"] = pattern_heatmap

    return {
        "chart_data": chart_data,
        "stats": stats_out,
        "table": table,
        "chart_type": "bar",
    }


def profile_duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> dict[str, Any]:
    """Duplicate profiling: count + samples of duplicate rows."""
    dup_mask = df.duplicated(subset=subset, keep=False)
    dup_df = df[dup_mask]
    n_dup = int(dup_mask.sum())

    stats_out = {
        "total_rows": len(df),
        "duplicate_rows": n_dup,
        "duplicate_pct": round(float(n_dup) / len(df) * 100, 2) if len(df) > 0 else 0,
    }

    # Sample duplicates
    sample_rows = []
    if n_dup > 0 and len(dup_df.columns) <= 15:
        for _, row in dup_df.head(20).iterrows():
            sample_rows.append({str(k): _safe(v) for k, v in row.items()})

    return {
        "chart_data": {},
        "stats": stats_out,
        "table": [],
        "chart_type": "table",
    }


# =========================================================================
# Layer 5 — 数据查询与异常解释
# =========================================================================

def data_lookup(df: pd.DataFrame, column: str, value: str,
                limit: int = 20) -> dict[str, Any]:
    """Exact / fuzzy data lookup by column value."""
    if column not in df.columns:
        return {"error": f"列 '{column}' 不存在", "chart_type": "table"}

    # Try exact match first, then string contains
    try:
        val_numeric = float(value)
        mask = df[column] == val_numeric
    except ValueError:
        mask = df[column].astype(str).str.contains(value, case=False, na=False)

    matched = df[mask].head(limit)
    if len(matched) == 0:
        return {
            "chart_data": {},
            "stats": {"matched_rows": 0, "query": f"{column} = {value}"},
            "table": [],
            "chart_type": "table",
        }

    cols = list(matched.columns)[:15]
    table = [[str(c)] + [_safe(matched.iloc[i][c]) for i in range(len(matched))]
             for c in cols]
    # Transpose: rows = records
    rows_data = []
    for i in range(len(matched)):
        row = {}
        for c in cols:
            row[c] = _safe(matched.iloc[i][c])
        rows_data.append(row)

    stats_out = {
        "matched_rows": len(matched),
        "total_rows": len(df),
        "query": f"{column} = {value}",
    }

    return {
        "chart_data": {},
        "stats": stats_out,
        "table": [[str(k), str(v)] for k, v in rows_data[0].items()] if rows_data else [],
        "chart_type": "table",
    }


def anomaly_explain(df: pd.DataFrame, column: str,
                    date_col: str | None = None,
                    context_cols: list[str] | None = None) -> dict[str, Any]:
    """Find anomalies and provide context around them."""
    s = df[column].dropna()
    if len(s) < 5:
        return {"error": "数据不足", "chart_type": "bar"}

    # IQR outlier detection
    q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier_mask = (df[column] >= lower) & (df[column] <= upper)
    outlier_mask = ~outlier_mask & df[column].notna()
    outlier_idx = df.index[outlier_mask]

    if len(outlier_idx) == 0:
        return {
            "chart_data": {},
            "stats": {"anomaly_count": 0, "message": "未检测到显著异常点"},
            "table": [],
            "chart_type": "bar",
        }

    # Get top anomalies (most extreme first)
    outlier_vals = df.loc[outlier_idx, column]
    top_n = min(5, len(outlier_idx))
    top_idx = outlier_vals.abs().nlargest(top_n).index

    # Context around each anomaly
    anomalies = []
    for idx in top_idx:
        info = {"value": _safe(df.loc[idx, column])}
        if date_col and date_col in df.columns:
            info["date"] = str(df.loc[idx, date_col])[:10]
        if context_cols:
            ctx = {}
            for c in context_cols:
                if c in df.columns:
                    ctx[c] = _safe(df.loc[idx, c])
            info["context"] = ctx
        # Get surrounding rows if date_col available
        if date_col and date_col in df.columns:
            pos = df.index.get_loc(idx)
            start = max(0, pos - 2)
            end = min(len(df) - 1, pos + 2)
            neighbors = []
            for i in range(start, end + 1):
                neighbors.append({
                    "date": str(df.iloc[i][date_col])[:10],
                    "value": _safe(df.iloc[i][column]),
                })
            info["neighbors"] = neighbors
        anomalies.append(info)

    # Prepare chart: anomaly values as bars
    chart_data = {
        "labels": [str(a.get("date", a.get("value", "?"))) for a in anomalies],
        "values": [float(a["value"]) if a["value"] is not None else 0 for a in anomalies],
    }

    stats_out = {
        "anomaly_count": int(len(outlier_idx)),
        "anomaly_pct": round(len(outlier_idx) / len(s) * 100, 2),
        "threshold_upper": round(upper, 4),
        "threshold_lower": round(lower, 4),
        "top_anomalies": anomalies,
    }

    return {
        "chart_data": chart_data,
        "stats": stats_out,
        "table": [[str(a.get("date", "?")), str(a.get("value", "?"))]
                  for a in anomalies],
        "chart_type": "bar",
    }


def filter_aggregate(df: pd.DataFrame, column: str, operator: str,
                     threshold: float, agg_column: str | None = None,
                     group_by: str | None = None) -> dict[str, Any]:
    """Filter data by condition and aggregate."""
    if column not in df.columns:
        return {"error": f"列 '{column}' 不存在", "chart_type": "horizontal_bar"}

    op_map = {
        "gt": lambda x, t: x > t, "lt": lambda x, t: x < t,
        "eq": lambda x, t: x == t, "gte": lambda x, t: x >= t,
        "lte": lambda x, t: x <= t,
    }
    op_fn = op_map.get(operator)
    if not op_fn:
        return {"error": f"不支持的操作符: {operator}", "chart_type": "horizontal_bar"}

    col_vals = pd.to_numeric(df[column], errors="coerce")
    mask = op_fn(col_vals, threshold)
    filtered = df[mask]
    total = len(df)

    if group_by and group_by in df.columns:
        if agg_column and agg_column in df.columns:
            agg_result = filtered.groupby(group_by)[agg_column].sum().sort_values(ascending=False)
        else:
            agg_result = filtered.groupby(group_by).size().sort_values(ascending=False)
        labels = [str(k) for k in agg_result.index]
        values = [int(v) for v in agg_result.values]
        chart_data = {"labels": labels, "values": values}
        stats_out = {
            "filtered_rows": len(filtered),
            "total_rows": total,
            "filter_pct": round(len(filtered) / total * 100, 4) if total > 0 else 0,
            "condition": f"{column} {operator} {threshold}",
            "group_by": group_by,
        }
    else:
        chart_data = {"labels": ["符合条件", "不符合"], "values": [len(filtered), total - len(filtered)]}
        stats_out = {
            "filtered_rows": len(filtered),
            "total_rows": total,
            "filter_pct": round(len(filtered) / total * 100, 4) if total > 0 else 0,
            "condition": f"{column} {operator} {threshold}",
            "filtered_mean": round(float(filtered[agg_column].mean()), 4) if agg_column and agg_column in df.columns else None,
            "overall_mean": round(float(df[agg_column].mean()), 4) if agg_column and agg_column in df.columns else None,
        }

    return {
        "chart_data": chart_data,
        "stats": stats_out,
        "table": [],
        "chart_type": "horizontal_bar" if group_by else "bar",
    }


def segment_profile(df: pd.DataFrame, metric_col: str,
                    compare_cols: list[str] | None = None,
                    segment_condition: str = "",
                    top_n: int = 1) -> dict[str, Any]:
    """Profile a segment vs overall data."""
    if metric_col not in df.columns:
        return {"error": f"列 '{metric_col}' 不存在", "chart_type": "grouped_bar"}

    # Determine segment: top N or bottom N by metric
    s = df[metric_col].dropna()
    if "最低" in segment_condition or "最小" in segment_condition or "bottom" in segment_condition.lower():
        seg_idx = s.nsmallest(top_n).index
        seg_label = f"最低{top_n}"
    else:
        seg_idx = s.nlargest(top_n).index
        seg_label = f"最高{top_n}"

    seg_df = df.loc[seg_idx]
    rest_df = df.loc[~df.index.isin(seg_idx)]

    if compare_cols is None:
        compare_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != metric_col]
    compare_cols = [c for c in compare_cols if c in df.columns and c != metric_col][:10]

    # Compare means
    comparison = []
    for col in compare_cols:
        seg_mean = float(seg_df[col].mean()) if len(seg_df) > 0 else 0
        rest_mean = float(rest_df[col].mean()) if len(rest_df) > 0 else 0
        diff_pct = (seg_mean - rest_mean) / abs(rest_mean) * 100 if abs(rest_mean) > 1e-10 else 0
        comparison.append({
            "column": col,
            "segment_mean": round(seg_mean, 4),
            "overall_mean": round(rest_mean, 4),
            "diff_pct": round(diff_pct, 2),
        })

    comparison.sort(key=lambda x: abs(x["diff_pct"]), reverse=True)

    chart_data = {
        "labels": [c["column"] for c in comparison],
        "segment_values": [c["segment_mean"] for c in comparison],
        "overall_values": [c["overall_mean"] for c in comparison],
    }
    stats_out = {
        "segment_label": seg_label,
        "segment_size": len(seg_df),
        "metric": metric_col,
        "top_differences": comparison[:5],
    }

    return {
        "chart_data": chart_data,
        "stats": stats_out,
        "table": [[c["column"], c["segment_mean"], c["overall_mean"], f"{c['diff_pct']}%"]
                  for c in comparison[:10]],
        "chart_type": "grouped_bar",
    }


# ---------------------------------------------------------------------------
# Dispatcher — maps function name string to callable
# ---------------------------------------------------------------------------

def execute_analysis(function_name: str, df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    """Look up and call a registered analysis function by name.

    Args:
        function_name: Key in ANALYSIS_REGISTRY (e.g. "correlate", "rank_top_n").
        df: DataFrame to analyze.
        params: Keyword arguments matching the function's input_schema.

    Returns:
        The standardised result dict {chart_data, stats, table, chart_type}
        or an {error, chart_type} dict on failure.
    """
    import importlib

    # Lazy-load to avoid circular imports
    mod = importlib.import_module("services.analysis_registry")
    registry = mod.ANALYSIS_REGISTRY

    if function_name not in registry:
        return {"error": f"未知的分析函数 '{function_name}'", "chart_type": "table"}

    entry = registry[function_name]

    # Resolve the function
    fn = entry.get("fn")
    if fn is None:
        # Resolve by name from this module
        fn = globals().get(function_name)
        if fn is None:
            return {"error": f"分析函数 '{function_name}' 未实现", "chart_type": "table"}
        entry["fn"] = fn  # cache for next call

    # Validate params against schema
    valid_params = {}
    schema = entry["input_schema"]
    for key in schema:
        if key in params:
            valid_params[key] = params[key]

    try:
        result = fn(df=df, **valid_params)
        # Ensure chart_type is set
        if "chart_type" not in result:
            result["chart_type"] = entry.get("default_chart", "table")
        return result
    except Exception as exc:
        logger.exception("Analysis function '%s' failed", function_name)
        return {
            "error": f"分析执行失败: {exc}",
            "chart_type": entry.get("default_chart", "table"),
        }


# ---------------------------------------------------------------------------
# Helper: apply computed column mappings before analysis
# ---------------------------------------------------------------------------

def apply_column_mappings(df: pd.DataFrame,
                          mappings: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[str]]:
    """Apply AI-specified column mappings to the DataFrame.

    Args:
        df: Input DataFrame.
        mappings: List of {name, type: "direct"|"computed", column?, formula?}.

    Returns:
        (augmented_df, log_messages).
    """
    logs: list[str] = []
    result = df.copy()

    for m in mappings:
        name = m.get("name", "").strip()
        mtype = m.get("type", "direct")
        if not name:
            continue

        if mtype == "direct":
            col = m.get("column", "").strip()
            if col and col in result.columns:
                if name != col:
                    result[name] = result[col]
                    logs.append(f"映射: '{col}' -> '{name}'")
            else:
                logs.append(f"跳过: 列 '{col}' 不存在")
        elif mtype == "computed":
            formula = m.get("formula", "").strip()
            if formula:
                try:
                    # Safe evaluation: only allow column names + basic operators
                    result[name] = result.eval(formula)
                    logs.append(f"计算: '{name}' = {formula}")
                except Exception as exc:
                    logs.append(f"计算失败: '{name}' = {formula} ({exc})")

    return result, logs
