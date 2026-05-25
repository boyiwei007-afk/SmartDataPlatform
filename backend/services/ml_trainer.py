"""
ML Trainer — Industrial-Grade Linear Regression Pipeline

Provides ``train_linear_regression()`` with a 4-stage data-cleaning pipeline:

1. **Input unwrap & validation** — unwrap single-element arrays, verify
   column existence.
2. **Forced numeric conversion** — ``pd.to_numeric(errors="coerce")`` on
   every selected column; non-numeric values silently become NaN.
3. **Smart imputation** — rows with NaN in Y are dropped; NaN cells in X
   are filled via ``SimpleImputer(strategy="median")``.
4. **Type coercion** — all NumPy scalars cast to native Python ``float`` /
   ``int`` before JSON serialisation.

5. **Standardized feature importance** — computes standardized coefficients
   (beta weights) so users can compare feature impact regardless of unit scale.
   Original coefficients are preserved for prediction; standardized coefficients
   are returned separately for importance ranking.

Zero hard-coded column names — fully data-driven.
学生手动更改代码部分：增加数据标准化流程，增加数据的泛化功能，不局限于原始的数据。更改比例大概是30%
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_linear_regression(
    df: pd.DataFrame,
    target_col: str | list[str],
    feature_cols: list[str],
) -> dict[str, Any]:
    """Train a linear regression model with automatic data cleaning.

    Args:
        df:           The full DataFrame (already loaded by the data engine).
        target_col:   Column name (or single-element list) for Y.
        feature_cols: Column names for X.

    Returns:
        ``{target, features, intercept, coefficients, r2_score,
           feature_stats, feature_importance, n_samples, n_dropped,
           n_imputed}`` — all values are native Python types, ready for
        JSON serialisation.

        The new ``feature_importance`` field contains:
        - standardized_coefficient: beta weight (how many std-dev changes
          in Y per 1 std-dev change in X)
        - relative_importance: percentage of total absolute importance
        - correlation_with_target: Pearson r with Y
        - x_mean, x_std: mean and std-dev of the feature in training data

    Raises:
        ValueError: If a column is missing, the cleaned dataset is empty,
                    or the feature set is invalid.
    """
    # =====================================================================
    # Stage 0 — defensive input unwrap
    # =====================================================================
    if isinstance(target_col, (list, tuple)):
        if len(target_col) == 0:
            raise ValueError("目标变量 (Y) 为空，请选择一个数值型列。")
        target_col = str(target_col[0])
        logger.info("Unwrapped target_col from list → '%s'", target_col)

    target_col = str(target_col).strip()
    if not target_col:
        raise ValueError("目标变量 (Y) 不能为空，请选择一个数值型列。")

    feature_cols = [str(c).strip() for c in feature_cols if str(c).strip()]
    if not feature_cols:
        raise ValueError("影响因素 (X) 不能为空，请至少选择一个特征列。")

    # =====================================================================
    # Stage 1 — column existence check
    # =====================================================================
    all_cols = set(df.columns)
    missing: list[str] = []
    if target_col not in all_cols:
        missing.append(target_col)
    for c in feature_cols:
        if c not in all_cols:
            missing.append(c)
    if missing:
        raise ValueError(
            f"以下列在数据中不存在: {missing}。"
            f"当前数据列: {sorted(all_cols)}"
        )

    # =====================================================================
    # Stage 2 — forced numeric conversion (silent coercion)
    # =====================================================================
    cols_to_use = [target_col] + feature_cols
    sub = df[cols_to_use].copy()
    total_before = len(sub)

    for col in cols_to_use:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")

    # =====================================================================
    # Stage 3a — drop rows where Y is NaN (useless for training)
    # =====================================================================
    before_y_drop = len(sub)
    sub = sub.dropna(subset=[target_col])
    n_dropped_y = before_y_drop - len(sub)

    if len(sub) == 0:
        raise ValueError(
            f"目标变量 '{target_col}' 在数值化后全部为空，无法训练模型。"
            f"原始 {total_before} 行数据均无法使用，请检查该列是否包含有效数值。"
        )

    # =====================================================================
    # Stage 3b — impute X NaN cells with median (preserve sample size)
    # =====================================================================
    X_raw = sub[feature_cols].values
    y_raw = sub[target_col].values

    nan_mask = np.isnan(X_raw)
    n_nan_cells = int(np.sum(nan_mask))

    imputer = SimpleImputer(strategy="median")
    X_clean = imputer.fit_transform(X_raw)

    n_samples = len(sub)

    if n_samples < len(feature_cols) * 2:
        logger.warning(
            "Sample size (%d) small vs feature count (%d). R² may be unreliable.",
            n_samples, len(feature_cols),
        )

    # =====================================================================
    # Stage 4 — train model on raw (unstandardized) data
    # =====================================================================
    model = LinearRegression()
    model.fit(X_clean, y_raw)

    r2 = float(model.score(X_clean, y_raw))
    intercept = float(model.intercept_)
    coefficients: dict[str, float] = {}
    for i, col in enumerate(feature_cols):
        coefficients[col] = float(model.coef_[i])

    # =====================================================================
    # Stage 4.5 — compute standardized feature importance
    #
    # Why not standardize before training?
    #   We want the ORIGINAL coefficients for the sandbox prediction
    #   (so users see "每增加1单位广告投入，销售额增加0.032万"),
    #   but we ALSO want STANDARDIZED coefficients for fair comparison
    #   (so users know which feature matters most regardless of units).
    #
    # Formula:
    #   β_std = β_raw × (σ_x / σ_y)
    #
    # Interpretation:
    #   "If X increases by 1 standard deviation, Y changes by β_std
    #    standard deviations."
    #
    # Why this is a post-hoc calculation instead of training on
    # standardized data: it avoids the complexity of de-standardizing
    # coefficients back to original units for prediction, and keeps
    # the training pipeline simple.
    # =====================================================================
    feature_importance: dict[str, dict[str, float]] = {}

    # Compute means and standard deviations (ddof=1 for sample std)
    x_means = np.mean(X_clean, axis=0)
    x_stds = np.std(X_clean, axis=0, ddof=1)
    y_mean = float(np.mean(y_raw))
    y_std = float(np.std(y_raw, ddof=1))

    # Guard against zero variance in Y (would make standardized
    # coefficients meaningless)
    if y_std < 1e-10:
        logger.warning(
            "Target variable '%s' has near-zero variance (σ=%.6f). "
            "Standardized coefficients will be skipped.",
            target_col, y_std,
        )
    else:
        for i, col in enumerate(feature_cols):
            # Standardized coefficient (beta weight)
            if x_stds[i] > 1e-10:
                std_beta = float(model.coef_[i] * x_stds[i] / y_std)
            else:
                # Feature has zero variance — it contributes nothing
                std_beta = 0.0
                logger.warning(
                    "Feature '%s' has near-zero variance, "
                    "standardized coefficient set to 0.", col,
                )

            # Correlation between this feature and Y
            corr_matrix = np.corrcoef(X_clean[:, i], y_raw)
            corr_xy = float(corr_matrix[0, 1])
            if np.isnan(corr_xy):
                corr_xy = 0.0

            feature_importance[col] = {
                "standardized_coefficient": round(std_beta, 6),
                "original_coefficient": round(float(model.coef_[i]), 6),
                "correlation_with_target": round(corr_xy, 6),
                "relative_importance": 0.0,  # computed below
                "x_mean": round(float(x_means[i]), 4),
                "x_std": round(float(x_stds[i]), 4),
            }

        # Compute relative importance (percentage of total absolute beta)
        total_abs_beta = sum(
            abs(v["standardized_coefficient"])
            for v in feature_importance.values()
        )
        if total_abs_beta > 1e-10:
            for col in feature_importance:
                feature_importance[col]["relative_importance"] = round(
                    abs(feature_importance[col]["standardized_coefficient"])
                    / total_abs_beta
                    * 100,
                    2,
                )

        # Sort by absolute standardized coefficient (descending)
        feature_importance = dict(
            sorted(
                feature_importance.items(),
                key=lambda item: abs(item[1]["standardized_coefficient"]),
                reverse=True,
            )
        )

    # =====================================================================
    # Stage 5 — feature statistics (for sandbox slider bounds)
    # =====================================================================
    feature_stats: dict[str, dict[str, float]] = {}
    for i, col in enumerate(feature_cols):
        col_vals = X_clean[:, i]
        feature_stats[col] = {
            "min": float(np.min(col_vals)),
            "max": float(np.max(col_vals)),
            "mean": float(np.mean(col_vals)),
        }

    # =====================================================================
    # Stage 6 — summary statistics for the target variable
    # =====================================================================
    target_stats = {
        "mean": round(y_mean, 6),
        "std": round(y_std, 6),
    }

    # =====================================================================
    # Logging
    # =====================================================================
    logger.info(
        "Model trained: target=%s, features=%s, R²=%.4f, n=%d, "
        "dropped_y=%d, imputed=%d",
        target_col, feature_cols, r2, n_samples, n_dropped_y, n_nan_cells,
    )

    if feature_importance:
        top_feature = next(iter(feature_importance))
        top_importance = feature_importance[top_feature]["relative_importance"]
        logger.info(
            "Top feature: '%s' (%.1f%% relative importance, β_std=%.4f)",
            top_feature, top_importance,
            feature_importance[top_feature]["standardized_coefficient"],
        )

    # =====================================================================
    # Stage 5.5 — VIF (multicollinearity diagnostic)
    # =====================================================================
    vif_data: dict[str, float] = {}
    vif_warnings: list[str] = []
    if len(feature_cols) >= 2:
        try:
            vif_data = _compute_vif(X_clean, feature_cols)
            for col, v in vif_data.items():
                if v > 10:
                    vif_warnings.append(f"'{col}' VIF={v:.1f}, 存在严重多重共线性")
                elif v > 5:
                    vif_warnings.append(f"'{col}' VIF={v:.1f}, 存在中等多重共线性")
        except Exception:
            logger.warning("VIF calculation failed", exc_info=True)

    # =====================================================================
    # Stage 5.6 — Residual analysis
    # =====================================================================
    y_pred = model.predict(X_clean)
    residuals = y_raw - y_pred
    residuals_data = _residual_analysis(y_raw, y_pred, residuals)

    # =====================================================================
    # Stage 5.7 — K-fold cross-validation
    # =====================================================================
    cv_r2_mean, cv_r2_std = _cross_val_r2(X_clean, y_raw)

    # =====================================================================
    # Stage 5.8 — Ridge / Lasso comparison
    # =====================================================================
    model_comparison = _compare_models(X_clean, y_raw, len(feature_cols))

    # =====================================================================
    # Build return value
    # =====================================================================
    result: dict[str, Any] = {
        "target": target_col,
        "features": feature_cols,
        "intercept": round(intercept, 6),
        "coefficients": {
            k: round(v, 6) for k, v in coefficients.items()
        },
        "r2_score": round(r2, 6),
        "feature_stats": {
            k: {kk: round(vv, 6) for kk, vv in v.items()}
            for k, v in feature_stats.items()
        },
        "target_stats": target_stats,
        "n_samples": int(n_samples),
        "n_dropped": int(n_dropped_y),
        "n_imputed": int(n_nan_cells),
    }

    # Attach feature importance only if we successfully computed it
    if feature_importance:
        result["feature_importance"] = feature_importance
        result["importance_note"] = (
            "标准化系数 (standardized_coefficient) 表示该特征每增加1个标准差，"
            "目标变量变化的标准差数量。"
            "相对重要性 (relative_importance) 基于标准化系数的绝对值计算，"
            "总和为100%。"
            "原始系数 (original_coefficient) 用于沙盘预测，"
            "表示特征每增加1个原始单位时目标变量的变化量。"
        )

    # ---- Diagnostics (new) ----
    result["diagnostics"] = {
        "vif": {k: round(v, 2) for k, v in vif_data.items()},
        "vif_warnings": vif_warnings,
        "residuals": residuals_data,
        "cross_val_r2": round(cv_r2_mean, 6),
        "cross_val_r2_std": round(cv_r2_std, 6),
        "cross_val_folds": 5,
        "model_comparison": model_comparison,
        "residual_std": round(float(np.std(residuals, ddof=len(feature_cols)+1)), 6),
    }

    return result


# =========================================================================
# Diagnostic helpers
# =========================================================================


def _compute_vif(X: np.ndarray, feature_names: list[str]) -> dict[str, float]:
    """Compute Variance Inflation Factor for each feature.

    VIF = 1 / (1 - R²_i) where R²_i is from regressing feature i on all others.
    VIF > 5  → moderate multicollinearity
    VIF > 10 → severe multicollinearity
    """
    vif: dict[str, float] = {}
    n_features = X.shape[1]
    for i in range(n_features):
        # Regress X[:, i] on all other columns
        x_i = X[:, i]
        x_others = np.delete(X, i, axis=1)
        if x_others.shape[1] == 0:
            vif[feature_names[i]] = 1.0
            continue
        # Add intercept
        x_others = np.column_stack([np.ones(x_others.shape[0]), x_others])
        try:
            coeffs, residuals_ss, rank, _ = np.linalg.lstsq(x_others, x_i, rcond=None)
            ss_total = np.sum((x_i - np.mean(x_i)) ** 2)
            if len(residuals_ss) > 0 and ss_total > 1e-10:
                r_squared = 1 - residuals_ss[0] / ss_total
                vif[feature_names[i]] = float(1 / (1 - r_squared)) if r_squared < 1 else 1e6
            else:
                vif[feature_names[i]] = 1.0
        except Exception:
            vif[feature_names[i]] = 1.0
    return vif


def _residual_analysis(y_true: np.ndarray, y_pred: np.ndarray,
                       residuals: np.ndarray) -> dict[str, Any]:
    """Compute residual diagnostics: QQ data, stats, outlier count."""
    n = len(residuals)
    # Standardized residuals
    std_resid = residuals / np.std(residuals, ddof=1) if np.std(residuals) > 1e-10 else residuals

    # QQ plot data: theoretical quantiles vs sample quantiles
    theo_q = sp_stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    sample_q = np.sort(std_resid)
    # Downsample for rendering
    step = max(1, n // 300)
    qq_x = [round(float(x), 4) for x in theo_q[::step]]
    qq_y = [round(float(y), 4) for y in sample_q[::step]]

    # Predicted vs actual (downsampled)
    step2 = max(1, n // 500)
    pred_vs_actual = [
        [round(float(y_true[i]), 4), round(float(y_pred[i]), 4)]
        for i in range(0, n, step2)
    ]

    # Outlier residuals (|std_resid| > 3)
    outlier_mask = np.abs(std_resid) > 3

    return {
        "qq_x": qq_x,
        "qq_y": qq_y,
        "pred_vs_actual": pred_vs_actual,
        "mean_residual": round(float(np.mean(residuals)), 6),
        "std_residual": round(float(np.std(residuals)), 6),
        "skewness": round(float(sp_stats.skew(residuals)), 4),
        "kurtosis": round(float(sp_stats.kurtosis(residuals)), 4),
        "n_outliers": int(np.sum(outlier_mask)),
        "durbin_watson": round(_durbin_watson(residuals), 4),
    }


def _durbin_watson(residuals: np.ndarray) -> float:
    """Durbin-Watson statistic for autocorrelation in residuals."""
    diff = np.diff(residuals)
    numerator = np.sum(diff ** 2)
    denominator = np.sum(residuals ** 2)
    if denominator < 1e-10:
        return 2.0
    return float(numerator / denominator)


def _cross_val_r2(X: np.ndarray, y: np.ndarray, cv: int = 5) -> tuple[float, float]:
    """K-fold cross-validated R² score."""
    n = len(y)
    if n < cv * 2:
        cv = min(3, n // 2)
    if cv < 2:
        return 0.0, 0.0
    try:
        model = LinearRegression()
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
        return float(np.mean(scores)), float(np.std(scores))
    except Exception:
        return 0.0, 0.0


def _compare_models(X: np.ndarray, y: np.ndarray,
                    n_features: int) -> dict[str, dict[str, float]]:
    """Compare LinearRegression vs Ridge vs Lasso on the same data.

    Uses an 80/20 split for a quick comparison (cross-validation is done
    separately for the main model).
    """
    comparison: dict[str, dict[str, float]] = {}
    n = len(y)
    if n < 10:
        return comparison

    # Simple train/test split
    split_idx = int(n * 0.8)
    indices = np.random.RandomState(42).permutation(n)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    alpha = 1.0  # default regularisation

    models: dict[str, Any] = {
        "Linear": LinearRegression(),
    }
    if n_features > 3:
        models["Ridge"] = Ridge(alpha=alpha)
        models["Lasso"] = Lasso(alpha=alpha / 10, max_iter=5000)

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_hat = model.predict(X_test)
            ss_res = np.sum((y_test - y_hat) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0
            rmse = np.sqrt(np.mean((y_test - y_hat) ** 2))
            comparison[name] = {
                "r2": round(float(r2), 6),
                "rmse": round(float(rmse), 4),
            }
        except Exception:
            continue

    return comparison