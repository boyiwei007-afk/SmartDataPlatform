"""
Analysis Registry — central catalogue of all analysis functions the AI can invoke.

Each entry maps a function name to its metadata:
- fn: the callable (lazy-imported at execution time)
- category: one of distribution | relationship | trend | quality
- description_for_ai: natural-language description fed into the LLM prompt
- input_schema: parameter names and accepted values
- default_chart: the chart type this analysis naturally maps to

Adding a new function is two steps:
1. Implement it in analysis_functions.py
2. Register it here — the AI prompt auto-reflects the change
"""

from __future__ import annotations

from typing import Any, Callable

# ---------------------------------------------------------------------------
# Registry entry type
# ---------------------------------------------------------------------------

RegistryEntry = dict[str, Any]  # {fn, category, description_for_ai, input_schema, default_chart}

# ---------------------------------------------------------------------------
# Master registry
# ---------------------------------------------------------------------------

ANALYSIS_REGISTRY: dict[str, RegistryEntry] = {
    # =========================================================================
    # Layer 1 — 看分布 (univariate)
    # =========================================================================
    "describe_distribution": {
        "category": "distribution",
        "description_for_ai": (
            "数值列的完整分布画像，输出直方图+均值/标准差/四分位数/偏度/峰度+正态性检验p值。"
            "适合问题：'这个变量的分布情况？''数据是否符合正态分布？''XX的整体水平如何？'"
        ),
        "input_schema": {"column": "str (数值列名)"},
        "default_chart": "histogram",
    },
    "describe_frequency": {
        "category": "distribution",
        "description_for_ai": (
            "分类列的频次统计，输出横向柱状图+各类计数/占比+信息熵+累积占比。"
            "适合问题：'各类别分别有多少？''哪个类别最多？''类别的均衡程度如何？'"
        ),
        "input_schema": {"column": "str (分类列名)", "top_n": "int (可选, 默认10)"},
        "default_chart": "horizontal_bar",
    },
    "detect_outliers": {
        "category": "distribution",
        "description_for_ai": (
            "数值列的异常值检测，使用IQR(四分位距)+Z-score双重方法，输出箱线图+异常值列表+异常占比。"
            "适合问题：'有没有异常值？''哪些记录是异常的？''数据的离群程度有多严重？'"
        ),
        "input_schema": {"column": "str (数值列名)", "method": "iqr|zscore|both (默认both)"},
        "default_chart": "boxplot",
    },
    "test_normality": {
        "category": "distribution",
        "description_for_ai": (
            "数值列的正态性检验，输出QQ图+Shapiro-Wilk/D'Agostino检验结果+偏度/峰度。"
            "适合问题：'这个变量符合正态分布吗？''数据是否需要做对数变换？'"
        ),
        "input_schema": {"column": "str (数值列名)"},
        "default_chart": "qq",
    },
    "analyze_composition": {
        "category": "distribution",
        "description_for_ai": (
            "分类列的构成占比分析，输出环形图+HHI集中度指数+各类占比表。"
            "适合问题：'各类别的占比是多少？''市场集中度如何？''成分结构是什么样的？'"
        ),
        "input_schema": {"column": "str (分类列名)", "top_n": "int (可选, 默认10, 其余归为'其他')"},
        "default_chart": "donut",
    },

    # =========================================================================
    # Layer 2 — 看关系 (bivariate / multivariate)
    # =========================================================================
    "correlate": {
        "category": "relationship",
        "description_for_ai": (
            "两个数值列的相关性分析，输出散点图+趋势线+Pearson/Spearman相关系数+p值+置信区间。"
            "适合问题：'X和Y有关系吗？''X增加会导致Y增加吗？''两个变量的关联强度如何？'"
        ),
        "input_schema": {"x": "str (数值列名)", "y": "str (数值列名)", "method": "pearson|spearman (默认pearson)"},
        "default_chart": "scatter",
    },
    "cross_tabulate": {
        "category": "relationship",
        "description_for_ai": (
            "两个分类列的交叉分析，输出堆叠柱状图+列联表+卡方独立性检验+Cramér's V效应量。"
            "适合问题：'A和B之间有关联吗？''不同X类别下的Y分布是否有差异？'"
        ),
        "input_schema": {"x": "str (分类列名)", "y": "str (分类列名)"},
        "default_chart": "stacked_bar",
    },
    "compare_groups": {
        "category": "relationship",
        "description_for_ai": (
            "按分类列分组比较数值列，输出分组箱线图+各组均值/中位数表+ANOVA/Kruskal-Wallis检验+eta-squared效应量。"
            "适合问题：'不同X的Y有什么区别？''哪个组的Y最高？''组间差异是否显著？'"
        ),
        "input_schema": {"group_col": "str (分类列名)", "value_col": "str (数值列名)"},
        "default_chart": "boxplot",
    },
    "compare_pairs": {
        "category": "relationship",
        "description_for_ai": (
            "多组之间的两两配对比较，输出Tukey HSD事后检验+显著性字母标记+组间差异矩阵。"
            "适合问题：'具体哪两个组之间有显著差异？''各组之间的差异有多大？'"
        ),
        "input_schema": {"group_col": "str (分类列名)", "value_col": "str (数值列名)"},
        "default_chart": "horizontal_bar",
    },
    "correlation_matrix": {
        "category": "relationship",
        "description_for_ai": (
            "多个数值列的相关性矩阵，输出聚类热力图+相关系数表+显著性标记。"
            "适合问题：'所有数值列之间的关联关系？''哪些变量彼此高度相关？''是否存在多重共线性？'"
        ),
        "input_schema": {"columns": "[str, ...] (可选, 默认所有数值列)", "method": "pearson|spearman (默认pearson)"},
        "default_chart": "heatmap",
    },
    "dimension_reduce": {
        "category": "relationship",
        "description_for_ai": (
            "PCA主成分分析降维，输出PC1-PC2散点图+载荷图+各主成分解释方差比+累计方差。"
            "适合问题：'数据的主要维度是什么？''哪些变量在同一个方向上变化？''数据可以降维到几个维度？'"
        ),
        "input_schema": {"columns": "[str, ...] (可选, 默认所有数值列)", "n_components": "int (可选, 默认2)"},
        "default_chart": "scatter",
    },
    "cluster_analysis": {
        "category": "relationship",
        "description_for_ai": (
            "K-Means聚类分析，输出聚类散点图（PC空间）+簇标签+簇中心+轮廓系数+簇大小分布。"
            "适合问题：'数据天然分成几类？''哪些样本彼此相似？''各类的特征是什么？'"
        ),
        "input_schema": {"columns": "[str, ...] (可选, 默认所有数值列)", "n_clusters": "int (可选, 自动由轮廓系数确定)"},
        "default_chart": "scatter",
    },

    # =========================================================================
    # Layer 3 — 看趋势 / 结构 / 排名 (trend / ranking)
    # =========================================================================
    "timeseries_line": {
        "category": "trend",
        "description_for_ai": (
            "时间序列折线图，输出折线图+趋势强度指标+均值/标准差。"
            "适合问题：'X随时间的变化趋势？''数据是否有上升/下降趋势？'"
        ),
        "input_schema": {"date_col": "str (日期列名)", "metric_col": "str (数值列名)", "freq": "D|W|M|Q|Y (默认自动推断)"},
        "default_chart": "line",
    },
    "timeseries_decompose": {
        "category": "trend",
        "description_for_ai": (
            "时间序列三要素分解（趋势+季节+残差），输出三分图+季节性强度+趋势方向判断。"
            "适合问题：'数据有季节性吗？''去掉季节效应后的趋势是什么？''残差中还有没有未解释的模式？'"
        ),
        "input_schema": {"date_col": "str (日期列名)", "metric_col": "str (数值列名)", "period": "int (周期, 如12表示12个月)"},
        "default_chart": "line",
    },
    "timeseries_growth": {
        "category": "trend",
        "description_for_ai": (
            "时间序列的同比/环比增长率分析，输出增长率柱状图+平均增长率+波动率+增长稳定性评分。"
            "适合问题：'增长速度是多少？''哪个时期增长最快？''增长是否在加速还是放缓？'"
        ),
        "input_schema": {"date_col": "str (日期列名)", "metric_col": "str (数值列名)", "mode": "yoy|mom (同比/环比, 默认自动)"},
        "default_chart": "bar",
    },
    "moving_average": {
        "category": "trend",
        "description_for_ai": (
            "移动平均平滑去噪，输出原始数据+移动平均线叠加折线图+不同窗口的平滑效果对比。"
            "适合问题：'去除噪声后的长期趋势是什么？''数据的平滑走势是怎样的？'"
        ),
        "input_schema": {"date_col": "str (日期列名)", "metric_col": "str (数值列名)", "window": "int (窗口大小, 默认自动)"},
        "default_chart": "line",
    },
    "rank_top_n": {
        "category": "trend",
        "description_for_ai": (
            "Top-N排名分析，输出横向柱状图+排名表+累计占比+帕累托比。"
            "适合问题：'排名前10的是哪些？''最大的几个占比多少？''长尾分布有多明显？'"
        ),
        "input_schema": {"column": "str (数值列名)", "label_col": "str (标签列名)", "n": "int (默认10)", "group_col": "str (可选, 分组排名用的列)"},
        "default_chart": "horizontal_bar",
    },
    "pareto_analysis": {
        "category": "trend",
        "description_for_ai": (
            "帕累托分析（80/20法则），输出帕累托图（双轴：柱状+累积折线）+80%临界点标记+关键少数占比。"
            "适合问题：'是否符合二八定律？''哪些是关键的少数？''80%的效果来自哪20%的原因？'"
        ),
        "input_schema": {"column": "str (数值列名)", "label_col": "str (标签列名)"},
        "default_chart": "pareto",
    },

    # =========================================================================
    # Layer 4 — 数据质量 (data quality)
    # =========================================================================
    "profile_missing": {
        "category": "quality",
        "description_for_ai": (
            "缺失值全面画像，输出缺失率柱状图+缺失模式热力图（哪些列倾向于同时缺失）+逐列缺失统计表。"
            "适合问题：'数据缺失情况如何？''哪些列缺失最严重？''缺失值之间是否有关联模式？'"
        ),
        "input_schema": {"columns": "[str, ...] (可选, 默认所有列)"},
        "default_chart": "bar",
    },
    "profile_duplicates": {
        "category": "quality",
        "description_for_ai": (
            "重复值分析，输出重复行数+重复样本展示+逐列重复率。"
            "适合问题：'有多少重复数据？''重复记录是什么样的？''哪些列的值高度重复？'"
        ),
        "input_schema": {"subset": "[str, ...] (可选, 指定去重参照列)"},
        "default_chart": "table",
    },

    # =========================================================================
    # Layer 5 — 数据查询与异常解释 (lookup / anomaly / segment)
    # =========================================================================
    "data_lookup": {
        "category": "trend",
        "description_for_ai": (
            "按条件精确查询数据行，返回匹配行的完整数据。"
            "适合问题：'2023年3月15日的数据是多少？''单价大于100的记录有哪些？'"
            "'某个具体日期/ID/名称对应的记录？'"
        ),
        "input_schema": {"column": "str (用于匹配的列名)", "value": "str (要查找的值)", "limit": "int (可选, 返回行数上限, 默认20)"},
        "default_chart": "table",
    },
    "anomaly_explain": {
        "category": "distribution",
        "description_for_ai": (
            "定位异常点并提取其上下文数据，帮助解释异常原因。先找出异常值，再展示异常点前后的数据上下文。"
            "适合问题：'为什么X这一天异常高？''异常点附近发生了什么？''解释这个高峰的原因'"
        ),
        "input_schema": {"column": "str (要分析的数值列)", "date_col": "str (可选, 时间列, 用于展示前后文)", "context_cols": "[str, ...] (可选, 需要展示的上下文列)"},
        "default_chart": "bar",
    },
    "filter_aggregate": {
        "category": "trend",
        "description_for_ai": (
            "按条件过滤数据后聚合统计。支持数值条件（>、<、=）。"
            "适合问题：'销售额大于1000的订单有哪些特征？''价格超过平均值2倍的产品有多少？'"
        ),
        "input_schema": {"column": "str (用于过滤的列)", "operator": "gt|lt|eq|gte|lte (条件)", "threshold": "float (阈值)", "agg_column": "str (可选, 要聚合的列)", "group_by": "str (可选, 分组列)"},
        "default_chart": "horizontal_bar",
    },
    "segment_profile": {
        "category": "distribution",
        "description_for_ai": (
            "对数据的某个分段（如高峰日、低谷周、高价值客户群）做特征画像，将该分段与其他数据对比。"
            "适合问题：'高峰日和其他日子有什么不同？''高价值客户的特征是什么？'"
        ),
        "input_schema": {"segment_condition": "str (分段条件描述, 如'Revenue最高的一天')", "metric_col": "str (用于确定分段的指标列)", "compare_cols": "[str, ...] (要对比的特征列)", "top_n": "int (分段大小, 默认1)"},
        "default_chart": "grouped_bar",
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Reverse index: category -> function names
CATEGORY_MAP: dict[str, list[str]] = {}
for _name, _entry in ANALYSIS_REGISTRY.items():
    _cat = _entry["category"]
    CATEGORY_MAP.setdefault(_cat, []).append(_name)


def get_registry() -> dict[str, RegistryEntry]:
    """Return the full registry (convenience accessor)."""
    return ANALYSIS_REGISTRY


def get_function_names_by_category(category: str) -> list[str]:
    """Return function names belonging to a category."""
    return CATEGORY_MAP.get(category, [])


def generate_ai_functions_catalog() -> str:
    """Generate the 'available analysis functions' section for the LLM prompt.

    The output is a Markdown-formatted catalogue that gets embedded directly
    into the analysis-planning system prompt.  Each function entry includes
    its natural-language description, input parameters, and default chart type.
    """
    lines: list[str] = []
    _category_labels = {
        "distribution": "看分布（单变量）",
        "relationship": "看关系（双变量 / 多变量）",
        "trend": "看趋势 / 结构 / 排名",
        "quality": "数据质量",
    }

    for cat, label in _category_labels.items():
        names = CATEGORY_MAP.get(cat, [])
        if not names:
            continue
        lines.append(f"### {label}")
        for name in names:
            entry = ANALYSIS_REGISTRY[name]
            lines.append(f"- **{name}**({_format_params(entry['input_schema'])})")
            lines.append(f"  {entry['description_for_ai']}")
            lines.append(f"  默认图表: {entry['default_chart']}")
        lines.append("")

    return "\n".join(lines)


def _format_params(schema: dict[str, str]) -> str:
    """Format input_schema into a concise parameter string for the prompt."""
    parts: list[str] = []
    for param, desc in schema.items():
        parts.append(f"{param}: {desc}")
    return ", ".join(parts)


def generate_chart_types_list() -> str:
    """Return a list of all supported chart types for the LLM prompt."""
    chart_types = sorted({
        entry["default_chart"]
        for entry in ANALYSIS_REGISTRY.values()
    })
    return ", ".join(chart_types)
