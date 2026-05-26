"""
LLM Agent — Natural-Language Q&A Engine + Analysis Planner + Interpreter

Three modes:
1. ``ask_data()`` — legacy synchronous NL Q&A (kept for Tab 1 data summary).
2. ``stream_analyze()`` — async generator that yields SSE events for the
   4-stage analysis pipeline (translation -> plan -> (execution done by caller) -> interpretation).
3. Helper builders: ``build_translation_prompt()``, ``build_planning_prompt()``,
   ``build_interpretation_prompt()`` — reusable prompt constructors.

All credentials arrive from the frontend per-request; nothing is persisted.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API (legacy, kept for backward compatibility)
# ---------------------------------------------------------------------------


def ask_data(
    query: str,
    schema: dict[str, Any],
    api_credentials: dict[str, str],
) -> dict[str, Any]:
    """Legacy synchronous NL Q&A.  Kept for Tab 1 data-summary and simple queries."""
    system_prompt = _build_prompt(schema)
    raw_answer = _call_llm(system_prompt, query, api_credentials)
    return {"answer": raw_answer.strip()}


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
你是一个专业的数据分析助手。用户上传了一个表格，下面是该表格的结构摘要。

## 表格基本信息
- 文件名: {filename}
- 总行数: {total_rows}
- 总列数: {total_columns}

## 字段详情
{columns_desc}

## 样本数据（前 3 行）
{sample_data}

## 回答要求
1. 用**自然语言**直接回答用户的问题，不要生成代码、图表或 JSON。
2. 回答要简洁、有数据支撑。引用具体列名和可能的数值范围。
3. 如果需要做统计，可以根据样本数据和字段统计量（min/max/mean 等）给出合理推断。
4. 不要说"根据样本数据"或"我只能看到前3行"——直接给出有用的分析结论。
5. 用中文回答。
"""


def _build_prompt(schema: dict[str, Any]) -> str:
    """Render the legacy system prompt with schema metadata and sample rows."""
    columns_desc_lines: list[str] = []
    for col in schema["columns"]:
        extra = ""
        if col["missing_rate"] > 0:
            extra = f"  [缺失率 {col['missing_rate']:.1%}]"
        stats = col.get("stats", {})
        top_info = ""
        if "top_values" in stats:
            top_items = list(stats["top_values"].items())[:3]
            top_info = f"  常见值: {top_items}"
        columns_desc_lines.append(
            f"  - {col['name']} ({col['dtype']}){extra}{top_info}"
        )

    sample_rows = schema.get("sample_data", [])
    sample_str = json.dumps(sample_rows, ensure_ascii=False, indent=2) if sample_rows else "(无样本)"

    return _SYSTEM_PROMPT_TEMPLATE.format(
        filename=schema.get("filename", "unknown"),
        total_rows=schema["total_rows"],
        total_columns=schema["total_columns"],
        columns_desc="\n".join(columns_desc_lines),
        sample_data=sample_str,
    )


# =========================================================================
# New: Analysis pipeline prompts
# =========================================================================

# Import registry catalogue at module load time
def _get_functions_catalog() -> str:
    try:
        from .analysis_registry import generate_ai_functions_catalog, generate_chart_types_list
        catalog = generate_ai_functions_catalog()
        charts = generate_chart_types_list()
        return catalog + f"\n## 支持的图表类型\n{charts}\n"
    except Exception:
        return "(分析函数注册表暂不可用)"


_TRANSLATION_PROMPT = """\
你是一个数据分析系统的"概念翻译器"。你的任务是：
1. 识别用户问题中的模糊词语，给出操作性的精确定义
2. 将用户提到的业务概念映射到表中的实际列名

## 模糊词语翻译规则
- "top N"、"前几名" → N 默认为 10
- "高/低价值" → 默认 P80/P20 分位数
- "最近"、"近期" → 默认最近 30 天（若数据跨年则为最近 3 个月）
- "好/差/异常" → 偏离均值 2 个标准差
- "快速增长" → 增长率 > 均值 + 1个标准差
- "主要"、"大部分" → 累积占比 >= 80%
- "有关"、"影响" → Pearson |r| > 0.3 且 p < 0.05
- "显著差异" → ANOVA / Tukey HSD, p < 0.05

## 表格结构
{columns_desc}

## 样本数据
{sample_data}

## 用户问题
{question}

## 输出格式
输出一个严格的 JSON 对象，不要包含代码块标记或其他文本：
{{
  "translation": "用中文写出你对用户问题的理解，包括将模糊词翻译为操作性定义。例如：'你想找出消费总额排名前 10（即「top」定义为10）的客户（按 CustomerID 分组）。'",
  "column_mappings": [
    {{
      "concept": "用户说到的业务概念（如'销售额'）",
      "name": "映射后的列名（如'销售额'）",
      "type": "direct 或 computed",
      "column": "对应的原始列名（type=direct时）或 null（type=computed时）",
      "formula": "计算公式（type=computed时，如'UnitPrice * Quantity'）或 null（type=direct时）",
      "explanation": "简短解释映射理由"
    }}
  ]
}}

注意：
- column_mappings 中只包含用户问题中实际涉及的概念
- 如果概念直接对应某个列，type 为 "direct"
- 如果概念需要通过计算得到，type 为 "computed"，并在 formula 中给出 pandas eval 可用的表达式
"""

_ANALYSIS_PLANNING_PROMPT = """\
你是一个数据分析系统的"分析规划器"。用户已经提了一个问题，系统完成了概念翻译。现在你需要规划具体的分析步骤。

## 可用分析函数
{functions_catalog}

## 表格结构
{columns_desc}

## 用户问题
{question}

## 概念翻译结果
{translation}

## 输出格式
输出一个严格的 JSON 对象：
{{
  "plan_text": "用中文简述分析思路（1-3句话），让用户知道你要做什么",
  "actions": [
    {{
      "function": "analysis_function_name",
      "params": {{"param1": "value1", "param2": "value2"}},
      "chart": "chart_type",
      "reason": "为什么选择这个分析"
    }}
  ]
}}

## 规则
1. 每个问题选择 1-4 个分析函数，按逻辑顺序排列
2. 同一个问题尽量同时覆盖"整体"和"局部"（如：整体分布 + 分组比较）
3. chart 字段可以从支持的图表类型中选择，如果不确定就用函数的默认图表
4. params 中的列名必须使用映射后的列名（即 column_mappings 中的 name）
5. 如果问题无法用现有函数回答，请选择最接近的替代方案
"""

_INTERPRETATION_PROMPT = """\
你是一个数据分析系统的"结果解释器"。分析已经执行完毕，现在需要你用自然语言解读结果。

## 用户原始问题
{question}

## 概念翻译
{translation}

## 分析计划
{plan_text}

## 分析结果
{analysis_results}

## 输出要求
1. 用自然语言（中文）直接解读分析结果，不要生成代码或 JSON
2. 从数据中得出结论，而非仅描述数据
3. 指出值得注意的发现，包括异常或反直觉的点
4. 如果合适，给出 1-2 个建议的追问方向
5. 保持简洁：3-5 个自然段，每段 2-4 句

## 错误处理规则
- 如果分析结果中含"[列映射日志]"或"[列缺失提示]"，说明部分分析因列名不匹配而失败。请向用户解释具体哪些列缺失、可能的原因，并给出修正建议（如检查列名拼写、确认数据中是否有该字段）。
- 如果所有分析都失败了，请用"根据上传数据，无法完成本次分析，因为……"的格式诚实说明原因，不要编造结论。
"""


def build_translation_prompt(schema: dict[str, Any], question: str) -> str:
    """Build the concept-translation prompt."""
    columns_desc = _format_columns_desc(schema)
    sample_str = _format_sample(schema)
    return _TRANSLATION_PROMPT.format(
        columns_desc=columns_desc,
        sample_data=sample_str,
        question=question,
    )


def build_planning_prompt(schema: dict[str, Any], question: str,
                          translation_text: str) -> str:
    """Build the analysis-planning prompt."""
    columns_desc = _format_columns_desc(schema)
    catalog = _get_functions_catalog()
    return _ANALYSIS_PLANNING_PROMPT.format(
        functions_catalog=catalog,
        columns_desc=columns_desc,
        question=question,
        translation=translation_text,
    )


# =========================================================================
# Combined prompt: translation + planning in one call (faster)
# =========================================================================

_COMBINED_PROMPT = """\
你是一个数据分析系统。请同时完成概念翻译和分析规划，在一次回复中输出一个 JSON 对象。

## 第1步：理解数据

仔细阅读下方的"表格结构"和"样本数据"。你需要推断每一列的**业务含义**：
- 根据列名、数据类型、取值范围、高频值、样本值，判断每一列在实际业务中代表什么
- 例如：列名 "Qty" + 数值范围 1~100 + 样本 [6, 12, 24] → 很可能是"销售数量"
- 例如：列名 "UnitPrice" + 数值范围 -11062~38970 + 均值 4.6 → 很可能是"单价"，但存在负值异常
- 例如：列名 "Country" + 38 个唯一值 + 高频 "United Kingdom" → 很可能是"国家"

## 第2步：概念翻译

识别用户问题中的模糊词语并给出操作性定义。将用户问题中的每个**业务概念**匹配到表中的实际列名。

**模糊词语翻译规则：**
- "top N"、"前几名" → N 默认为 10
- "高/低价值" → 默认 P80/P20 分位数
- "最近"、"近期" → 默认最近 30 天（若数据跨年则为最近 3 个月）
- "好/差/异常" → 偏离均值 2 个标准差
- "快速增长" → 增长率 > 均值 + 1σ
- "主要"、"大部分" → 累积占比 >= 80%
- "有关"、"影响" → Pearson |r| > 0.3 且 p < 0.05
- "显著差异" → p < 0.05

**列映射规则（非常重要）：**
- 用户说的概念如果直接对应某列 → type="direct"，在 column 字段填写该列的原始列名
- 用户说的概念需要通过现有列计算得到 → type="computed"，在 formula 中给出 pandas eval 可用的表达式（如 "UnitPrice * Quantity"）
- **column_mappings 中必须包含 actions 中将要引用的每一个列名**。换句话说，actions 的 params 中出现的所有列名，都要先在 column_mappings 中声明

## 第3步：分析规划

确定能否基于现有数据回答用户的问题：
- **如果可以分析**：在 actions 中选择 1-4 个分析函数
- **如果无法分析**：actions 留空 []，在 plan_text 中清楚地解释原因，格式为"根据上传数据，无法……因为……"

从下列分析函数中选择（actions 不为空时）：

{functions_catalog}

## 表格结构
{columns_desc}

## 样本数据
{sample_data}

## 用户问题
{question}

## 之前的分析结果（如果有）
{previous_context}

## 输出格式（严格 JSON）
{{
  "translation": "对用户问题的操作性翻译（一句话，含定义）",
  "column_mappings": [
    {{
      "concept": "用户说到的业务概念（如'销售额'）",
      "name": "映射后的列名（作为 actions 中引用的列名，必须与 actions.params 中使用的列名完全一致）",
      "type": "direct 或 computed",
      "column": "type=direct 时填写原始列名（必须与表格结构中的列名完全一致），type=computed 时填 null",
      "formula": "type=computed 时填写 pandas eval 表达式（如 'UnitPrice * Quantity'），只能引用表格结构中的原始列名；type=direct 时填 null",
      "explanation": "简短解释映射理由"
    }}
  ],
  "plan_text": "若可分析：分析思路简述（1-2句）。若不可分析：根据上传数据，无法……因为……",
  "actions": [
    {{
      "function": "analysis_function_name",
      "params": {{"param1": "value1"}},
      "chart": "chart_type",
      "reason": "为什么需要这一步分析"
    }}
  ]
}}

## 关键约束（违反将导致分析失败）

1. **列名一致性**：actions 中 params 引用的每一个列名，必须作为 column_mappings 中某个条目的 `name` 字段出现。例如：如果 actions 中有 `{{"column": "销售额"}}`，则 column_mappings 中必须有一条 `{{"name": "销售额", ...}}`。
2. **computed 列必须声明**：如果某个列需要通过公式计算得到（如 销售额 = Quantity * UnitPrice），必须在 column_mappings 中声明 type="computed"，然后在 actions 中引用它的 name。
3. **formula 只能引用原始列名**：computed 列的 formula 只能使用表格结构中列出的原始列名，不能引用另一个 computed 列。
4. **无法分析时不要硬编**：如果用户问题涉及的概念在现有数据中完全无法找到对应的列，也无法通过现有列计算得到，请将 actions 置为 []，并在 plan_text 中诚实说明原因。例如："根据上传数据，无法分析利润率，因为数据中没有成本相关字段，也无法通过现有列（数量、单价、国家）计算得到。"
5. **优先尝试但不勉强**：如果概念和列之间的匹配不太确定，可以尝试最有把握的分析（如整体分布），同时在 plan_text 中说明不确定性的来源。
6. **直接输出 JSON**：不要用 ``` 代码块包裹，不要输出任何 JSON 以外的文本。"""


def build_combined_prompt(schema: dict[str, Any], question: str,
                          previous_context: str = "") -> str:
    """Build a combined translation+planning prompt for a single LLM call."""
    columns_desc = _format_columns_desc(schema)
    sample_str = _format_sample(schema)
    catalog = _get_functions_catalog()
    ctx = previous_context or "（无，这是第一次分析）"
    return _COMBINED_PROMPT.format(
        functions_catalog=catalog,
        columns_desc=columns_desc,
        sample_data=sample_str,
        question=question,
        previous_context=ctx,
    )


def build_interpretation_prompt(question: str, translation_text: str,
                                plan_text: str,
                                analysis_results_json: str) -> str:
    """Build the results-interpretation prompt."""
    return _INTERPRETATION_PROMPT.format(
        question=question,
        translation=translation_text,
        plan_text=plan_text,
        analysis_results=analysis_results_json,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_columns_desc(schema: dict[str, Any]) -> str:
    """Build a rich column description with sample values and semantic cues.

    The output helps the LLM infer *what* each column represents (e.g., "likely a
    quantity sold" vs "likely a unit price") so it can match user concepts to
    actual columns more accurately.
    """
    lines: list[str] = []
    sample_rows = schema.get("sample_data", [])

    for col in schema["columns"]:
        name = col["name"]
        dtype = col["dtype"]
        missing_rate = col.get("missing_rate", 0)
        unique_count = col.get("unique_values", 0)
        stats = col.get("stats", {})

        parts: list[str] = [f"  - **{name}**"]
        type_hint = {
            "numeric": "数值",
            "categorical": "分类",
            "datetime": "日期",
            "text": "文本",
            "boolean": "布尔",
            "identifier": "标识符",
        }.get(dtype, dtype)
        parts.append(f"({type_hint}")

        # Missing rate
        if missing_rate > 0:
            parts.append(f", 缺失率{missing_rate:.1%}")

        parts.append(")")

        # Unique count hint
        if dtype == "categorical":
            parts.append(f"  唯一值数: {unique_count}")

        # Numeric stats: range + mean
        if dtype == "numeric":
            if "min" in stats and "max" in stats:
                parts.append(f"  范围: {stats['min']} ~ {stats['max']}")
            if "mean" in stats:
                parts.append(f"  均值: {stats['mean']}")

        # Top values for categorical
        if "top_values" in stats and stats["top_values"]:
            top_items = list(stats["top_values"].items())[:5]
            parts.append(f"  高频值: {top_items}")

        # Sample values (from first 3 rows)
        if sample_rows:
            samples = []
            for row in sample_rows[:3]:
                val = row.get(name)
                if val is not None:
                    samples.append(str(val))
            if samples:
                unique_samples = list(dict.fromkeys(samples))[:3]
                parts.append(f"  样本: {unique_samples}")

        lines.append("".join(parts))

    return "\n".join(lines)


def _format_sample(schema: dict[str, Any]) -> str:
    sample_rows = schema.get("sample_data", [])
    if sample_rows:
        return json.dumps(sample_rows, ensure_ascii=False, indent=2)
    return "(无样本)"


# ---------------------------------------------------------------------------
# LLM API call (synchronous, legacy)
# ---------------------------------------------------------------------------


def _call_llm(
    system_prompt: str,
    user_query: str,
    credentials: dict[str, str],
    timeout: int = 30,
) -> str:
    """Synchronous chat-completions call.  Kept for backward compatibility."""
    api_key = credentials.get("api_key", "").strip()
    base_url = credentials.get("base_url", "").strip().rstrip("/")
    model_name = credentials.get("model_name", "").strip()

    if not api_key or not base_url or not model_name:
        raise ValueError("缺少 LLM 配置（API Key / Base URL / Model Name）。")

    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        "temperature": 0.3,
        "max_tokens": 2048,
    }

    try:
        import httpx
        try:
            resp = httpx.post(url, headers=headers, json=payload, timeout=timeout)
        except httpx.HTTPError as exc:
            raise ValueError(f"LLM 网络请求失败: {exc}") from exc

        if resp.status_code != 200:
            detail = resp.text[:500]
            raise ValueError(f"LLM API 返回错误 {resp.status_code}: {detail}")

        data = resp.json()
    except ImportError:
        import requests as req
        try:
            resp = req.post(url, headers=headers, json=payload, timeout=timeout)
        except req.RequestException as exc:
            raise ValueError(f"LLM 网络请求失败: {exc}") from exc

        if resp.status_code != 200:
            detail = resp.text[:500]
            raise ValueError(f"LLM API 返回错误 {resp.status_code}: {detail}")

        data = resp.json()

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        logger.warning("Unexpected LLM response shape: %s", str(data)[:300])
        raise ValueError(f"LLM 返回了非预期的数据结构: {exc}") from exc


# ---------------------------------------------------------------------------
# Streaming LLM API call (new)
# ---------------------------------------------------------------------------


async def _call_llm_stream(
    system_prompt: str,
    user_prompt: str,
    credentials: dict[str, str],
    timeout: int = 60,
) -> AsyncGenerator[str, None]:
    """Stream tokens from an OpenAI-compatible chat-completions API.

    Yields content fragments as they arrive.  The caller should concatenate them.
    """
    api_key = credentials.get("api_key", "").strip()
    base_url = credentials.get("base_url", "").strip().rstrip("/")
    model_name = credentials.get("model_name", "").strip()

    if not api_key or not base_url or not model_name:
        raise ValueError("缺少 LLM 配置")

    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 4096,
        "stream": True,
    }

    import httpx

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise ValueError(f"LLM API 返回错误 {resp.status_code}: {body[:500]}")

                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
    except httpx.HTTPError as exc:
        raise ValueError(f"LLM 流式请求失败: {exc}") from exc


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> dict[str, Any]:
    """Best-effort JSON extraction from LLM output.

    Handles common LLM output quirks:
    - Markdown code fences (```json ... ```)
    - Free text before/after JSON ("好的，以下是方案：{...}希望对你有帮助")
    - Trailing commas before closing braces
    """
    import re

    text = text.strip()

    # Step 1 — remove markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Step 2 — strip free text before first '{' and after last '}'
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    # Step 3 — remove trailing commas (common LLM mistake before } or ])
    text = re.sub(r",(\s*[}\]])", r"\1", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Step 4 — last resort: try to fix single-quoted JSON
    # (some LLMs use Python dict syntax instead of JSON)
    try:
        fixed = re.sub(r"'([^']*)':", r'"\1":', text)  # 'key': → "key":
        fixed = re.sub(r":\s*'([^']*)'", r': "\1"', fixed)  # : 'value' → : "value"
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Re-raise with context: head (for format check) + tail (for truncation check)
    if len(text) > 400:
        snippet = (
            f"前 200 字符: {text[:200]}\n"
            f"后 200 字符: ...{text[-200:]}\n"
            f"总长度: {len(text)} 字符"
        )
    else:
        snippet = f"完整文本 ({len(text)} 字符): {text}"
    raise ValueError(f"无法解析 LLM 返回的 JSON。\n{snippet}")
