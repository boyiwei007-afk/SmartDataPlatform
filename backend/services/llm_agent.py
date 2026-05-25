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

## 第1步：概念翻译
识别用户问题中的模糊词语并给出操作性定义。将概念映射到表中的实际列名。

**模糊词语翻译规则：**
- "top N"、"前几名" → N 默认为 10
- "高/低价值" → 默认 P80/P20 分位数
- "最近"、"近期" → 默认最近 30 天（若数据跨年则为最近 3 个月）
- "好/差/异常" → 偏离均值 2 个标准差
- "快速增长" → 增长率 > 均值 + 1σ
- "主要"、"大部分" → 累积占比 >= 80%
- "有关"、"影响" → Pearson |r| > 0.3 且 p < 0.05
- "显著差异" → p < 0.05

**列映射：** 用户说的概念可能对应实际列名（direct），也可能需要计算（computed，如"销售额=单价x数量"）。

## 第2步：分析规划
从下列分析函数中选择 1-4 个，按逻辑顺序排列。

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
  "column_mappings": [...],
  "plan_text": "分析思路简述（1-2句）",
  "actions": [...]
}}

## 重要规则
1. **优先尝试数据分析**：只要问题涉及数据，就尝试规划分析动作。如果现有分析函数无法覆盖，选择最接近的替代方案。
2. **actions 可以为空数组**：当问题完全是对话性的（如"这个分析结果意味着什么"、"为什么会出现这个现象"），不需要查新数据时，actions 置为 []。
3. 不要在 JSON 里混入自由文本回答——若 actions 为空，系统会自动切换为对话模式回答。

注意：直接输出 JSON，不要用 ``` 代码块包裹。"""


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
    lines: list[str] = []
    for col in schema["columns"]:
        extra = ""
        if col.get("missing_rate", 0) > 0:
            extra = f" [缺失率 {col['missing_rate']:.1%}]"
        stats = col.get("stats", {})
        if stats and "min" in stats:
            extra += f" [范围: {stats['min']} - {stats['max']}]"
        lines.append(f"  - {col['name']} ({col['dtype']}){extra}")
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
        "max_tokens": 1024,
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
        "max_tokens": 2048,
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
    """Best-effort JSON extraction from LLM output (may contain markdown fences)."""
    text = text.strip()
    # Remove markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    # Try to find JSON object boundaries
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    return json.loads(text)
