"""
SmartAnalysis Pro — Backend Entry Point

A FastAPI application that provides:
- File upload & data ingestion (CSV / Excel)
- Data cleaning & summarization services
- LLM proxy (Text-to-Pandas) with pluggable configuration
- Online ML training (scikit-learn) with weight extraction
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from services.data_engine import process_dataframe, get_advanced_analysis
from services.llm_agent import (
    ask_data,
    build_combined_prompt,
    build_interpretation_prompt,
    _call_llm,
    _call_llm_stream,
    _extract_json,
)
from services.ml_trainer import train_linear_regression
from services.data_profiler import profile_data
from services.data_cleaner import apply_cleaning
from services.analysis_functions import execute_analysis, apply_column_mappings
from services.auth import (
    register as auth_register,
    login as auth_login,
    list_users,
    verify_and_get_user,
    delete_user as auth_delete_user,
    record_upload,
    record_analysis,
    get_upload_history,
    get_analysis_history,
    delete_history_record,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SmartAnalysis Pro API",
    description="智能问数与销售模拟沙盘 — 后端服务",
    version="0.1.0",
)

# Directory for uploaded files (relative to the backend package)
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Session context: stores previous analysis results per file for multi-turn conversations
# Limited to 20 most-recently-used files to prevent unbounded memory growth.
_session_context: dict[str, str] = {}
_MAX_CONTEXT_FILES = 20


def _set_context(key: str, value: str) -> None:
    """Store analysis context with LRU eviction."""
    _session_context[key] = value
    while len(_session_context) > _MAX_CONTEXT_FILES:
        oldest = next(iter(_session_context))
        del _session_context[oldest]


# ---------------------------------------------------------------------------
# Utility: fast DataFrame loading (CSV cache)
# ---------------------------------------------------------------------------

def _ensure_csv_cache(file_path: Path) -> Path:
    """If *file_path* is an Excel file, create a CSV sibling for fast reads."""
    if file_path.suffix.lower() in (".xls", ".xlsx"):
        csv_path = file_path.with_suffix(".csv")
        if not csv_path.exists():
            try:
                df = pd.read_excel(file_path)
                df.to_csv(csv_path, index=False)
                logger.info("Cached CSV: %s", csv_path.name)
            except Exception:
                logger.warning("Failed to cache CSV for %s", file_path.name)
        return csv_path
    return file_path


def _load_dataframe(filename: str) -> pd.DataFrame:
    """Load a previously-uploaded file, preferring the CSV cache."""
    path = UPLOAD_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"文件 '{filename}' 不存在")
    # Prefer CSV if available
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    if path.suffix.lower() in (".xls", ".xlsx"):
        return pd.read_excel(path)
    return pd.read_csv(path)

# ---------------------------------------------------------------------------
# CORS — allow all origins during development
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health-check / ping
# ---------------------------------------------------------------------------
@app.get("/ping", tags=["system"])
async def ping():
    """Lightweight health-check endpoint."""
    return {"status": "ok", "service": "SmartAnalysis Pro API", "version": "0.1.0"}


# ---------------------------------------------------------------------------
# Authentication dependency
# ---------------------------------------------------------------------------
from fastapi import Header


def get_current_user(authorization: str = Header(None)) -> dict[str, Any] | None:
    """FastAPI dependency: extract and verify Bearer token. Returns user dict or None."""
    if not authorization:
        return None
    parts = authorization.split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return verify_and_get_user(parts[1])


# ---------------------------------------------------------------------------
# Pydantic schemas — auth
# ---------------------------------------------------------------------------

class AuthRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

@app.post("/register", tags=["auth"])
async def register_user(req: AuthRequest):
    """Register a new user account."""
    result = auth_register(req.username, req.password)
    if not result["ok"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/login", tags=["auth"])
async def login_user(req: AuthRequest):
    """Login and receive a JWT token."""
    result = auth_login(req.username, req.password)
    if not result["ok"]:
        raise HTTPException(status_code=401, detail=result["error"])
    return result


@app.get("/users", tags=["auth"])
async def get_users(current_user: dict = Depends(get_current_user)):
    """Admin: list all registered users."""
    if not current_user:
        raise HTTPException(status_code=401, detail="请先登录")
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="仅管理员可查看")
    return {"users": list_users()}


@app.delete("/users/{username}", tags=["auth"])
async def remove_user(username: str,
                      current_user: dict = Depends(get_current_user)):
    """Delete a user account. Admin can delete anyone; users can delete themselves."""
    if not current_user:
        raise HTTPException(status_code=401, detail="请先登录")
    # Admin can delete any user; regular users can only self-delete
    is_admin = current_user.get("role") == "admin"
    is_self = current_user.get("username") == username
    if not is_admin and not is_self:
        raise HTTPException(status_code=403, detail="仅可注销自己的账号")
    result = auth_delete_user(username)
    if not result["ok"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.get("/history/uploads", tags=["history"])
async def history_uploads(current_user: dict = Depends(get_current_user)):
    """Get current user's upload history."""
    if not current_user:
        raise HTTPException(status_code=401, detail="请先登录")
    return {"uploads": get_upload_history(current_user["username"])}


@app.get("/history/analysis", tags=["history"])
async def history_analysis(current_user: dict = Depends(get_current_user)):
    """Get current user's analysis history."""
    if not current_user:
        raise HTTPException(status_code=401, detail="请先登录")
    return {"analysis": get_analysis_history(current_user["username"])}


@app.get("/admin/history/uploads", tags=["history"])
async def admin_all_uploads(current_user: dict = Depends(get_current_user)):
    """Admin: view all users' upload history."""
    if not current_user or current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="仅管理员可查看")
    return {"uploads": get_upload_history(None)}


@app.get("/admin/history/analysis", tags=["history"])
async def admin_all_analysis(current_user: dict = Depends(get_current_user)):
    """Admin: view all users' analysis history."""
    if not current_user or current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="仅管理员可查看")
    return {"analysis": get_analysis_history(None)}


@app.delete("/admin/history/uploads/{record_id}", tags=["history"])
async def admin_delete_upload(record_id: int,
                               current_user: dict = Depends(get_current_user)):
    """Admin: delete a single upload history record."""
    if not current_user or current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="仅管理员可操作")
    if not delete_history_record("upload_history", record_id):
        raise HTTPException(status_code=404, detail="记录不存在")
    return {"ok": True}


@app.delete("/admin/history/analysis/{record_id}", tags=["history"])
async def admin_delete_analysis(record_id: int,
                                 current_user: dict = Depends(get_current_user)):
    """Admin: delete a single analysis history record."""
    if not current_user or current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="仅管理员可操作")
    if not delete_history_record("analysis_history", record_id):
        raise HTTPException(status_code=404, detail="记录不存在")
    return {"ok": True}


# ---------------------------------------------------------------------------
# File upload & schema extraction
# ---------------------------------------------------------------------------
@app.post("/upload", tags=["data"])
async def upload_file(file: UploadFile = File(...),
                      current_user: dict = Depends(get_current_user)):
    """Accept a CSV or Excel file, persist it, and return a structural schema."""
    if not current_user:
        raise HTTPException(status_code=401, detail="请先登录")
    # --- validate extension ------------------------------------------------
    filename = file.filename or "unknown"
    suffix = Path(filename).suffix.lower()
    if suffix not in (".csv", ".xls", ".xlsx"):
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式 '{suffix}'。请上传 CSV 或 Excel 文件。",
        )

    # --- validate file size (50 MB) ---------------------------------------
    MAX_BYTES = 50 * 1024 * 1024
    file.file.seek(0, 2)  # seek to end
    fsize = file.file.tell()
    file.file.seek(0)     # seek back to start
    if fsize > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大（{fsize / 1024 / 1024:.1f} MB），上限为 50 MB。请压缩文件或拆分后重新上传。",
        )

    # --- persist to disk ---------------------------------------------------
    save_path = UPLOAD_DIR / filename
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info("File saved: %s (%.1f KB)", filename, save_path.stat().st_size / 1024)
    except Exception as exc:
        logger.exception("Failed to save uploaded file")
        raise HTTPException(status_code=500, detail=f"文件保存失败: {exc}")

    # --- cache a CSV copy for fast subsequent reads ------------------------
    _ensure_csv_cache(save_path)

    # --- process & return schema -------------------------------------------
    try:
        schema = process_dataframe(str(save_path))
        logger.info(
            "Schema extracted: %d rows x %d cols (%s)",
            schema["total_rows"],
            schema["total_columns"],
            filename,
        )
        record_upload(current_user["username"], filename, filename)
        return schema
    except ValueError as exc:
        logger.warning("Data engine rejected file: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error during schema extraction")
        raise HTTPException(status_code=500, detail=f"数据处理异常: {exc}")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Payload sent by the frontend for each conversational turn."""

    query: str = Field(..., description="用户自然语言提问", min_length=1)
    table_schema: dict = Field(..., description="前端缓存的表结构（含列信息和样本数据）")
    api_key: str = Field(..., description="LLM API Key（阅后即焚）")
    base_url: str = Field(..., description="LLM Base URL")
    model_name: str = Field(..., description="LLM Model Name")


class ChatResponse(BaseModel):
    """Plain-text answer returned to the frontend."""

    answer: str = ""
    error: str | None = None


# ---------------------------------------------------------------------------
# Chat / LLM proxy endpoint
# ---------------------------------------------------------------------------
@app.post("/chat", tags=["chat"], response_model=ChatResponse)
async def chat_query(req: ChatRequest):
    """Natural-language Q&A about the uploaded table.

    The LLM receives only the table schema (column names, types, stats,
    and a few sample rows) — **not** the full dataset.  It answers in
    plain natural language without generating code or charts.
    """
    credentials = {
        "api_key": req.api_key,
        "base_url": req.base_url,
        "model_name": req.model_name,
    }

    try:
        result = ask_data(
            query=req.query,
            schema=req.table_schema,
            api_credentials=credentials,
        )
        return ChatResponse(answer=result["answer"])
    except ValueError as exc:
        logger.warning("LLM agent error: %s", exc)
        return ChatResponse(answer="", error=str(exc))
    except Exception as exc:
        logger.exception("Unexpected /chat error")
        raise HTTPException(status_code=500, detail=f"对话服务异常: {exc}")


# ---------------------------------------------------------------------------
# Pydantic schemas — training
# ---------------------------------------------------------------------------

class TrainRequest(BaseModel):
    """Payload for the model-training endpoint."""

    filename: str = Field(..., description="当前已上传的文件名")
    target_col: str = Field(..., description="目标变量 Y 的列名", min_length=1)
    feature_cols: list[str] = Field(..., description="特征变量 X 的列名列表", min_length=1)


class TrainResponse(BaseModel):
    """Training result returned to the frontend."""

    target: str
    features: list[str]
    intercept: float
    coefficients: dict[str, float]
    r2_score: float
    feature_stats: dict[str, dict[str, float]] = {}
    n_samples: int
    n_dropped: int
    n_imputed: int = 0
    error: str | None = None
    diagnostics: dict[str, Any] | None = None
    feature_importance: dict[str, dict[str, float]] | None = None
    importance_note: str | None = None
    target_stats: dict[str, float] | None = None
    warnings: list[str] | None = None


# ---------------------------------------------------------------------------
# Training preview endpoint (lightweight, no model fitting)
# ---------------------------------------------------------------------------
@app.post("/preview-train", tags=["train"])
async def preview_train(req: TrainRequest,
                        current_user: dict = Depends(get_current_user)):
    """Quick preview of training data quality before actual training.

    Returns effective sample count, NaN rates, and warnings about
    non-numeric features — so users know what to expect before clicking
    "Start Training".
    """
    if not current_user:
        raise HTTPException(status_code=401, detail="请先登录")
    file_path = UPLOAD_DIR / req.filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"文件 '{req.filename}' 不存在。")

    try:
        df = _load_dataframe(req.filename)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"无法读取数据文件: {exc}")

    target = req.target_col.strip()
    feats = [c.strip() for c in req.feature_cols if c.strip()]

    # Basic validation
    all_cols = set(df.columns)
    missing = [c for c in ([target] + feats) if c not in all_cols]
    if missing:
        raise HTTPException(status_code=400, detail=f"列 {missing} 在数据中不存在。")

    # Run numeric coercion
    sub = df[[target] + feats].copy()
    total_rows = len(sub)
    bad_feats: list[str] = []

    for col in [target] + feats:
        try:
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
        except TypeError as exc:
            raise HTTPException(status_code=400, detail=f"列 '{col}' 无法转换为数值: {exc}")
        if col in feats and sub[col].isna().all():
            bad_feats.append(col)

    # Count Y NaN
    y_nan = int(sub[target].isna().sum())
    effective = total_rows - y_nan

    # Per-feature NaN rates
    feat_preview: dict[str, dict] = {}
    for col in feats:
        nan_count = int(sub[col].isna().sum())
        feat_preview[col] = {
            "nan_count": nan_count,
            "nan_pct": round(nan_count / total_rows * 100, 2) if total_rows else 0,
            "all_nan": bool(sub[col].isna().all()),
        }

    return {
        "total_rows": total_rows,
        "effective_rows": effective,
        "y_dropped": y_nan,
        "y_dropped_pct": round(y_nan / total_rows * 100, 2) if total_rows else 0,
        "features": feat_preview,
        "bad_features": bad_feats,
        "warnings": [
            f"'{f}' 不是数值列，将被排除" for f in bad_feats
        ],
    }


# ---------------------------------------------------------------------------
# Model training endpoint
# ---------------------------------------------------------------------------
@app.post("/train", tags=["train"], response_model=TrainResponse)
async def train_model(req: TrainRequest,
                      current_user: dict = Depends(get_current_user)):
    """Train a linear regression model on user-selected columns."""
    if not current_user:
        raise HTTPException(status_code=401, detail="请先登录")
    file_path = UPLOAD_DIR / req.filename
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"文件 '{req.filename}' 不存在，请先上传数据。",
        )

    try:
        df = _load_dataframe(req.filename)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"无法读取数据文件: {exc}")

    try:
        result = train_linear_regression(
            df=df,
            target_col=req.target_col,
            feature_cols=req.feature_cols,
        )
        return TrainResponse(
            target=result["target"],
            features=result["features"],
            intercept=result["intercept"],
            coefficients=result["coefficients"],
            r2_score=result["r2_score"],
            feature_stats=result["feature_stats"],
            n_samples=result["n_samples"],
            n_dropped=result["n_dropped"],
            n_imputed=result["n_imputed"],
            diagnostics=result.get("diagnostics"),
            feature_importance=result.get("feature_importance"),
            importance_note=result.get("importance_note"),
            target_stats=result.get("target_stats"),
            warnings=result.get("warnings"),
        )
    except ValueError as exc:
        logger.warning("Training validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected /train error")
        raise HTTPException(status_code=500, detail=f"模型训练异常: {exc}")


# ---------------------------------------------------------------------------
# Pydantic schemas — advanced analysis
# ---------------------------------------------------------------------------

class AdvancedAnalysisRequest(BaseModel):
    """Payload for the automated EDA endpoint."""

    filename: str = Field(..., description="当前已上传的文件名")


# ---------------------------------------------------------------------------
# Advanced analysis endpoint
# ---------------------------------------------------------------------------
@app.post("/advanced-analysis", tags=["analysis"])
async def advanced_analysis(req: AdvancedAnalysisRequest):
    """Run automated exploratory data analysis on the uploaded file.

    Returns chart-ready data for histograms, correlation heatmap,
    categorical pie charts, and time-series line charts.
    All heavy computations are sampled to ensure responsiveness
    even on 500k+ row datasets.
    """
    file_path = UPLOAD_DIR / req.filename
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"文件 '{req.filename}' 不存在，请先上传数据。",
        )

    try:
        result = get_advanced_analysis(str(file_path))
        return result
    except ValueError as exc:
        logger.warning("Advanced analysis error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected /advanced-analysis error")
        raise HTTPException(status_code=500, detail=f"数据分析异常: {exc}")


# ---------------------------------------------------------------------------
# Pydantic schemas — data profile
# ---------------------------------------------------------------------------

class DataProfileRequest(BaseModel):
    """Payload for the data-profiling endpoint."""
    filename: str = Field(..., description="当前已上传的文件名")


# ---------------------------------------------------------------------------
# Data profile endpoint (Tab 2 — stage 2a)
# ---------------------------------------------------------------------------
@app.post("/data-profile", tags=["analysis"])
async def data_profile(req: DataProfileRequest):
    """Generate a comprehensive data profile for the uploaded file.

    Returns per-column distribution stats, sparkline data, outlier
    counts, and overall quality metrics.
    """
    file_path = UPLOAD_DIR / req.filename
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"文件 '{req.filename}' 不存在，请先上传数据。",
        )
    try:
        result = profile_data(str(file_path))
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected /data-profile error")
        raise HTTPException(status_code=500, detail=f"数据画像异常: {exc}")


# ---------------------------------------------------------------------------
# Pydantic schemas — data cleaning
# ---------------------------------------------------------------------------

class CleanDataRequest(BaseModel):
    """Payload for the data-cleaning endpoint."""
    filename: str = Field(..., description="当前已上传的文件名")
    operations: list[dict[str, Any]] = Field(
        ..., description="清洗操作列表，每项含 type 及其他参数"
    )


# ---------------------------------------------------------------------------
# Data cleaning endpoint (Tab 2 — stage 2b)
# ---------------------------------------------------------------------------
@app.post("/clean-data", tags=["analysis"])
async def clean_data(req: CleanDataRequest):
    """Apply user-confirmed cleaning operations and return the cleaned schema."""
    file_path = UPLOAD_DIR / req.filename
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"文件 '{req.filename}' 不存在。",
        )
    try:
        result = apply_cleaning(req.filename, req.operations)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected /clean-data error")
        raise HTTPException(status_code=500, detail=f"数据清洗异常: {exc}")


# ---------------------------------------------------------------------------
# Pydantic schemas — AI analysis
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    """Payload for the AI-assisted analysis endpoint."""
    question: str = Field(..., description="用户自然语言提问", min_length=1)
    filename: str = Field(..., description="当前已上传的文件名")
    table_schema: dict = Field(..., description="表结构元数据")
    llm_config: dict = Field(..., description="{api_key, base_url, model_name}")


# ---------------------------------------------------------------------------
# AI analysis endpoint (Tab 3 — SSE streaming, 4 stages)
# ---------------------------------------------------------------------------
@app.post("/analyze", tags=["analysis"])
async def analyze(req: AnalyzeRequest,
                  current_user: dict = Depends(get_current_user)):
    """AI-assisted exploratory analysis with SSE streaming."""
    if not current_user:
        raise HTTPException(status_code=401, detail="请先登录")
    file_path = UPLOAD_DIR / req.filename
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"文件 '{req.filename}' 不存在，请先上传数据。",
        )

    credentials = {
        "api_key": req.llm_config.get("api_key", ""),
        "base_url": req.llm_config.get("base_url", ""),
        "model_name": req.llm_config.get("model_name", ""),
    }

    # Load data once
    try:
        df = _load_dataframe(req.filename)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"无法读取数据文件: {exc}")

    async def event_generator():
        translation_text = ""
        plan_text = ""
        column_mappings: list[dict] = []
        actions: list[dict] = []
        all_results: list[dict] = []

        # ---- Stage 0+1 combined: Translation + Planning in ONE LLM call ----
        yield _sse_event("status", {"stage": "analyzing", "message": "正在理解问题并规划分析..."})
        previous_context = _session_context.get(req.filename, "")
        try:
            combined_prompt = build_combined_prompt(req.table_schema, req.question, previous_context)
            # Stream the LLM output — push plan text as it arrives, collect full JSON
            buffer = ""
            async for token in _call_llm_stream(combined_prompt, "", credentials, timeout=60):
                buffer += token
                # Try to extract partial plan_text for early display
                # Look for "plan_text" field in partially-arrived JSON
                import re
                plan_match = re.search(r'"plan_text"\s*:\s*"([^"]*)', buffer)
                if plan_match:
                    partial_plan = plan_match.group(1)
                    if len(partial_plan) > len(plan_text):
                        plan_text = partial_plan
                        yield _sse_event("status", {
                            "stage": "analyzing",
                            "message": partial_plan[:120],
                        })
                # Also try to extract translation early
                trans_match = re.search(r'"translation"\s*:\s*"([^"]*)', buffer)
                if trans_match:
                    partial_trans = trans_match.group(1)
                    if len(partial_trans) > len(translation_text):
                        translation_text = partial_trans
                        yield _sse_event("translation", {
                            "text": partial_trans,
                            "column_mappings": [],
                        })

            # Parse the full JSON response
            combined_json = _extract_json(buffer)
            translation_text = combined_json.get("translation", "")
            column_mappings = combined_json.get("column_mappings", [])
            plan_text = combined_json.get("plan_text", "")
            actions = combined_json.get("actions", [])
        except Exception as exc:
            logger.warning("Combined planning failed: %s", exc)
            yield _sse_event("error", {
                "message": f"LLM 调用失败: {exc}。请检查 API Key 和网络连接。",
                "stage": "planning",
            })
            translation_text = req.question
            column_mappings = []
            plan_text = ""
            actions = []

        # ---- No analysis actions: fall back to conversational Q&A ----
        if not actions or not isinstance(actions, list):
            yield _sse_event("translation", {
                "text": translation_text or req.question,
                "column_mappings": [],
            })
            yield _sse_event("plan", {"plan_text": "（对话模式）", "actions": []})
            try:
                # Build enriched query with previous context
                enriched_query = req.question
                if previous_context:
                    enriched_query = f"之前的分析结果:\n{previous_context}\n\n用户新问题: {req.question}\n\n请基于之前的分析结果回答用户的新问题。"
                chat_result = ask_data(enriched_query, req.table_schema, credentials)
                answer = chat_result.get("answer", "")
                yield _sse_event("interpretation", {"text": answer})
                _set_context(req.filename, f"Q: {req.question}\nA: {answer[:300]}")
                record_analysis(current_user["username"], req.question, answer[:500], req.filename)
            except Exception as exc:
                yield _sse_event("interpretation", {"text": f"(回答暂时不可用: {exc})"})
            yield _sse_event("done", {"session_id": req.filename})
            return

        yield _sse_event("translation", {
            "text": translation_text,
            "column_mappings": column_mappings,
        })
        yield _sse_event("plan", {
            "plan_text": plan_text,
            "actions": actions,
        })

        # Apply column mappings (guarded) — use a separate var to avoid shadowing closure 'df'
        analysis_df = df
        mapping_logs: list[str] = []
        try:
            if isinstance(column_mappings, list) and len(column_mappings) > 0:
                analysis_df, mapping_logs = apply_column_mappings(df, column_mappings)
                logger.info("Mappings applied: %s", mapping_logs)
        except Exception as exc:
            logger.exception("Column mapping crash")
            yield _sse_event("error", {
                "message": f"列映射处理失败: {exc}",
                "stage": "mapping",
            })

        # ---- Validate all column references before execution ----
        actual_columns = set(str(c) for c in analysis_df.columns)
        actual_columns = set(str(c) for c in analysis_df.columns)
        # Build a whitelist of params whose values are NOT column names.
        # We do this by consulting the registry's input_schema: any param whose
        # schema description does NOT contain "列名" is not checked as a column.
        try:
            from services.analysis_registry import ANALYSIS_REGISTRY
        except Exception:
            ANALYSIS_REGISTRY = {}

        def _column_param_names(func_name: str) -> set[str]:
            """Return the subset of param names that are column references."""
            entry = ANALYSIS_REGISTRY.get(func_name, {})
            schema = entry.get("input_schema", {})
            col_params: set[str] = set()
            for pname, pdesc in schema.items():
                if "列" in pdesc:  # "数值列名", "分类列名", "默认所有数值列", etc.
                    col_params.add(pname)
            return col_params

        def _collect_missing_columns(func_name: str, params: dict) -> list[str]:
            """Scan *params* for column-name values not present in *actual_columns*."""
            col_params = _column_param_names(func_name)
            missing: list[str] = []
            for key, val in params.items():
                if key not in col_params:
                    continue  # not a column param, skip
                if isinstance(val, str):
                    if val not in actual_columns:
                        missing.append(val)
                elif isinstance(val, list):
                    for item in val:
                        if isinstance(item, str) and item not in actual_columns:
                            missing.append(item)
            return missing

        # Pre-validate and collect missing-column info for the interpretation stage
        _missing_col_info: list[str] = []  # human-readable messages for the LLM

        # ---- Stage 2: Execution ----
        total_actions = len(actions)
        logger.info("Executing %d analysis actions", total_actions)
        for idx, action in enumerate(actions):
            if not isinstance(action, dict):
                continue
            func_name = str(action.get("function", ""))
            if not func_name:
                continue
            params = action.get("params", {})
            if not isinstance(params, dict):
                params = {}
            chart_override = str(action.get("chart", ""))

            # --- Column-name guard ---
            missing_cols = _collect_missing_columns(func_name, params)
            if missing_cols:
                available_hint = ", ".join(sorted(actual_columns))
                msg = (
                    f"列 {missing_cols} 在数据中不存在。"
                    f"当前可用列: [{available_hint}]"
                )
                _missing_col_info.append(f"[{func_name}] {msg}")
                logger.warning("Column guard blocked '%s': %s", func_name, msg)
                yield _sse_event("result", {
                    "step": idx + 1,
                    "function": func_name,
                    "error": msg,
                    "chart_type": chart_override or "table",
                    "reason": str(action.get("reason", "")),
                })
                continue

            yield _sse_event("progress", {
                "step": idx + 1,
                "total": total_actions,
                "message": f"正在执行: {func_name}...",
            })

            try:
                result = execute_analysis(func_name, analysis_df, params)
                # Apply chart override from AI plan, or fall back to registry default
                if chart_override:
                    result["chart_type"] = chart_override
                elif result.get("chart_type") == "table" and "error" in result:
                    # Try to get default chart from registry
                    try:
                        from services.analysis_registry import ANALYSIS_REGISTRY
                        entry = ANALYSIS_REGISTRY.get(func_name, {})
                        result["chart_type"] = entry.get("default_chart", "table")
                    except Exception:
                        pass
                result["step"] = idx + 1
                result["function"] = func_name
                result["reason"] = str(action.get("reason", ""))
                all_results.append(result)
                logger.info("Action %d/%d '%s' done, chart=%s", idx+1, total_actions, func_name, result.get("chart_type"))
                yield _sse_event("result", result)
            except Exception as exc:
                logger.exception("Action '%s' failed", func_name)
                fallback_chart = "table"
                if chart_override:
                    fallback_chart = chart_override
                yield _sse_event("result", {
                    "step": idx + 1,
                    "function": func_name,
                    "error": f"分析执行失败: {exc}",
                    "chart_type": fallback_chart,
                })

        # ---- Stage 3: Interpretation ----
        if all_results:
            logger.info("Starting interpretation for %d results", len(all_results))
            yield _sse_event("status", {"stage": "interpretation", "message": "正在解读结果..."})
            try:
                # Build a compact summary of results for the LLM
                summary_parts: list[str] = []
                # Prepend mapping diagnostics so the LLM can explain column issues
                if mapping_logs:
                    summary_parts.append("[列映射日志] " + "; ".join(mapping_logs))
                if _missing_col_info:
                    summary_parts.append("[列缺失提示] " + "; ".join(_missing_col_info))
                for r in all_results:
                    if "error" in r:
                        summary_parts.append(f"[{r.get('function')}] 执行失败: {r['error']}")
                    else:
                        stats_str = json.dumps(r.get("stats", {}), ensure_ascii=False)
                        summary_parts.append(
                            f"[{r.get('function')}] chart={r.get('chart_type')} "
                            f"stats={stats_str}"
                        )
                results_summary = "\n".join(summary_parts)

                interp_prompt = build_interpretation_prompt(
                    req.question, translation_text, plan_text, results_summary
                )
                # Stream the interpretation
                buffer = ""
                async for token in _call_llm_stream(interp_prompt, "", credentials, timeout=60):
                    buffer += token
                    yield _sse_event("interpretation", {"text": token})
            except Exception as exc:
                logger.warning("Interpretation failed: %s", exc)
                yield _sse_event("error", {
                    "message": f"AI 解读失败: {exc}",
                    "stage": "interpretation",
                })
                yield _sse_event("interpretation", {
                    "text": f"(AI 解读暂时不可用: {exc})"
                })

        # Save context for multi-turn conversation
        context_parts = [f"Q: {req.question}"]
        if translation_text:
            context_parts.append(f"理解: {translation_text}")
        for r in all_results:
            if "error" not in r:
                context_parts.append(
                    f"[{r.get('function','')}] {json.dumps(r.get('stats',{}), ensure_ascii=False)[:300]}"
                )
        _set_context(req.filename, "\n".join(context_parts[-6:]))

        # Record analysis history
        try:
            summary = json.dumps({"question": req.question, "actions": [r.get("function","") for r in all_results]}, ensure_ascii=False)
            record_analysis(current_user["username"], req.question, summary, req.filename)
        except Exception:
            pass

        yield _sse_event("done", {"session_id": req.filename})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------

class _SafeEncoder(json.JSONEncoder):
    def default(self, o):
        return str(o)
    def encode(self, o):
        return super().encode(self._sanitise(o))
    def _sanitise(self, o):
        if isinstance(o, float):
            if o != o or o == float('inf') or o == float('-inf'):
                return None
            return o
        if isinstance(o, dict):
            return {k: self._sanitise(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [self._sanitise(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            val = float(o)
            return None if (val != val or val == float('inf') or val == float('-inf')) else val
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.ndarray,)):
            return self._sanitise(o.tolist())
        if isinstance(o, (pd.Timestamp,)):
            return o.isoformat()
        return o


def _sse_event(event: str, data: dict) -> str:
    """Format a dict as an SSE event string, with NaN/Inf -> null protection."""
    safe = json.dumps(data, ensure_ascii=False, cls=_SafeEncoder)
    return f"event: {event}\ndata: {safe}\n\n"
