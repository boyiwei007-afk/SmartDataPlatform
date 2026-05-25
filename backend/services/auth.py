"""
Authentication module — SQLite-backed user store with JWT tokens.

Database: backend/auth.db (auto-created)
Tables: users(id, username, password_hash, salt, role, created_at)

Built-in accounts:
  - admin  / @wby@1235789  (role: admin)
  - test   / 123456         (role: user)
"""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database path
# ---------------------------------------------------------------------------
_DB_PATH = Path(__file__).resolve().parent.parent / "auth.db"

# ---------------------------------------------------------------------------
# JWT constants (simple HMAC-based — no external JWT lib needed)
# ---------------------------------------------------------------------------
_JWT_SECRET = "smartanalysis-jwt-secret-key-2025"  # In production, use env var
_TOKEN_EXPIRE_SECONDS = 86400 * 7  # 7 days

# ---------------------------------------------------------------------------
# Built-in accounts
# ---------------------------------------------------------------------------
_BUILTIN_USERS = [
    {"username": "admin", "password": "@wby@1235789", "role": "admin"},
    {"username": "test",  "password": "123456",       "role": "user"},
]

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _init_db() -> None:
    """Create tables and seed built-in accounts if first run."""
    conn = _get_conn()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT    UNIQUE NOT NULL,
                password_hash TEXT    NOT NULL,
                salt          TEXT    NOT NULL,
                role          TEXT    NOT NULL DEFAULT 'user',
                created_at    TEXT    NOT NULL
            )
        """)
        # Seed built-in accounts
        for u in _BUILTIN_USERS:
            existing = conn.execute(
                "SELECT id FROM users WHERE username = ?", (u["username"],)
            ).fetchone()
            if not existing:
                salt = _gen_salt()
                pw_hash = _hash_password(u["password"], salt)
                conn.execute(
                    "INSERT INTO users (username, password_hash, salt, role, created_at) VALUES (?,?,?,?,?)",
                    (u["username"], pw_hash, salt, u["role"], _now()),
                )
                logger.info("Seeded built-in user: %s (role=%s)", u["username"], u["role"])
        # History tables
        conn.execute("""
            CREATE TABLE IF NOT EXISTS upload_history (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT    NOT NULL,
                filename      TEXT    NOT NULL,
                original_name TEXT    NOT NULL,
                uploaded_at   TEXT    NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT    NOT NULL,
                filename      TEXT    NOT NULL DEFAULT '',
                question      TEXT    NOT NULL,
                result_json   TEXT,
                created_at    TEXT    NOT NULL
            )
        """)
        # Migration: add filename column if missing (for existing DB)
        try:
            conn.execute("ALTER TABLE analysis_history ADD COLUMN filename TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError:
            pass  # column already exists
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Password helpers
# ---------------------------------------------------------------------------


def _gen_salt() -> str:
    return secrets.token_hex(16)


def _hash_password(password: str, salt: str) -> str:
    """PBKDF2-SHA256 with 100k iterations."""
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000
    ).hex()


def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


# ---------------------------------------------------------------------------
# Token helpers (simple HMAC-based JWT)
# ---------------------------------------------------------------------------


def _b64_encode(data: bytes) -> str:
    import base64
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64_decode(s: str) -> bytes:
    import base64
    padding = 4 - len(s) % 4
    if padding != 4:
        s += "=" * padding
    return base64.urlsafe_b64decode(s)


def create_token(username: str, role: str) -> str:
    """Create a simple JWT-style token: header.payload.signature."""
    header = _b64_encode(b'{"alg":"HS256","typ":"JWT"}')
    payload_dict = {
        "username": username,
        "role": role,
        "iat": int(time.time()),
        "exp": int(time.time()) + _TOKEN_EXPIRE_SECONDS,
    }
    import json
    payload = _b64_encode(json.dumps(payload_dict).encode())
    msg = f"{header}.{payload}"
    sig = hashlib.sha256(f"{msg}.{_JWT_SECRET}".encode()).hexdigest()
    return f"{msg}.{sig}"


def verify_token(token: str) -> dict[str, Any] | None:
    """Verify token and return payload dict, or None if invalid/expired."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header, payload, sig = parts
        msg = f"{header}.{payload}"
        expected_sig = hashlib.sha256(f"{msg}.{_JWT_SECRET}".encode()).hexdigest()
        if sig != expected_sig:
            return None
        import json
        data = json.loads(_b64_decode(payload))
        if data.get("exp", 0) < time.time():
            return None
        return data
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def register(username: str, password: str) -> dict[str, Any]:
    """Register a new user. Returns {ok: true, user: {...}} or {ok: false, error: ...}."""
    username = username.strip()
    if not username or len(username) < 2:
        return {"ok": False, "error": "用户名至少 2 个字符"}
    if not password or len(password) < 4:
        return {"ok": False, "error": "密码至少 4 个字符"}

    conn = _get_conn()
    try:
        existing = conn.execute(
            "SELECT id FROM users WHERE username = ?", (username,)
        ).fetchone()
        if existing:
            return {"ok": False, "error": "用户名已存在"}

        salt = _gen_salt()
        pw_hash = _hash_password(password, salt)
        conn.execute(
            "INSERT INTO users (username, password_hash, salt, role, created_at) VALUES (?,?,?,?,?)",
            (username, pw_hash, salt, "user", _now()),
        )
        conn.commit()
        logger.info("New user registered: %s", username)
        return {"ok": True, "user": {"username": username, "role": "user"}}
    finally:
        conn.close()


def login(username: str, password: str) -> dict[str, Any]:
    """Validate credentials and return a JWT token, or error."""
    username = username.strip()
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT username, password_hash, salt, role FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        if not row:
            return {"ok": False, "error": "用户名或密码错误"}

        expected = _hash_password(password, row["salt"])
        if expected != row["password_hash"]:
            return {"ok": False, "error": "用户名或密码错误"}

        token = create_token(row["username"], row["role"])
        logger.info("User logged in: %s", username)
        return {
            "ok": True,
            "token": token,
            "username": row["username"],
            "role": row["role"],
        }
    finally:
        conn.close()


def list_users() -> list[dict[str, Any]]:
    """Admin: list all registered users."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT id, username, role, created_at FROM users ORDER BY id"
        ).fetchall()
        return [{"id": r["id"], "username": r["username"],
                 "role": r["role"], "created_at": r["created_at"]} for r in rows]
    finally:
        conn.close()


def verify_and_get_user(token: str) -> dict[str, Any] | None:
    """Verify token and return user info dict, or None."""
    payload = verify_token(token)
    if not payload:
        return None
    return {"username": payload["username"], "role": payload["role"]}


def delete_user(username: str) -> dict[str, Any]:
    """Admin: delete a user account. Cannot delete admin."""
    if username == "admin":
        return {"ok": False, "error": "不能删除管理员账号"}
    conn = _get_conn()
    try:
        cur = conn.execute("DELETE FROM users WHERE username = ?", (username,))
        conn.commit()
        if cur.rowcount == 0:
            return {"ok": False, "error": "用户不存在"}
        logger.info("User deleted: %s", username)
        return {"ok": True}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# History recording & querying
# ---------------------------------------------------------------------------


_MAX_UPLOAD_HISTORY = 50
_MAX_ANALYSIS_HISTORY = 100


def record_upload(username: str, filename: str, original_name: str) -> None:
    """Record a file upload event. Auto-cleanup oldest entries when exceeding limit."""
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO upload_history (username, filename, original_name, uploaded_at) VALUES (?,?,?,?)",
            (username, filename, original_name, _now()),
        )
        # Cleanup: keep only the last N entries per user
        conn.execute(
            "DELETE FROM upload_history WHERE id NOT IN ("
            "SELECT id FROM upload_history WHERE username = ? ORDER BY id DESC LIMIT ?"
            ") AND username = ?",
            (username, _MAX_UPLOAD_HISTORY, username),
        )
        conn.commit()
    finally:
        conn.close()


def record_analysis(username: str, question: str, result_json: str = "",
                    filename: str = "") -> None:
    """Record an analysis question. If same file as last entry, overwrite instead of adding."""
    conn = _get_conn()
    try:
        if filename:
            # Check if last analysis for this user was on the same file
            last = conn.execute(
                "SELECT id, filename FROM analysis_history WHERE username = ? ORDER BY id DESC LIMIT 1",
                (username,),
            ).fetchone()
            if last and last["filename"] == filename:
                # Overwrite: same file, replace the question
                conn.execute(
                    "UPDATE analysis_history SET question = ?, result_json = ?, created_at = ? WHERE id = ?",
                    (question, result_json, _now(), last["id"]),
                )
                conn.commit()
                return

        conn.execute(
            "INSERT INTO analysis_history (username, filename, question, result_json, created_at) VALUES (?,?,?,?,?)",
            (username, filename, question, result_json, _now()),
        )
        conn.execute(
            "DELETE FROM analysis_history WHERE id NOT IN ("
            "SELECT id FROM analysis_history WHERE username = ? ORDER BY id DESC LIMIT ?"
            ") AND username = ?",
            (username, _MAX_ANALYSIS_HISTORY, username),
        )
        conn.commit()
    finally:
        conn.close()


def delete_history_record(table: str, record_id: int) -> bool:
    """Delete a single history record by id. Returns True if deleted."""
    conn = _get_conn()
    try:
        cur = conn.execute(f"DELETE FROM {table} WHERE id = ?", (record_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def get_upload_history(username: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """Get upload history for a user (or all if username is None)."""
    conn = _get_conn()
    try:
        if username:
            rows = conn.execute(
                "SELECT * FROM upload_history WHERE username = ? ORDER BY id DESC LIMIT ?",
                (username, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM upload_history ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_analysis_history(username: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """Get analysis history for a user (or all if username is None)."""
    conn = _get_conn()
    try:
        if username:
            rows = conn.execute(
                "SELECT * FROM analysis_history WHERE username = ? ORDER BY id DESC LIMIT ?",
                (username, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM analysis_history ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# Init database on import
_init_db()
