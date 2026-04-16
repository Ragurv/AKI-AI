"""
Local web UI for Ollama chat with login, session history, and persisted context.

Start Ollama, then:
  python web_chat.py
  python web_chat.py --role-file roles\\my_coach.txt
  python web_chat.py --port 8765

Open the printed URL (default http://127.0.0.1:8088). Only binds to loopback by default.

Login is configured with environment variables (see README). Defaults are suitable for local
demos only — change WEB_CHAT_PASSWORD (or WEB_CHAT_PASSWORD_HASH) before any shared deployment.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import re
import secrets
import sys
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_MODEL = "dolphin-mistral"
DEFAULT_OLLAMA = "127.0.0.1:11434"
DEFAULT_LISTEN = "127.0.0.1"
DEFAULT_PORT = 8088
ROLES_DIR = Path(__file__).resolve().parent / "roles"
SESSIONS_DIR = Path(__file__).resolve().parent / "chat_sessions"
AUTH_COOKIE = "local_chat_auth"


def _login_creds_from_env() -> tuple[str, str]:
    """Resolve (username, password_sha256_hex) from env for GitHub-safe defaults."""
    # Accept both user var names to avoid silent login mismatches across older setups.
    user = (os.getenv("WEB_CHAT_USER") or os.getenv("WEB_CHAT_USERNAME") or "demo").strip() or "demo"
    hash_hex = (os.getenv("WEB_CHAT_PASSWORD_HASH") or "").strip().lower()
    if hash_hex:
        if len(hash_hex) != 64 or any(c not in "0123456789abcdef" for c in hash_hex):
            print("WEB_CHAT_PASSWORD_HASH must be 64 hex chars; falling back to WEB_CHAT_PASSWORD.", file=sys.stderr)
            hash_hex = ""
    if not hash_hex:
        password = os.getenv("WEB_CHAT_PASSWORD", "change-me")
        hash_hex = hashlib.sha256(password.encode("utf-8")).hexdigest()
    return user, hash_hex


LOGIN_USER, LOGIN_PASSWORD_HASH = _login_creds_from_env()
# Keep model context conservative by default to reduce local runner crashes on low-memory systems.
CONTEXT_MAX_MESSAGES = 24  # Sent to the model for each reply (storage keeps full history)
MIN_CONTEXT_MAX_MESSAGES = 4
MAX_CONTEXT_MAX_MESSAGES = 80
# Model selector: client sends "auto" to pick smallest installed model by disk size (good default on laptops).
AUTO_MODEL = "auto"
MAX_UPLOAD_TOTAL_BYTES = 12 * 1024 * 1024
MAX_TEXT_EXCERPT_CHARS = 24_000
TEXT_FILE_SUFFIXES = (
    ".txt",
    ".md",
    ".csv",
    ".json",
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".html",
    ".css",
    ".xml",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".log",
    ".sh",
    ".ps1",
    ".bat",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".rs",
    ".go",
    ".java",
    ".kt",
    ".sql",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_filename(name: str, max_len: int = 80) -> str:
    # Windows filename invalid characters: < > : " / \ | ? *
    bad = set('<>:"/\\|?*')
    cleaned = "".join("_" if c in bad else c for c in name).strip()
    if not cleaned:
        cleaned = "chat"
    return cleaned[:max_len]


def derive_title_from_query(text: str) -> str:
    """Create a short, readable chat title from a user query."""
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return "New chat"
    # Keep it concise and stable.
    words = cleaned.split(" ")
    title = " ".join(words[:8])
    if len(title) < 8:
        title = cleaned
    return title[:60]


def mode_system_prompt(mode: str) -> str | None:
    """
    Return mode-specific guidance appended as an additional system message.
    """
    normalized = (mode or "").strip().lower()
    if normalized == "ask":
        return (
            "Mode ASK: answer directly and briefly. Ask one concise clarifying question only when required. "
            "Avoid long multi-step plans unless explicitly requested."
        )
    if normalized == "planning":
        return (
            "Mode PLANNING: think in a planning-first style. Provide a clear phased plan, assumptions, options, "
            "trade-offs, risks, and a short validation checklist. Do not jump into implementation details unless asked."
        )
    if normalized == "agent":
        return (
            "Mode AGENT: provide structured, actionable help. Include practical steps, checks, and caveats when useful."
        )
    return None


def normalize_mode(mode: str | None) -> str:
    normalized = (mode or "").strip().lower()
    if normalized in {"ask", "planning", "agent"}:
        return normalized
    return "agent"


def is_auto_model(name: str | None) -> bool:
    if name is None:
        return False
    return str(name).strip().lower() in (AUTO_MODEL, "__auto__")


def normalize_context_max_messages(value: int | None, fallback: int) -> int:
    if not isinstance(value, int):
        return fallback
    return max(MIN_CONTEXT_MAX_MESSAGES, min(MAX_CONTEXT_MAX_MESSAGES, value))


def format_ollama_http_error(code: int, detail: str) -> str:
    clean_detail = (detail or "").strip()
    if code == 500 and "llama runner process has terminated" in clean_detail.lower():
        return (
            "Ollama runner crashed during generation (often memory/context pressure). "
            "Retry, lower context, or use a smaller model. "
            f"Raw detail: {clean_detail}"
        )
    return f"Ollama HTTP {code}: {clean_detail}"


def is_ollama_runner_terminated_error(code: int, detail: str) -> bool:
    return code == 500 and "llama runner process has terminated" in (detail or "").lower()


def reduced_context_limit(current_limit: int) -> int:
    return max(MIN_CONTEXT_MAX_MESSAGES, current_limit // 2)


def _safe_upload_file_path(uploads_root: Path, session_id: str, stored: str) -> Path | None:
    if not stored or ".." in stored:
        return None
    norm = stored.replace("\\", "/").strip().lstrip("/")
    parts = norm.split("/")
    if len(parts) != 2:
        return None
    sid, fname = parts[0], parts[1]
    if sid != session_id or not fname or ".." in fname:
        return None
    try:
        root = uploads_root.resolve()
        target = (uploads_root / sid / fname).resolve()
        target.relative_to(root / sid)
    except (OSError, ValueError):
        return None
    return target


def _attachment_kind(name: str, mime: str) -> str:
    m = (mime or "").lower()
    ext = Path(name or "").suffix.lower()
    if m.startswith("image/") or ext in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"):
        return "image"
    if m.startswith("text/") or ext in TEXT_FILE_SUFFIXES:
        return "text"
    return "other"


def _user_message_to_ollama(msg: dict, uploads_root: Path, session_id: str) -> dict:
    text = (msg.get("content") or "").strip()
    images: list[str] = []
    extra: list[str] = []
    for att in msg.get("attachments") or []:
        if not isinstance(att, dict):
            continue
        stored = att.get("stored")
        if not isinstance(stored, str):
            continue
        name = str(att.get("name") or Path(stored).name)
        mime = str(att.get("mime") or "")
        path = _safe_upload_file_path(uploads_root, session_id, stored)
        if path is None or not path.is_file():
            extra.append(f"\n[Attachment missing or invalid: {name}]")
            continue
        try:
            raw = path.read_bytes()
        except OSError:
            extra.append(f"\n[Could not read attachment: {name}]")
            continue
        kind = att.get("kind") or _attachment_kind(name, mime)
        if kind == "image" or _attachment_kind(name, mime) == "image":
            images.append(base64.b64encode(raw).decode("ascii"))
        elif kind == "text" or _attachment_kind(name, mime) == "text":
            excerpt = raw.decode("utf-8", errors="replace")[:MAX_TEXT_EXCERPT_CHARS]
            extra.append(f"\n\n--- {name} ---\n{excerpt}")
        else:
            extra.append(f"\n[Binary file not inlined for the model: {name} ({mime or 'unknown mime'})]")
    body = (text + "".join(extra)).strip() or ("(attachment)" if images else "")
    out: dict = {"role": "user", "content": body}
    if images:
        out["images"] = images
    return out


def build_ollama_messages(
    session_messages: list[dict],
    system_prompt: str | None,
    mode: str,
    limit: int,
    uploads_root: Path,
    session_id: str,
) -> list[dict]:
    trimmed: list[dict] = []
    for m in session_messages[-limit:]:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "assistant":
            content = m.get("content")
            if isinstance(content, str):
                trimmed.append({"role": "assistant", "content": content})
        elif role == "user":
            trimmed.append(_user_message_to_ollama(m, uploads_root, session_id))

    full: list[dict] = []
    if system_prompt:
        full.append({"role": "system", "content": system_prompt})
    mode_prompt = mode_system_prompt(mode)
    if mode_prompt:
        full.append({"role": "system", "content": mode_prompt})
    full.extend(trimmed)
    return full


def ollama_tags_payload(ollama_host: str) -> dict:
    url = f"http://{ollama_host}/api/tags"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def ollama_model_sizes(ollama_host: str) -> dict[str, int]:
    try:
        data = ollama_tags_payload(ollama_host)
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, ValueError):
        return {}
    out: dict[str, int] = {}
    models = data.get("models")
    if not isinstance(models, list):
        return out
    for item in models:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        size = item.get("size")
        try:
            sz = int(size) if size is not None else 10**15
        except (TypeError, ValueError):
            sz = 10**15
        out[name.strip()] = sz
    return out


def pick_auto_model(ollama_host: str, fallback: str) -> str:
    sizes = ollama_model_sizes(ollama_host)
    if not sizes:
        return fallback
    return min(sizes.items(), key=lambda kv: kv[1])[0]


def resolve_effective_model(ollama_host: str, requested: str | None, cli_default: str) -> str:
    if requested is None:
        return pick_auto_model(ollama_host, cli_default)
    name = str(requested).strip()
    if not name:
        return pick_auto_model(ollama_host, cli_default)
    if is_auto_model(name):
        return pick_auto_model(ollama_host, cli_default)
    return name[:120]


def ollama_show_num_ctx(ollama_host: str, model_name: str) -> int | None:
    url = f"http://{ollama_host}/api/show"
    body = json.dumps({"name": model_name}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    params = data.get("parameters")
    if isinstance(params, str):
        m = re.search(r"num_ctx\s+(\d+)", params, re.I)
        if m:
            return int(m.group(1))
    mi = data.get("model_info")
    if isinstance(mi, dict):
        for k, v in mi.items():
            ks = str(k).lower()
            if ("context" in ks or "ctx" in ks) and isinstance(v, int) and v > 0:
                return int(v)
    return None


def usage_from_ollama_response(data: dict) -> dict:
    return {
        "prompt_eval_count": data.get("prompt_eval_count"),
        "eval_count": data.get("eval_count"),
        "total_duration_ns": data.get("total_duration"),
        "load_duration_ns": data.get("load_duration"),
        "prompt_eval_duration_ns": data.get("prompt_eval_duration"),
        "eval_duration_ns": data.get("eval_duration"),
    }


def ollama_list_models(ollama_host: str) -> list[str]:
    try:
        data = ollama_tags_payload(ollama_host)
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, ValueError):
        return []
    models = data.get("models")
    if not isinstance(models, list):
        return []
    names: list[str] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


def ollama_chat(ollama_host: str, model: str, messages: list[dict]) -> tuple[str, dict]:
    url = f"http://{ollama_host}/api/chat"
    body = json.dumps({"model": model, "messages": messages, "stream": False}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.load(resp)
    if not isinstance(data, dict):
        return "", {}
    msg = data.get("message") or {}
    text = (msg.get("content") or "").strip()
    usage = usage_from_ollama_response(data)
    return text, usage


def ollama_chat_stream_iter(ollama_host: str, model: str, messages: list[dict]):
    """
    Stream from Ollama. Yields ("token", str) for each content chunk, then ("usage", dict) once if available.
    """
    url = f"http://{ollama_host}/api/chat"
    body = json.dumps({"model": model, "messages": messages, "stream": True}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw_line in resp:
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if line.startswith("data:"):
                line = line[len("data:") :].strip()
            if line in ("[DONE]", "DONE"):
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            if obj.get("done") is True:
                u = usage_from_ollama_response(obj)
                if any(v is not None for v in u.values()):
                    yield ("usage", u)
                break
            msg = obj.get("message") or {}
            token = msg.get("content")
            if isinstance(token, str) and token:
                yield ("token", token)


def ollama_chat_stream(ollama_host: str, model: str, messages: list[dict]):
    """Backward-compatible: yield only text tokens (usage discarded)."""
    for kind, payload in ollama_chat_stream_iter(ollama_host, model, messages):
        if kind == "token":
            yield payload


def load_role_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty role file: {path}")
    return text


def resolve_role_path(role: str | None, role_file: Path | None) -> Path | None:
    if role_file is not None:
        return role_file.expanduser().resolve()
    if role:
        return (ROLES_DIR / f"{role}.txt").resolve()
    return None


class SessionStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def _path(self, session_id: str) -> Path:
        return self.root / f"{session_id}.json"

    def list_sessions(self) -> list[dict]:
        items: list[dict] = []
        with self.lock:
            for path in sorted(self.root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                items.append(
                    {
                        "id": data.get("id", path.stem),
                        "title": data.get("title", "Untitled chat"),
                        "createdAt": data.get("createdAt"),
                        "updatedAt": data.get("updatedAt"),
                        "messageCount": len(data.get("messages", [])),
                    }
                )
        return items

    def create_session(self, title: str | None = None) -> dict:
        session_id = secrets.token_hex(8)
        now = utc_now()
        data = {
            "id": session_id,
            "title": title or "New chat",
            "createdAt": now,
            "updatedAt": now,
            "messages": [],
        }
        self.save_session(data)
        return data

    def load_session(self, session_id: str) -> dict | None:
        path = self._path(session_id)
        if not path.is_file():
            return None
        with self.lock:
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None

    def save_session(self, data: dict) -> None:
        path = self._path(data["id"])
        data["updatedAt"] = utc_now()
        with self.lock:
            path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")

    def delete_session(self, session_id: str) -> bool:
        path = self._path(session_id)
        with self.lock:
            if not path.exists():
                return False
            path.unlink()
        return True


def build_app_html(model: str, ollama_host: str, has_system: bool) -> bytes:
    flags = json.dumps({"model": model, "ollamaHost": ollama_host, "hasSystem": has_system, "autoToken": AUTO_MODEL})
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Local chat — {model}</title>
  <style>
    :root {{
      --bg: #1b1c1f;
      --surface: #24262a;
      --surface-2: #2b2e33;
      --border: #383b41;
      --text: #eceef1;
      --muted: #a5aab3;
      --accent: #d9dce2;
      --accent-text: #17191c;
      --danger: #a54d4d;
      --user: #2f3338;
      --assistant: transparent;
      --chat-height-percent: 100;
      --control-bg: #2a2d31;
      --control-border: #4a4e57;
      --composer-border-top: #2f3238;
      --user-border: #3a3e45;
      --soft-separator: rgba(255, 255, 255, 0.06);
      --error-bg: #4d1f26;
      --error-border: #9f4a56;
      --error-text: #ffe8ec;
      --error-muted: #ffd3db;
      --status-error-bg: #4a1f24;
      --status-error-text: #ffdfe5;
      --font-ui: 14px;
      --font-title: 16px;
      --font-meta: 12px;
      --font-msg: 14px;
      --font-status: 12px;
    }}
    :root[data-theme="light"] {{
      --bg: #f6f7f9;
      --surface: #ffffff;
      --surface-2: #f1f3f6;
      --border: #d9dde5;
      --text: #1c2430;
      --muted: #66758a;
      --accent: #3f4858;
      --accent-text: #f5f7fb;
      --danger: #a54d4d;
      --user: #eceff4;
      --assistant: transparent;
      --control-bg: #f1f3f6;
      --control-border: #d0d6e0;
      --composer-border-top: #d8dde6;
      --user-border: #d3d9e4;
      --soft-separator: rgba(24, 34, 49, 0.08);
      --error-bg: #fdecef;
      --error-border: #e9b8c0;
      --error-text: #7f1d2a;
      --error-muted: #8f3a4a;
      --status-error-bg: #fde8ec;
      --status-error-text: #7f1d2a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: system-ui, Segoe UI, Roboto, sans-serif;
      font-size: var(--font-ui);
      background: var(--bg);
      color: var(--text);
      height: 100vh;
      overflow: hidden; /* Keep header fixed; only the chat area scrolls */
      transition: background-color 0.2s ease, color 0.2s ease;
    }}
    button, input, textarea, select {{
      font: inherit;
    }}
    button {{
      border: none;
      border-radius: 10px;
      padding: 0.58rem 0.9rem;
      background: var(--accent);
      color: var(--accent-text);
      font-weight: 600;
      cursor: pointer;
      transition: background 0.15s ease, border-color 0.15s ease;
    }}
    button.secondary {{
      background: var(--control-bg);
      color: var(--text);
      border: 1px solid var(--border);
    }}
    summary.secondary {{
      display: inline-flex;
      align-items: center;
      border-radius: 12px;
      padding: 0.58rem 0.9rem;
      background: var(--control-bg);
      color: var(--text);
      border: 1px solid var(--border);
      font-weight: 600;
      cursor: pointer;
      user-select: none;
    }}
    button.danger {{
      background: var(--danger);
      color: white;
    }}
    button:disabled {{
      opacity: 0.5;
      cursor: not-allowed;
    }}
    input, textarea, select {{
      width: 100%;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: var(--control-bg);
      color: var(--text);
      padding: 0.7rem 0.85rem;
    }}
    textarea {{
      resize: vertical;
      min-height: 3.5rem;
      max-height: 12rem;
    }}
    .hidden {{
      display: none !important;
    }}
    .login-shell {{
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 1rem;
    }}
    .login-card {{
      width: min(100%, 420px);
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 1.25rem;
      box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }}
    .login-card h1 {{
      margin: 0 0 0.5rem;
      font-size: 1.2rem;
      text-align: center;
      color: #9ef7b6;
    }}
    .meta, .hint {{
      color: var(--muted);
      font-size: var(--font-meta);
    }}
    .hint.error {{
      color: var(--error-text);
      background: var(--error-bg);
      border: 1px solid var(--error-border);
      border-radius: 10px;
      padding: 0.5rem 0.65rem;
    }}
    .stack {{
      display: grid;
      gap: 0.75rem;
    }}
    .app {{
      display: grid;
      grid-template-columns: 280px 1fr;
      height: calc(var(--chat-height-percent) * 1vh);
      min-height: 0;
    }}
    .sidebar {{
      border-right: 1px solid var(--soft-separator);
      background: var(--surface);
      padding: 1rem;
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }}
    .sidebar-brand {{
      display: grid;
      gap: 0.15rem;
      padding-bottom: 0.2rem;
      border-bottom: 1px solid var(--soft-separator);
    }}
    .sidebar-brand-title {{
      font-size: 0.95rem;
      letter-spacing: 0.03em;
      font-weight: 700;
    }}
    .sidebar-main {{
      display: grid;
      gap: 0.65rem;
      min-height: 0;
      flex: 1;
    }}
    .sidebar-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 0.5rem;
    }}
    .session-list {{
      display: grid;
      gap: 0.5rem;
      overflow-y: auto;
      min-height: 0;
    }}
    .sidebar-footer {{
      margin-top: auto;
      display: grid;
      gap: 0.45rem;
      padding-top: 0.65rem;
      border-top: 1px solid var(--soft-separator);
    }}
    .sidebar-action {{
      width: 100%;
      justify-content: flex-start;
      text-align: left;
    }}
    .session-item {{
      background: transparent;
      border: 1px solid transparent;
      border-radius: 12px;
      padding: 0.6rem 0.7rem;
      cursor: pointer;
    }}
    .session-item.active {{
      border-color: var(--soft-separator);
      background: var(--surface-2);
    }}
    .session-title {{
      font-weight: 600;
      margin-bottom: 0.25rem;
      font-size: 0.88rem;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .session-sub {{
      color: var(--muted);
      font-size: 0.73rem;
      opacity: 0.8;
    }}
    .searchBox {{
      margin-top: -0.1rem;
      margin-bottom: 0.2rem;
      padding: 0.55rem 0.7rem;
    }}
    .mini-btn {{
      padding: 0.35rem 0.65rem;
      border-radius: 8px;
      font-weight: 600;
      font-size: 0.85rem;
    }}
    details.dropdown {{
      position: relative;
    }}
    details.dropdown > summary {{
      list-style: none;
      cursor: pointer;
    }}
    details.dropdown[open] .dropdown-menu {{
      display: flex;
      flex-direction: column;
    }}
    .dropdown-menu {{
      display: none;
      position: absolute;
      right: 0;
      top: calc(100% - 0.1rem);
      padding: 0.45rem;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      z-index: 50;
      gap: 0.4rem;
      min-width: 140px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    }}
    .msg-meta {{
      margin-top: 0.35rem;
      color: var(--muted);
      font-size: 0.72rem;
      line-height: 1.2;
      opacity: 0.75;
    }}
    .msg-role {{
      color: var(--muted);
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 0.35rem;
      font-weight: 600;
      opacity: 0.85;
    }}
    .copy-btn {{
      margin-top: 0.45rem;
      font-size: 0.75rem;
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.15s ease;
    }}
    .bubble.assistant:hover .copy-btn {{
      opacity: 1;
      pointer-events: auto;
    }}
    .main {{
      display: flex;
      flex-direction: column;
      min-width: 0;
      height: 100vh;
      background: var(--bg);
    }}
    .topbar {{
      padding: 0.85rem 1.2rem;
      border-bottom: 1px solid var(--soft-separator);
      background: var(--bg);
      display: flex;
      justify-content: space-between;
      gap: 0.75rem;
      align-items: center;
      flex-wrap: wrap;
      position: sticky;
      top: 0;
      z-index: 10;
    }}
    .topbar h2 {{
      margin: 0;
      font-size: var(--font-title);
      font-weight: 600;
    }}
    .settings-panel {{
      position: fixed;
      top: 0;
      right: 0;
      height: 100vh;
      width: min(420px, 92vw);
      background: var(--surface);
      border-left: 1px solid var(--border);
      z-index: 100;
      padding: 1rem;
      transform: translateX(100%);
      transition: transform 0.18s ease;
      overflow-y: auto;
    }}
    .settings-panel.open {{
      transform: translateX(0);
    }}
    .settings-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.6rem;
      margin-bottom: 0.9rem;
    }}
    .settings-group {{
      display: grid;
      gap: 0.45rem;
      margin-bottom: 0.95rem;
    }}
    .inline-row {{
      display: flex;
      gap: 0.45rem;
      align-items: center;
    }}
    .settings-help {{
      font-size: 0.8rem;
      color: var(--muted);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 0.7rem;
      background: var(--surface-2);
      line-height: 1.45;
    }}
    .processing {{
      display: inline-block;
      width: 0.8rem;
      height: 0.8rem;
      border: 2px solid var(--muted);
      border-top-color: transparent;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      vertical-align: -0.12rem;
      margin-right: 0.35rem;
    }}
    @keyframes spin {{
      to {{ transform: rotate(360deg); }}
    }}
    .messages {{
      flex: 1;
      overflow-y: auto;
      min-height: 0;
      padding: 1rem 1.1rem;
      display: flex;
      flex-direction: column;
      gap: 0.65rem;
      max-width: none;
      width: 100%;
      margin: 0;
    }}
    .bubble {{
      max-width: 100%;
      padding: 0.72rem 0.82rem;
      border-radius: 10px;
      font-size: var(--font-msg);
      line-height: 1.5;
      white-space: pre-wrap;
      word-break: break-word;
      border: 1px solid transparent;
    }}
    .bubble.user {{
      align-self: stretch;
      background: var(--user);
      border-color: var(--soft-separator);
    }}
    .bubble.assistant {{
      align-self: stretch;
      background: var(--assistant);
      border-color: transparent;
    }}
    .bubble.error {{
      align-self: stretch;
      background: var(--error-bg);
      border: 1px solid var(--error-border);
      color: var(--error-text);
    }}
    .bubble.error .msg-role,
    .bubble.error .msg-meta {{
      color: var(--error-muted);
      opacity: 1;
    }}
    .composer {{
      padding: 0.65rem 1rem 1rem;
      border-top: 1px solid var(--composer-border-top);
      background: var(--bg);
    }}
    .composer-inner {{
      max-width: none;
      width: 100%;
      margin: 0;
      display: grid;
      gap: 0.45rem;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 0.75rem;
    }}
    .composer-row {{
      display: flex;
      gap: 0.6rem;
      align-items: flex-end;
    }}
    .mode-toggle {{
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      color: var(--muted);
      font-size: 0.82rem;
      white-space: nowrap;
      padding-bottom: 0.25rem;
    }}
    .mode-badge {{
      display: inline-flex;
      align-items: center;
      gap: 0.25rem;
      margin-left: 0.45rem;
      font-size: 0.66rem;
      border: 1px solid var(--soft-separator);
      color: var(--muted);
      border-radius: 999px;
      padding: 0.1rem 0.42rem;
      vertical-align: middle;
      letter-spacing: 0.04em;
    }}
    .mode-toggle select {{
      min-width: 98px;
      padding: 0.4rem 0.6rem;
      border-radius: 10px;
      background: var(--control-bg);
      border: 1px solid var(--control-border);
      color: var(--text);
    }}
    .model-toggle {{
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      color: var(--muted);
      font-size: 0.82rem;
      white-space: nowrap;
      padding-bottom: 0.25rem;
    }}
    .model-toggle select {{
      min-width: 150px;
      padding: 0.4rem 0.6rem;
      border-radius: 10px;
      background: var(--control-bg);
      border: 1px solid var(--control-border);
      color: var(--text);
    }}
    .composer-toggle {{
      display: flex;
      align-items: center;
      gap: 0.35rem;
      color: var(--muted);
      font-size: 0.82rem;
      white-space: nowrap;
      padding-bottom: 0.25rem;
    }}
    .composer textarea {{
      background: var(--control-bg);
      border: 1px solid var(--control-border);
      border-radius: 14px;
      padding: 0.75rem 0.9rem;
      font-size: var(--font-msg);
      line-height: 1.45;
      color: var(--text);
      caret-color: var(--text);
    }}
    .context-bar {{
      font-size: 0.72rem;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      padding: 0.25rem 0.2rem 0;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      flex-wrap: wrap;
    }}
    .context-bar strong {{
      color: var(--text);
      font-weight: 600;
    }}
    .attachment-chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.35rem;
      margin-bottom: 0.35rem;
    }}
    .attachment-chip {{
      font-size: 0.72rem;
      padding: 0.2rem 0.45rem;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: var(--surface-2);
      color: var(--text);
      max-width: 100%;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .attachment-chip button {{
      margin-left: 0.25rem;
      padding: 0 0.25rem;
      font-size: 0.7rem;
      min-height: auto;
      background: transparent;
      color: var(--muted);
      border-radius: 4px;
    }}
    .bubble .attachment-list {{
      margin-top: 0.4rem;
      font-size: 0.75rem;
      color: var(--muted);
    }}
    .composer button {{
      min-height: 40px;
      font-size: 0.86rem;
      letter-spacing: 0.01em;
    }}
    .status {{
      color: var(--muted);
      font-size: var(--font-status);
      padding-left: 0.2rem;
    }}
    .status.error {{
      color: var(--status-error-text);
      background: var(--status-error-bg);
      border: 1px solid var(--error-border);
      border-radius: 8px;
      padding: 0.35rem 0.5rem;
      display: inline-flex;
      align-items: center;
      gap: 0.25rem;
    }}
    @media (max-width: 900px) {{
      .app {{
        grid-template-columns: 1fr;
      }}
      .sidebar {{
        border-right: none;
        border-bottom: 1px solid var(--border);
      }}
    }}
  </style>
</head>
<body>
  <section id="loginView" class="login-shell">
    <div class="login-card stack">
      <div>
        <h1>AKI-AI Login</h1>
        <div class="meta">Model: {model}{' · custom role enabled' if has_system else ''}</div>
      </div>
      <input id="loginUser" placeholder="Username" autocomplete="username" />
      <input id="loginPass" type="password" placeholder="Password" autocomplete="current-password" />
      <button id="loginBtn" type="button">Login</button>
      <div id="loginError" class="hint"></div>
    </div>
  </section>

  <section id="appView" class="app hidden">
    <aside class="sidebar">
      <div class="sidebar-brand">
        <div class="sidebar-brand-title">AKI-AI</div>
        <div class="meta">Local AI workspace</div>
      </div>
      <div class="sidebar-main">
        <div class="sidebar-header">
          <strong>Chats</strong>
          <button id="newChatBtn" type="button">New chat</button>
        </div>
        <div class="meta">Local conversation history</div>
        <input id="searchBox" class="searchBox" type="text" placeholder="Search chats..." />
        <div id="sessionList" class="session-list"></div>
      </div>
      <div class="sidebar-footer">
        <button id="settingsBtn" type="button" class="secondary sidebar-action">Settings</button>
        <button id="renameChatBtn" type="button" class="secondary sidebar-action">Rename chat</button>
        <button id="deleteChatBtn" type="button" class="secondary sidebar-action">Delete chat</button>
        <button id="exportTxtBtn" type="button" class="secondary sidebar-action">Export TXT</button>
        <button id="exportJsonBtn" type="button" class="secondary sidebar-action">Export JSON</button>
        <button id="logoutBtn" type="button" class="danger sidebar-action">Logout</button>
      </div>
    </aside>

    <main class="main">
      <div class="topbar">
        <div>
          <h2 id="chatTitle">New conversation</h2>
          <div class="meta" id="subtitle"></div>
        </div>
      </div>
      <div id="messages" class="messages" aria-live="polite"></div>
      <div class="composer">
        <div class="composer-inner">
          <div class="composer-row">
            <div style="display:flex; flex-direction:column; gap:0.35rem; flex:1; min-width:0;">
              <div id="attachmentChips" class="attachment-chips"></div>
              <textarea id="input" rows="2" placeholder="Message… (Enter to send, Shift+Enter for newline)" spellcheck="true" lang="en-US"></textarea>
            </div>
            <input type="file" id="fileInput" class="hidden" multiple />
            <button id="attachBtn" type="button" class="secondary" title="Attach files">Attach</button>
            <button id="sendBtn" type="button">Send</button>
            <label class="model-toggle">
              Model
              <select id="modelSelect"></select>
            </label>
            <label class="mode-toggle">
              Mode
              <select id="modeSelect">
                <option value="ask">ASK</option>
                <option value="planning">PLANNING</option>
                <option value="agent" selected>AGENT</option>
              </select>
            </label>
            <label class="composer-toggle">
              <input id="streamToggle" type="checkbox" checked />
              Stream
            </label>
            <button id="stopBtn" type="button" class="secondary" disabled>Stop</button>
          </div>
          <div id="status" class="status"><span id="processingIcon" class="processing hidden"></span><span id="statusText">Ready.</span></div>
          <div id="contextBar" class="context-bar" title="Last request token usage from Ollama">Context: —</div>
        </div>
      </div>
    </main>
  </section>

  <aside id="settingsPanel" class="settings-panel" aria-label="Settings panel">
    <div class="settings-head">
      <strong>Profile & Settings</strong>
      <button id="closeSettingsBtn" type="button" class="secondary mini-btn">Close</button>
    </div>
    <div class="settings-group">
      <label for="profileNameInput" class="meta">Profile name</label>
      <input id="profileNameInput" type="text" placeholder="Your name" />
    </div>
    <div class="settings-group">
      <label for="themeSelect" class="meta">Theme</label>
      <select id="themeSelect">
        <option value="dark">Dark</option>
        <option value="light">Light</option>
        <option value="system">System (Auto)</option>
      </select>
    </div>
    <div class="settings-group">
      <label class="meta" for="spellcheckToggle">
        <input id="spellcheckToggle" type="checkbox" checked />
        Enable spellcheck in chat input
      </label>
    </div>
    <div class="settings-group">
      <label for="chatHeightRange" class="meta">Chat window height: <span id="chatHeightValue">100%</span></label>
      <input id="chatHeightRange" type="range" min="70" max="100" step="1" value="100" />
    </div>
    <div class="settings-group">
      <label for="modelSelect" class="meta">LLM model</label>
      <input id="modelSearchInput" type="text" placeholder="Filter models (e.g. mistral, llama)" />
      <div class="meta" id="modelFilterMeta">Showing 0 models</div>
      <div class="inline-row">
        <button id="refreshModelsBtn" type="button" class="secondary mini-btn">Refresh models</button>
      </div>
    </div>
    <div class="settings-help">
      <strong>Reference</strong><br/>
      - Stream ON: token-by-token response and Stop supported.<br/>
      - Stream OFF: full response at once.<br/>
      - ASK: short direct answers with minimal clarifying question.<br/>
      - PLANNING: phased plan with options, trade-offs, and risks.<br/>
      - AGENT: actionable execution steps and practical checks.<br/>
      - Model: <strong>Auto</strong> picks the smallest installed model (by size) for each request; or choose a fixed model.<br/>
      - <strong>Attach</strong>: upload files (images go to vision-capable models as images; text files are inlined).<br/>
      - <strong>Context bar</strong>: shows last reply token usage vs model context window when Ollama reports counts.<br/>
      - Model filter: quickly search model names when list is large.<br/>
      - Spellcheck applies to your message box only.<br/>
      - Chat height controls the app panel height in the browser window.
    </div>
  </aside>

  <script>
    const CFG = {flags};
    const loginView = document.getElementById("loginView");
    const appView = document.getElementById("appView");
    const loginUser = document.getElementById("loginUser");
    const loginPass = document.getElementById("loginPass");
    const loginBtn = document.getElementById("loginBtn");
    const loginError = document.getElementById("loginError");
    const searchBoxEl = document.getElementById("searchBox");
    const sessionListEl = document.getElementById("sessionList");
    const chatTitleEl = document.getElementById("chatTitle");
    const subtitleEl = document.getElementById("subtitle");
    const messagesEl = document.getElementById("messages");
    const inputEl = document.getElementById("input");
    const sendBtn = document.getElementById("sendBtn");
    const modelSelect = document.getElementById("modelSelect");
    const modelSearchInput = document.getElementById("modelSearchInput");
    const modelFilterMeta = document.getElementById("modelFilterMeta");
    const refreshModelsBtn = document.getElementById("refreshModelsBtn");
    const newChatBtn = document.getElementById("newChatBtn");
    const exportTxtBtn = document.getElementById("exportTxtBtn");
    const exportJsonBtn = document.getElementById("exportJsonBtn");
    const deleteChatBtn = document.getElementById("deleteChatBtn");
    const logoutBtn = document.getElementById("logoutBtn");
    const statusEl = document.getElementById("status");
    const statusTextEl = document.getElementById("statusText");
    const processingIconEl = document.getElementById("processingIcon");
    const stopBtn = document.getElementById("stopBtn");
    const streamToggle = document.getElementById("streamToggle");
    const modeSelect = document.getElementById("modeSelect");
    const settingsBtn = document.getElementById("settingsBtn");
    const settingsPanel = document.getElementById("settingsPanel");
    const closeSettingsBtn = document.getElementById("closeSettingsBtn");
    const profileNameInput = document.getElementById("profileNameInput");
    const themeSelect = document.getElementById("themeSelect");
    const spellcheckToggle = document.getElementById("spellcheckToggle");
    const chatHeightRange = document.getElementById("chatHeightRange");
    const chatHeightValue = document.getElementById("chatHeightValue");
    const attachmentChips = document.getElementById("attachmentChips");
    const fileInput = document.getElementById("fileInput");
    const attachBtn = document.getElementById("attachBtn");
    const contextBarEl = document.getElementById("contextBar");

    let sessions = [];
    let currentSessionId = null;
    let currentMessages = [];
    let activeController = null;
    let isBusy = false;
    let queue = [];
    let autoScroll = true;
    let availableModels = [];
    let modelFilterQuery = "";
    let pendingAttachments = [];
    let lastAutoResolved = "";
    const SETTINGS_KEY = "local_chat_settings_v2";
    const OLD_SETTINGS_KEY = "local_chat_settings_v1";
    const systemThemeMedia = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
    const AUTO_MODEL_TOKEN = (CFG.autoToken || "auto").toLowerCase();
    let uiSettings = {{
      profileName: "",
      theme: "dark",
      spellcheck: true,
      chatHeight: 100,
      mode: "agent",
      model: AUTO_MODEL_TOKEN,
    }};

    function normalizeMode(mode) {{
      const m = String(mode || "").trim().toLowerCase();
      return (m === "ask" || m === "planning" || m === "agent") ? m : "agent";
    }}

    function normalizeModelName(name) {{
      const raw = String(name || "").trim();
      if (!raw) return AUTO_MODEL_TOKEN;
      const v = raw.toLowerCase();
      if (v === "auto" || v === AUTO_MODEL_TOKEN || v === "__auto__") return AUTO_MODEL_TOKEN;
      return raw;
    }}

    function shouldFallbackModel(errorText, modelName) {{
      const selected = normalizeModelName(modelName);
      const fallback = normalizeModelName(CFG.model);
      if (selected === fallback || selected === AUTO_MODEL_TOKEN) return false;
      const msg = String(errorText || "").toLowerCase();
      if (!msg) return false;
      return (
        msg.includes("model") &&
        (
          msg.includes("not found") ||
          msg.includes("does not exist") ||
          msg.includes("pull") ||
          msg.includes("manifest") ||
          msg.includes("unknown")
        )
      );
    }}

    function filteredModelList() {{
      const base = availableModels.length ? availableModels : [CFG.model];
      const q = (modelFilterQuery || "").trim().toLowerCase();
      if (!q) return base;
      return base.filter((m) => String(m).toLowerCase().includes(q));
    }}

    function modelFamilyName(modelName) {{
      const normalized = normalizeModelName(modelName).toLowerCase();
      const noTag = normalized.split(":")[0];
      const family = noTag.split(/[-_.]/)[0];
      if (!family) return "other";
      if (/^[a-z0-9]+$/.test(family)) return family;
      return "other";
    }}

    function rebuildModelOptions(selectedName = null) {{
      const selected = normalizeModelName(selectedName || uiSettings.model || AUTO_MODEL_TOKEN);
      const filtered = filteredModelList();
      const sourceCount = availableModels.length ? availableModels.length : 1;
      const names = filtered.slice();
      if (selected !== AUTO_MODEL_TOKEN && !names.includes(selected)) {{
        names.unshift(selected);
      }}
      modelSelect.innerHTML = "";
      const optAuto = document.createElement("option");
      optAuto.value = AUTO_MODEL_TOKEN;
      optAuto.textContent = "Auto (smallest installed)";
      modelSelect.appendChild(optAuto);

      const seen = new Set([AUTO_MODEL_TOKEN]);
      const grouped = new Map();
      for (const modelName of names) {{
        const normalized = normalizeModelName(modelName);
        if (normalized === AUTO_MODEL_TOKEN) continue;
        if (seen.has(normalized)) continue;
        seen.add(normalized);
        const family = modelFamilyName(normalized);
        if (!grouped.has(family)) grouped.set(family, []);
        grouped.get(family).push(normalized);
      }}
      const families = Array.from(grouped.keys()).sort((a, b) => a.localeCompare(b));
      for (const family of families) {{
        const opts = grouped.get(family) || [];
        const groupEl = document.createElement("optgroup");
        groupEl.label = family.toUpperCase();
        for (const modelName of opts) {{
          const opt = document.createElement("option");
          opt.value = modelName;
          opt.textContent = modelName;
          groupEl.appendChild(opt);
        }}
        modelSelect.appendChild(groupEl);
      }}
      if (!families.length && selected !== AUTO_MODEL_TOKEN) {{
        const opt = document.createElement("option");
        opt.value = selected;
        opt.textContent = selected;
        modelSelect.appendChild(opt);
      }}
      modelSelect.value = selected;
      modelFilterMeta.textContent = "Showing " + String(Math.max(0, seen.size - 1)) + " of " + String(sourceCount) + " models (+ Auto)";
    }}

    function updateAutoScrollFlag() {{
      // If the user is near the bottom, we keep autoscroll enabled.
      const remaining = messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight;
      autoScroll = remaining < 80;
    }}

    function maybeAutoScroll() {{
      if (autoScroll) {{
        messagesEl.scrollTop = messagesEl.scrollHeight;
      }}
    }}

    messagesEl.addEventListener("scroll", () => {{
      updateAutoScrollFlag();
    }});
    updateAutoScrollFlag();

    function modelSubtitleLabel() {{
      const m = normalizeModelName(uiSettings.model);
      if (m === AUTO_MODEL_TOKEN) {{
        const hint = lastAutoResolved ? (" → " + lastAutoResolved) : "";
        return "Auto" + hint;
      }}
      return m;
    }}

    function updateSubtitle() {{
      const name = uiSettings.profileName ? (uiSettings.profileName + " · ") : "";
      subtitleEl.textContent = name + modelSubtitleLabel() + (CFG.hasSystem ? " · custom role" : "");
    }}

    function setProcessing(on) {{
      processingIconEl.classList.toggle("hidden", !on);
    }}

    function resolveTheme(theme) {{
      if (theme === "system" && systemThemeMedia) {{
        return systemThemeMedia.matches ? "dark" : "light";
      }}
      return theme === "light" ? "light" : "dark";
    }}

    function applyTheme(theme) {{
      document.documentElement.setAttribute("data-theme", resolveTheme(theme));
    }}

    function applySpellcheck(enabled) {{
      const isEnabled = !!enabled;
      inputEl.spellcheck = isEnabled;
      inputEl.setAttribute("spellcheck", isEnabled ? "true" : "false");
      if (isEnabled) {{
        inputEl.setAttribute("lang", "en-US");
      }}
      inputEl.setAttribute("autocorrect", "on");
      inputEl.setAttribute("autocomplete", "on");
      inputEl.setAttribute("autocapitalize", "sentences");
    }}

    function applyChatHeight(percent) {{
      const p = Math.max(70, Math.min(100, Number(percent) || 100));
      document.documentElement.style.setProperty("--chat-height-percent", String(p));
      chatHeightValue.textContent = p + "%";
    }}

    function ensureModelOption(name) {{
      const normalized = normalizeModelName(name);
      if (normalized === AUTO_MODEL_TOKEN) return normalized;
      if (!Array.from(modelSelect.options).some((o) => o.value === normalized)) {{
        const opt = document.createElement("option");
        opt.value = normalized;
        opt.textContent = normalized;
        modelSelect.appendChild(opt);
      }}
      return normalized;
    }}

    function applyModelSelection(name, persist = false) {{
      const normalized = normalizeModelName(name);
      rebuildModelOptions(normalized);
      if (normalized !== AUTO_MODEL_TOKEN) {{
        ensureModelOption(normalized);
      }}
      modelSelect.value = normalized;
      uiSettings.model = normalized;
      if (persist) saveSettings();
      updateSubtitle();
    }}

    function applySettings() {{
      applyTheme(uiSettings.theme);
      applySpellcheck(uiSettings.spellcheck);
      applyChatHeight(uiSettings.chatHeight);
      updateSubtitle();
      profileNameInput.value = uiSettings.profileName || "";
      themeSelect.value = (uiSettings.theme === "light" || uiSettings.theme === "system") ? uiSettings.theme : "dark";
      spellcheckToggle.checked = !!uiSettings.spellcheck;
      chatHeightRange.value = String(uiSettings.chatHeight || 100);
      modeSelect.value = normalizeMode(uiSettings.mode);
      modelSearchInput.value = modelFilterQuery;
      applyModelSelection(uiSettings.model, false);
    }}

    function saveSettings() {{
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(uiSettings));
    }}

    function loadSettings() {{
      try {{
        let raw = localStorage.getItem(SETTINGS_KEY);
        if (!raw) raw = localStorage.getItem(OLD_SETTINGS_KEY);
        if (!raw) return;
        const parsed = JSON.parse(raw);
        if (!parsed || typeof parsed !== "object") return;
        uiSettings = {{
          ...uiSettings,
          ...parsed,
        }};
      }} catch {{}}
    }}

    loadSettings();
    applySettings();

    async function api(path, options = {{}}) {{
      const res = await fetch(path, {{
        credentials: "same-origin",
        headers: {{ "Content-Type": "application/json", ...(options.headers || {{}}) }},
        ...options,
      }});
      const text = await res.text();
      const data = text ? JSON.parse(text) : {{}};
      if (!res.ok) {{
        throw new Error(data.error || ("HTTP " + res.status));
      }}
      return data;
    }}

    async function refreshModels(quiet = false) {{
      try {{
        const data = await api("/api/models", {{ method: "GET" }});
        const models = Array.isArray(data.models) ? data.models.filter((m) => typeof m === "string" && m.trim()) : [];
        availableModels = models;
        if (typeof data.autoModel === "string" && data.autoModel.trim()) {{
          lastAutoResolved = data.autoModel.trim();
        }}
        applyModelSelection(uiSettings.model || AUTO_MODEL_TOKEN, false);
        updateSubtitle();
        if (!quiet) setStatus("Models refreshed.");
      }} catch (err) {{
        availableModels = [];
        lastAutoResolved = "";
        applyModelSelection(uiSettings.model || AUTO_MODEL_TOKEN, false);
        if (!quiet) setStatus("Could not load models list.");
      }}
    }}

    function setLoggedIn(isLoggedIn) {{
      loginView.classList.toggle("hidden", isLoggedIn);
      appView.classList.toggle("hidden", !isLoggedIn);
      if (isLoggedIn) {{
        // Re-apply after view switch; some browsers reset textarea spellcheck on hidden nodes.
        applySpellcheck(uiSettings.spellcheck);
      }}
    }}

    function setStatus(text, level = "info") {{
      statusTextEl.textContent = text;
      statusEl.classList.toggle("error", level === "error");
    }}

    function addBubble(text, role, createdAt, pending = false, mode = "", modelName = "", attachments = null) {{
      const div = document.createElement("div");
      div.className = "bubble " + role;

      const roleLabel = document.createElement("div");
      roleLabel.className = "msg-role";
      roleLabel.textContent = role === "user" ? "You" : (role === "assistant" ? "Assistant" : role);
      if (role === "assistant") {{
        const badge = document.createElement("span");
        badge.className = "mode-badge";
        const modeText = normalizeMode(mode).toUpperCase();
        const modelText = modelName || lastAutoResolved || modelSubtitleLabel();
        badge.textContent = modeText + " · " + modelText;
        roleLabel.appendChild(badge);
      }}
      div.appendChild(roleLabel);

      const content = document.createElement("div");
      content.className = "bubble-content";
      content.textContent = text;
      div.appendChild(content);

      if (attachments && attachments.length && role === "user") {{
        const al = document.createElement("div");
        al.className = "attachment-list";
        al.textContent = "Attached: " + attachments.map((a) => (a && a.name) ? a.name : "file").join(", ");
        div.appendChild(al);
      }}

      if (createdAt) {{
        const meta = document.createElement("div");
        meta.className = "msg-meta";
        const d = new Date(createdAt);
        meta.textContent = isNaN(d.getTime()) ? "" : d.toLocaleString();
        if (meta.textContent) div.appendChild(meta);
      }}
      if (pending) {{
        const p = document.createElement("div");
        p.className = "msg-meta";
        p.textContent = "Queued";
        div.appendChild(p);
      }}

      if (role === "assistant") {{
        const copy = document.createElement("button");
        copy.type = "button";
        copy.className = "mini-btn secondary copy-btn";
        copy.textContent = "Copy";
        copy.addEventListener("click", async (e) => {{
          e.stopPropagation();
          try {{
            await navigator.clipboard.writeText(content.textContent);
            copy.textContent = "Copied";
            setTimeout(() => (copy.textContent = "Copy"), 900);
          }} catch {{
            copy.textContent = "Copy failed";
            setTimeout(() => (copy.textContent = "Copy"), 1200);
          }}
        }});
        div.appendChild(copy);
      }}

      messagesEl.appendChild(div);
      maybeAutoScroll();
    }}

    function renderMessages() {{
      messagesEl.innerHTML = "";
      for (const msg of currentMessages) {{
        addBubble(
          msg.content,
          msg.role,
          msg.createdAt,
          !!msg.pending,
          msg.mode,
          msg.model,
          msg.attachments || null,
        );
      }}
      updateContextBarFromMessages();
    }}

    function renderSessions() {{
      sessionListEl.innerHTML = "";
      const q = (searchBoxEl && searchBoxEl.value ? searchBoxEl.value : "").trim().toLowerCase();
      for (const session of sessions) {{
        if (q && !((session.title || "")).toLowerCase().includes(q)) continue;
        const item = document.createElement("div");
        item.className = "session-item" + (session.id === currentSessionId ? " active" : "");
        item.innerHTML =
          '<div style="display:flex; justify-content:space-between; align-items:center; gap:0.5rem;">' +
            '<div class="session-title"></div>' +
          '</div>' +
          '<div class="session-sub"></div>';
        item.querySelector(".session-title").textContent = session.title || "Untitled chat";
        item.querySelector(".session-sub").textContent =
          (session.messageCount || 0) + " messages";

        item.addEventListener("click", () => loadSession(session.id));
        sessionListEl.appendChild(item);
      }}
    }}

    async function refreshSessions() {{
      const data = await api("/api/sessions", {{ method: "GET" }});
      sessions = data.sessions || [];
      renderSessions();
      if (!currentSessionId && sessions.length) {{
        await loadSession(sessions[0].id);
      }}
    }}

    function fmtTok(n) {{
      if (typeof n !== "number" || !isFinite(n)) return "—";
      if (n >= 10000) return (n / 1000).toFixed(0) + "k";
      if (n >= 1000) return (n / 1000).toFixed(1) + "k";
      return String(Math.round(n));
    }}

    function updateContextBar(usage) {{
      if (!contextBarEl) return;
      if (!usage || typeof usage !== "object") {{
        contextBarEl.textContent = "Context: —";
        return;
      }}
      const pr = usage.prompt_eval_count;
      const ev = usage.eval_count;
      const total =
        typeof usage.total_tokens === "number"
          ? usage.total_tokens
          : typeof pr === "number" && typeof ev === "number"
            ? pr + ev
            : null;
      const ctx =
        typeof usage.num_ctx === "number"
          ? usage.num_ctx
          : typeof usage.numCtx === "number"
            ? usage.numCtx
            : null;
      const m = usage.model || "";
      let pct = "";
      if (typeof total === "number" && typeof ctx === "number" && ctx > 0) {{
        pct = " · " + Math.min(100, Math.round((total / ctx) * 100)) + "% of window";
      }}
      contextBarEl.textContent =
        "Context: " +
        (typeof total === "number" ? fmtTok(total) + " tok" : "—") +
        (typeof ctx === "number" ? " / " + fmtTok(ctx) + " ctx" : "") +
        (m ? " · " + m : "") +
        pct;
    }}

    function updateContextBarFromMessages() {{
      for (let i = currentMessages.length - 1; i >= 0; i--) {{
        const m = currentMessages[i];
        if (m.role === "assistant" && m.usage && typeof m.usage === "object") {{
          updateContextBar(m.usage);
          return;
        }}
      }}
      updateContextBar(null);
    }}

    function renderAttachmentChips() {{
      if (!attachmentChips) return;
      attachmentChips.innerHTML = "";
      pendingAttachments.forEach((a) => {{
        const chip = document.createElement("span");
        chip.className = "attachment-chip";
        chip.title = a.name || "file";
        chip.appendChild(document.createTextNode((a.name || "file").slice(0, 48)));
        const rm = document.createElement("button");
        rm.type = "button";
        rm.textContent = "×";
        const key = a.stored || a.name || "";
        rm.addEventListener("click", () => {{
          pendingAttachments = pendingAttachments.filter((x) => (x.stored || x.name || "") !== key);
          renderAttachmentChips();
        }});
        chip.appendChild(rm);
        attachmentChips.appendChild(chip);
      }});
    }}

    async function uploadPendingFiles(fileList) {{
      if (!currentSessionId || !fileList || !fileList.length) return;
      const files = Array.from(fileList).slice(0, 12);
      const payloadFiles = [];
      for (const f of files) {{
        const buf = await f.arrayBuffer();
        if (buf.byteLength > 8 * 1024 * 1024) {{
          setStatus("Skipped (too large): " + f.name, "error");
          continue;
        }}
        let b64 = "";
        const bytes = new Uint8Array(buf);
        let bin = "";
        const chunk = 0x8000;
        for (let j = 0; j < bytes.length; j += chunk) {{
          bin += String.fromCharCode.apply(null, bytes.subarray(j, j + chunk));
        }}
        b64 = btoa(bin);
        payloadFiles.push({{ name: f.name, mime: f.type || "application/octet-stream", data: b64 }});
      }}
      if (!payloadFiles.length) return;
      const data = await api("/api/upload", {{
        method: "POST",
        body: JSON.stringify({{ sessionId: currentSessionId, files: payloadFiles }}),
      }});
      const saved = Array.isArray(data.files) ? data.files : [];
      for (const s of saved) {{
        if (s && s.stored) pendingAttachments.push(s);
      }}
      renderAttachmentChips();
      setStatus("Attached " + String(saved.length) + " file(s).");
    }}

    async function loadSession(id) {{
      const data = await api("/api/sessions/" + encodeURIComponent(id), {{ method: "GET" }});
      currentSessionId = data.session.id;
      currentMessages = data.session.messages || [];
      chatTitleEl.textContent = data.session.title || "Untitled chat";
      renderSessions();
      renderMessages();
      setStatus("Loaded saved chat.");
    }}

    async function createSession() {{
      const data = await api("/api/sessions", {{ method: "POST", body: JSON.stringify({{}}) }});
      currentSessionId = data.session.id;
      currentMessages = [];
      chatTitleEl.textContent = data.session.title || "New chat";
      await refreshSessions();
      await loadSession(currentSessionId);
      inputEl.focus();
    }}

    async function ensureActiveSession() {{
      if (currentSessionId) return true;
      setStatus("No active chat found. Creating a new chat...");
      try {{
        await createSession();
        return !!currentSessionId;
      }} catch (err) {{
        addBubble(err && err.message ? err.message : String(err), "error");
        setStatus("Could not create a chat session.", "error");
        return false;
      }}
    }}

    async function deleteSession() {{
      if (!currentSessionId) return;
      if (!confirm("Delete this chat session?")) return;
      await api("/api/sessions/" + encodeURIComponent(currentSessionId), {{ method: "DELETE" }});
      currentSessionId = null;
      currentMessages = [];
      messagesEl.innerHTML = "";
      await refreshSessions();
      if (!sessions.length) {{
        await createSession();
      }}
    }}

    async function renameSession(id, currentTitle) {{
      const newTitle = prompt("Rename chat:", currentTitle || "");
      if (newTitle === null) return;
      const title = String(newTitle).trim();
      if (!title) return;
      await api("/api/sessions/" + encodeURIComponent(id) + "/title", {{
        method: "POST",
        body: JSON.stringify({{ title }}),
      }});
      await refreshSessions();
      await loadSession(id);
    }}

    async function exportCurrentSession(format) {{
      if (!currentSessionId) return;
      setStatus("Exporting...");
      try {{
        const data = await api(
          "/api/sessions/" + encodeURIComponent(currentSessionId) + "/export",
          {{
            method: "POST",
            body: JSON.stringify({{ format }}),
          }},
        );
        const content = data.content || "";
        const filename = data.filename || ("chat." + format);
        const type = format === "json" ? "application/json" : "text/plain";
        const blob = new Blob([content], {{ type }});
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(a.href);
        setStatus("Exported " + filename);
      }} catch (err) {{
        setStatus("Export failed.");
        alert(err.message || String(err));
      }}
    }}

    function chatBody(text, mode, selectedModel, attachmentsToSend) {{
      const body = {{ sessionId: currentSessionId, text, mode, model: selectedModel }};
      if (attachmentsToSend && attachmentsToSend.length) {{
        body.attachments = attachmentsToSend;
      }}
      return body;
    }}

    async function sendMessage(payload = null) {{
      let text = "";
      let preShown = false;
      let mode = normalizeMode(uiSettings.mode);
      let selectedModel = normalizeModelName(uiSettings.model || modelSelect.value);
      const isQueuedPayload =
        payload &&
        typeof payload === "object" &&
        Object.prototype.hasOwnProperty.call(payload, "text");
      if (isQueuedPayload) {{
        text = String(payload.text || "").trim();
        preShown = !!payload.preShown;
        mode = normalizeMode(payload.mode);
        selectedModel = normalizeModelName(payload.model);
      }} else {{
        text = inputEl.value.trim();
        mode = normalizeMode(modeSelect.value);
        selectedModel = normalizeModelName(modelSelect.value || uiSettings.model);
      }}
      let attachmentsToSend = [];
      if (isQueuedPayload && Array.isArray(payload.attachments)) {{
        attachmentsToSend = payload.attachments.slice();
      }} else if (!isQueuedPayload) {{
        attachmentsToSend = pendingAttachments.slice();
        pendingAttachments = [];
        renderAttachmentChips();
      }}
      if (!text && !attachmentsToSend.length) {{
        setStatus("Type a message or attach a file.");
        inputEl.focus();
        return;
      }}
      if (!(await ensureActiveSession())) return;

      if (!isQueuedPayload) {{
        inputEl.value = "";
      }}
      const streaming = !!(streamToggle && streamToggle.checked);

      if (isBusy) {{
        queue.push({{ text, mode, model: selectedModel, preShown: true, attachments: attachmentsToSend.slice() }});
        const umsg = {{ role: "user", content: text, createdAt: new Date().toISOString(), pending: true, mode, model: selectedModel }};
        if (attachmentsToSend.length) umsg.attachments = attachmentsToSend.slice();
        currentMessages.push(umsg);
        renderMessages();
        setStatus("Queued (" + queue.length + ") · " + mode.toUpperCase() + " · " + selectedModel);
        return;
      }}

      isBusy = true;
      setProcessing(true);
      sendBtn.disabled = false;
      stopBtn.disabled = !streaming;
      let didAddUserNow = false;

      if (!streaming) {{
        setStatus("Waiting for model reply... (" + mode.toUpperCase() + " · " + selectedModel + ")");
        if (!preShown) {{
          const umsg = {{ role: "user", content: text, createdAt: new Date().toISOString(), mode, model: selectedModel }};
          if (attachmentsToSend.length) umsg.attachments = attachmentsToSend.slice();
          currentMessages.push(umsg);
          didAddUserNow = true;
          renderMessages();
        }}
        try {{
          let data = null;
          try {{
            data = await api("/api/chat", {{
              method: "POST",
              body: JSON.stringify(chatBody(text, mode, selectedModel, attachmentsToSend)),
            }});
          }} catch (firstErr) {{
            const firstMsg = firstErr && firstErr.message ? firstErr.message : String(firstErr || "");
            if (shouldFallbackModel(firstMsg, selectedModel)) {{
              const fallbackModel = normalizeModelName(CFG.model);
              setStatus("Model unavailable. Retrying with " + fallbackModel + "...");
              selectedModel = fallbackModel;
              applyModelSelection(fallbackModel, true);
              data = await api("/api/chat", {{
                method: "POST",
                body: JSON.stringify(chatBody(text, mode, selectedModel, attachmentsToSend)),
              }});
            }} else {{
              throw firstErr;
            }}
          }}
          currentMessages = data.session.messages || [];
          chatTitleEl.textContent = data.session.title || "Untitled chat";
          renderMessages();
          if (data.usage) updateContextBar(data.usage);
          await refreshSessions();
          setStatus("Reply received.");
        }} catch (err) {{
          if (didAddUserNow) {{
            currentMessages.pop();
          }}
          renderMessages();
          addBubble(err.message || String(err), "error");
          setStatus("Request failed.", "error");
        }} finally {{
          sendBtn.disabled = false;
          stopBtn.disabled = true;
          activeController = null;
          isBusy = false;
          setProcessing(false);
          if (queue.length) {{
            const next = queue.shift();
            await sendMessage(next);
            return;
          }}
          inputEl.focus();
        }}
        return;
      }}

      setStatus("Assistant is typing... (" + mode.toUpperCase() + " · " + selectedModel + ")");

      if (activeController) {{
        activeController.abort();
      }}
      activeController = new AbortController();

      if (!preShown) {{
        const umsg = {{ role: "user", content: text, createdAt: new Date().toISOString(), mode, model: selectedModel }};
        if (attachmentsToSend.length) umsg.attachments = attachmentsToSend.slice();
        currentMessages.push(umsg);
        didAddUserNow = true;
      }}
      currentMessages.push({{ role: "assistant", content: "", createdAt: null, mode, model: selectedModel }});
      renderMessages();

      const assistantBubbleEl = Array.from(messagesEl.children).reverse().find((el) => el.classList.contains("assistant"));
      const assistantContentEl = assistantBubbleEl ? assistantBubbleEl.querySelector(".bubble-content") : null;
      const assistantMsg = currentMessages[currentMessages.length - 1];

      let finalSession = null;
      let buf = "";

      const handleSseEvent = (eventName, dataStr) => {{
        if (eventName === "token") {{
          const obj = JSON.parse(dataStr);
          const token = obj.token || "";
          if (token) {{
            assistantMsg.content += token;
            if (assistantContentEl) assistantContentEl.textContent = assistantMsg.content;
            maybeAutoScroll();
          }}
        }} else if (eventName === "context") {{
          const obj = JSON.parse(dataStr);
          updateContextBar(obj);
        }} else if (eventName === "done") {{
          const obj = JSON.parse(dataStr);
          finalSession = obj.session || null;
        }} else if (eventName === "error") {{
          const obj = JSON.parse(dataStr);
          throw new Error(obj.error || "Stream error");
        }}
      }};

      const drainSseBlocks = () => {{
        while (buf.includes("\\n\\n")) {{
          const idx = buf.indexOf("\\n\\n");
          const block = buf.slice(0, idx);
          buf = buf.slice(idx + 2);
          const lines = block.split("\\n");
          let eventName = null;
          let dataStr = null;
          for (const l of lines) {{
            if (l.startsWith("event:")) eventName = l.slice(6).trim();
            if (l.startsWith("data:")) dataStr = l.slice(5).trim();
          }}
          if (eventName && dataStr) handleSseEvent(eventName, dataStr);
        }}
      }};

      const parseSseTail = () => {{
        const tail = buf.trim();
        if (!tail) {{
          buf = "";
          return;
        }}
        const parts = tail.split(/\\n\\n+/);
        for (const part of parts) {{
          const lines = part.split("\\n");
          let eventName = null;
          let dataStr = null;
          for (const l of lines) {{
            if (l.startsWith("event:")) eventName = l.slice(6).trim();
            if (l.startsWith("data:")) dataStr = l.slice(5).trim();
          }}
          if (eventName && dataStr) {{
            try {{
              handleSseEvent(eventName, dataStr);
            }} catch (e) {{
              /* ignore malformed tail frames */
            }}
          }}
        }}
        buf = "";
      }};

      try {{
        let res = await fetch("/api/chat_stream", {{
          method: "POST",
          credentials: "same-origin",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify(chatBody(text, mode, selectedModel, attachmentsToSend)),
          signal: activeController.signal,
        }});

        if (!res.ok) {{
          let errText = "";
          try {{
            errText = await res.text();
          }} catch {{}}
          if (shouldFallbackModel(errText, selectedModel)) {{
            const fallbackModel = normalizeModelName(CFG.model);
            setStatus("Model unavailable. Retrying with " + fallbackModel + "...");
            selectedModel = fallbackModel;
            applyModelSelection(fallbackModel, true);
            assistantMsg.model = selectedModel;
            res = await fetch("/api/chat_stream", {{
              method: "POST",
              credentials: "same-origin",
              headers: {{ "Content-Type": "application/json" }},
              body: JSON.stringify(chatBody(text, mode, selectedModel, attachmentsToSend)),
              signal: activeController.signal,
            }});
            if (!res.ok) {{
              let retryErrText = "";
              try {{
                retryErrText = await res.text();
              }} catch {{}}
              throw new Error(retryErrText || ("HTTP " + res.status));
            }}
          }} else {{
            throw new Error(errText || ("HTTP " + res.status));
          }}
        }}

        const reader = res.body.getReader();
        const decoder = new TextDecoder();

        while (true) {{
          const {{ value, done }} = await reader.read();
          if (value && value.byteLength) {{
            buf += decoder.decode(value, {{ stream: !done }});
          }}
          drainSseBlocks();
          if (done) {{
            buf += decoder.decode(new Uint8Array(), {{ stream: false }});
            drainSseBlocks();
            parseSseTail();
            break;
          }}
        }}

        if (finalSession && finalSession.messages) {{
          currentSessionId = finalSession.id || currentSessionId;
          currentMessages = finalSession.messages || [];
          chatTitleEl.textContent = finalSession.title || "Untitled chat";
          renderMessages();
          setStatus("Reply received.");
          try {{
            await refreshSessions();
          }} catch (e) {{
            /* keep UI responsive if session list refresh fails */
          }}
        }} else {{
          setStatus("Reply received (stream ended).");
        }}
      }} catch (err) {{
        currentMessages = currentMessages.filter((m, i) => !(i === currentMessages.length - 1 && m.role === "assistant" && (m.content === "" || !finalSession)));
        if (didAddUserNow && !finalSession) {{
          currentMessages = currentMessages.filter((m, i) => !(i === currentMessages.length - 1 && m.role === "user"));
        }}
        renderMessages();
        if (err && err.name === "AbortError") {{
          setStatus("Stopped.");
        }} else {{
          addBubble(err.message || String(err), "error");
          setStatus("Request failed.", "error");
        }}
      }} finally {{
        sendBtn.disabled = false;
        stopBtn.disabled = true;
        activeController = null;
        isBusy = false;
        setProcessing(false);
        const st = statusTextEl.textContent || "";
        if (st.includes("Assistant is typing")) {{
          setStatus("Ready.");
        }}
        if (queue.length) {{
          const next = queue.shift();
          await sendMessage(next);
          return;
        }}
        inputEl.focus();
      }}
    }}

    async function login() {{
      loginError.textContent = "";
      loginError.classList.remove("error");
      try {{
        await api("/api/login", {{
          method: "POST",
          body: JSON.stringify({{
            username: loginUser.value.trim(),
            password: loginPass.value,
          }}),
        }});
        setLoggedIn(true);
        await refreshModels(true);
        await refreshSessions();
        if (!sessions.length) {{
          await createSession();
        }}
      }} catch (err) {{
        loginError.textContent = err.message || String(err);
        loginError.classList.add("error");
      }}
    }}

    async function logout() {{
      await api("/api/logout", {{ method: "POST", body: JSON.stringify({{}}) }});
      currentSessionId = null;
      currentMessages = [];
      sessions = [];
      setLoggedIn(false);
    }}

    async function init() {{
      try {{
        await api("/api/me", {{ method: "GET" }});
        setLoggedIn(true);
        await refreshModels(true);
        await refreshSessions();
        if (!sessions.length) {{
          await createSession();
        }}
      }} catch {{
        setLoggedIn(false);
      }}
    }}

    loginBtn.addEventListener("click", login);
    loginPass.addEventListener("keydown", (e) => {{
      if (e.key === "Enter") login();
    }});
    newChatBtn.addEventListener("click", createSession);
    deleteChatBtn.addEventListener("click", deleteSession);
    const renameChatBtn = document.getElementById("renameChatBtn");
    renameChatBtn.addEventListener("click", () => {{
      if (!currentSessionId) return;
      const cur = chatTitleEl.textContent || "";
      renameSession(currentSessionId, cur);
    }});
    exportTxtBtn.addEventListener("click", () => exportCurrentSession("txt"));
    exportJsonBtn.addEventListener("click", () => exportCurrentSession("json"));
    settingsBtn.addEventListener("click", () => {{
      settingsPanel.classList.add("open");
    }});
    closeSettingsBtn.addEventListener("click", () => {{
      settingsPanel.classList.remove("open");
    }});
    profileNameInput.addEventListener("input", () => {{
      uiSettings.profileName = profileNameInput.value.trim();
      saveSettings();
      updateSubtitle();
    }});
    modeSelect.addEventListener("change", () => {{
      uiSettings.mode = normalizeMode(modeSelect.value);
      modeSelect.value = uiSettings.mode;
      saveSettings();
      setStatus("Mode: " + uiSettings.mode.toUpperCase());
    }});
    modelSelect.addEventListener("change", () => {{
      applyModelSelection(modelSelect.value, true);
      setStatus("Model: " + modelSubtitleLabel());
    }});
    modelSearchInput.addEventListener("input", () => {{
      modelFilterQuery = modelSearchInput.value || "";
      rebuildModelOptions(uiSettings.model || AUTO_MODEL_TOKEN);
      modelSelect.value = normalizeModelName(uiSettings.model || AUTO_MODEL_TOKEN);
    }});
    refreshModelsBtn.addEventListener("click", async () => {{
      await refreshModels(false);
    }});
    themeSelect.addEventListener("change", () => {{
      uiSettings.theme = (themeSelect.value === "light" || themeSelect.value === "system") ? themeSelect.value : "dark";
      saveSettings();
      applyTheme(uiSettings.theme);
    }});
    spellcheckToggle.addEventListener("change", () => {{
      uiSettings.spellcheck = !!spellcheckToggle.checked;
      saveSettings();
      applySpellcheck(uiSettings.spellcheck);
    }});
    if (systemThemeMedia) {{
      const onSystemThemeChanged = () => {{
        if (uiSettings.theme === "system") {{
          applyTheme("system");
        }}
      }};
      if (typeof systemThemeMedia.addEventListener === "function") {{
        systemThemeMedia.addEventListener("change", onSystemThemeChanged);
      }} else if (typeof systemThemeMedia.addListener === "function") {{
        systemThemeMedia.addListener(onSystemThemeChanged);
      }}
    }}
    chatHeightRange.addEventListener("input", () => {{
      uiSettings.chatHeight = Number(chatHeightRange.value) || 100;
      saveSettings();
      applyChatHeight(uiSettings.chatHeight);
    }});
    stopBtn.addEventListener("click", () => {{
      if (activeController) {{
        activeController.abort();
        activeController = null;
      }}
      stopBtn.disabled = true;
      sendBtn.disabled = false;
      setStatus("Stopped.");
      setProcessing(false);
      inputEl.focus();
    }});
    logoutBtn.addEventListener("click", logout);
    sendBtn.addEventListener("click", () => sendMessage());
    searchBoxEl.addEventListener("input", renderSessions);
    inputEl.addEventListener("keydown", (e) => {{
      if (e.key === "Enter" && !e.shiftKey) {{
        e.preventDefault();
        sendMessage();
      }}
    }});
    inputEl.addEventListener("focus", () => {{
      applySpellcheck(uiSettings.spellcheck);
    }});
    attachBtn.addEventListener("click", () => {{
      if (!currentSessionId) {{
        setStatus("Open or create a chat first.", "error");
        return;
      }}
      fileInput.click();
    }});
    fileInput.addEventListener("change", async () => {{
      if (!fileInput.files || !fileInput.files.length) return;
      try {{
        await uploadPendingFiles(fileInput.files);
      }} catch (err) {{
        setStatus((err && err.message) || String(err), "error");
      }}
      fileInput.value = "";
    }});

    init();
  </script>
</body>
</html>
"""
    return page.encode("utf-8")


def make_handler(
    model: str,
    ollama_host: str,
    system_prompt: str | None,
    html_bytes: bytes,
    store: SessionStore,
    context_max_messages: int,
    uploads_root: Path,
):
    auth_sessions: dict[str, str] = {}
    auth_lock = threading.Lock()

    class ChatHandler(BaseHTTPRequestHandler):
        server_version = "LocalWebChat/2.0"

        def log_message(self, fmt: str, *args) -> None:
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def _send(self, code: int, body: bytes, content_type: str, extra_headers: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            if extra_headers:
                for key, value in extra_headers.items():
                    self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, payload: dict, extra_headers: dict | None = None) -> None:
            self._send(code, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8", extra_headers)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8") if length else "{}"
            return json.loads(raw or "{}")

        def _cookie_token(self) -> str | None:
            raw = self.headers.get("Cookie")
            if not raw:
                return None
            jar = cookies.SimpleCookie()
            jar.load(raw)
            morsel = jar.get(AUTH_COOKIE)
            return morsel.value if morsel else None

        def _require_auth(self) -> bool:
            token = self._cookie_token()
            if not token:
                return False
            with auth_lock:
                return token in auth_sessions

        def _set_auth_cookie(self, token: str) -> dict:
            return {"Set-Cookie": f"{AUTH_COOKIE}={token}; HttpOnly; Path=/; SameSite=Strict"}

        def _clear_auth_cookie(self) -> dict:
            return {"Set-Cookie": f"{AUTH_COOKIE}=; HttpOnly; Path=/; Max-Age=0; SameSite=Strict"}

        def _session_or_404(self, session_id: str) -> dict | None:
            session = store.load_session(session_id)
            if session is None:
                self._json(404, {"error": "Session not found"})
            return session

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                self._send(200, html_bytes, "text/html; charset=utf-8")
                return

            if path == "/api/me":
                if not self._require_auth():
                    self._json(401, {"error": "Not logged in"})
                    return
                self._json(200, {"username": LOGIN_USER})
                return

            if not self._require_auth():
                self._json(401, {"error": "Login required"})
                return

            if path == "/api/sessions":
                self._json(200, {"sessions": store.list_sessions()})
                return

            if path == "/api/models":
                try:
                    models = ollama_list_models(ollama_host)
                    sizes = ollama_model_sizes(ollama_host)
                    auto_pick = pick_auto_model(ollama_host, model)
                except urllib.error.HTTPError as e:
                    detail = e.read().decode("utf-8", errors="replace") if e.fp else str(e)
                    self._json(502, {"error": format_ollama_http_error(e.code, detail)})
                    return
                except urllib.error.URLError as e:
                    self._json(502, {"error": f"Cannot reach Ollama at http://{ollama_host}/ — {e.reason or e}"})
                    return
                self._json(200, {"models": models, "modelSizes": sizes, "autoModel": auto_pick})
                return

            if path.startswith("/api/sessions/"):
                session_id = path.rsplit("/", 1)[-1]
                session = self._session_or_404(session_id)
                if session is not None:
                    self._json(200, {"session": session})
                return

            self.send_error(404)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                payload = self._read_json()
            except json.JSONDecodeError:
                self._json(400, {"error": "Invalid JSON"})
                return

            if path == "/api/login":
                username = str(payload.get("username", ""))
                password = str(payload.get("password", ""))
                password_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
                if username != LOGIN_USER or password_hash != LOGIN_PASSWORD_HASH:
                    self._json(401, {"error": "Invalid username or password"})
                    return
                token = secrets.token_urlsafe(24)
                with auth_lock:
                    auth_sessions[token] = username
                self._json(200, {"ok": True, "username": username}, self._set_auth_cookie(token))
                return

            if path == "/api/logout":
                token = self._cookie_token()
                if token:
                    with auth_lock:
                        auth_sessions.pop(token, None)
                self._json(200, {"ok": True}, self._clear_auth_cookie())
                return

            if not self._require_auth():
                self._json(401, {"error": "Login required"})
                return

            if path == "/api/sessions":
                title = payload.get("title")
                session = store.create_session(title if isinstance(title, str) and title.strip() else None)
                self._json(200, {"session": session})
                return

            # Rename a session: POST /api/sessions/<sessionId>/title
            if path.startswith("/api/sessions/") and path.endswith("/title"):
                parts = path.strip("/").split("/")
                # api / sessions / {id} / title
                if len(parts) == 4 and parts[0] == "api" and parts[1] == "sessions":
                    session_id = parts[2]
                    session = store.load_session(session_id)
                    if session is None:
                        self._json(404, {"error": "Session not found"})
                        return
                    new_title = str(payload.get("title", "")).strip()
                    if not new_title:
                        self._json(400, {"error": "title is required"})
                        return
                    session["title"] = new_title
                    store.save_session(session)
                    self._json(200, {"ok": True, "session": session})
                    return
                self._json(400, {"error": "Invalid request path"})
                return

            # Export a session: POST /api/sessions/<sessionId>/export with { "format": "txt"|"json" }
            if path.startswith("/api/sessions/") and path.endswith("/export"):
                parts = path.strip("/").split("/")
                if len(parts) == 4 and parts[0] == "api" and parts[1] == "sessions":
                    session_id = parts[2]
                    session = store.load_session(session_id)
                    if session is None:
                        self._json(404, {"error": "Session not found"})
                        return

                    fmt = str(payload.get("format", "txt")).strip().lower()
                    if fmt not in {"txt", "json"}:
                        self._json(400, {"error": "format must be 'txt' or 'json'"})
                        return

                    title = session.get("title") or "Untitled chat"
                    messages = session.get("messages") or []
                    exported_at = utc_now()

                    if fmt == "json":
                        content = json.dumps(session, ensure_ascii=False, indent=2)
                        filename = f"{sanitize_filename(str(title))}.json"
                    else:
                        # Plain text export (easy to read/share)
                        lines: list[str] = []
                        lines.append(str(title))
                        lines.append(f"Session ID: {session_id}")
                        lines.append(f"ExportedAt: {exported_at}")
                        lines.append("")
                        for m in messages:
                            role = m.get("role") or "user"
                            content_text = m.get("content") or ""
                            created = m.get("createdAt")
                            prefix = f"[{created}] " if isinstance(created, str) and created else ""
                            lines.append(f"{prefix}{role.upper()}: {content_text}".strip())
                            lines.append("")
                        content = "\n".join(lines).rstrip() + "\n"
                        filename = f"{sanitize_filename(str(title))}.txt"

                    self._json(200, {"content": content, "filename": filename})
                    return

                self._json(400, {"error": "Invalid request path"})
                return

            if path == "/api/upload":
                session_id_u = payload.get("sessionId")
                if not isinstance(session_id_u, str) or not session_id_u.strip():
                    self._json(400, {"error": "sessionId is required"})
                    return
                sid_u = session_id_u.strip()
                session_u = self._session_or_404(sid_u)
                if session_u is None:
                    return
                files_in = payload.get("files")
                if not isinstance(files_in, list) or not files_in:
                    self._json(400, {"error": "files array is required"})
                    return
                uploads_root.mkdir(parents=True, exist_ok=True)
                out_dir = uploads_root / sid_u
                out_dir.mkdir(parents=True, exist_ok=True)
                total_bytes = 0
                saved_files: list[dict] = []
                for item in files_in[:16]:
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("name") or "file")
                    b64 = item.get("data")
                    if not isinstance(b64, str):
                        continue
                    try:
                        raw = base64.b64decode(b64, validate=False)
                    except (ValueError, TypeError):
                        continue
                    total_bytes += len(raw)
                    if total_bytes > MAX_UPLOAD_TOTAL_BYTES:
                        self._json(413, {"error": "Total upload size exceeds limit"})
                        return
                    mime_guess = str(item.get("mime") or "") or (mimetypes.guess_type(name)[0] or "application/octet-stream")
                    safe_name = sanitize_filename(Path(name).name, max_len=120)
                    uniq = f"{secrets.token_hex(6)}_{safe_name}"
                    target = out_dir / uniq
                    try:
                        target.write_bytes(raw)
                    except OSError:
                        self._json(500, {"error": "Could not save upload"})
                        return
                    stored = f"{sid_u}/{uniq}"
                    kind = _attachment_kind(safe_name, mime_guess)
                    saved_files.append(
                        {"name": name, "mime": mime_guess, "stored": stored, "kind": kind, "size": len(raw)}
                    )
                self._json(200, {"ok": True, "files": saved_files})
                return

            if path == "/api/chat_stream":
                session_id = payload.get("sessionId")
                text = payload.get("text")
                mode = normalize_mode(str(payload.get("mode", "agent")))
                requested_raw = payload.get("model")
                requested_label = (
                    AUTO_MODEL
                    if (requested_raw is None or is_auto_model(str(requested_raw)))
                    else str(requested_raw or "").strip()[:120]
                )
                eff_model = resolve_effective_model(ollama_host, str(requested_raw) if requested_raw is not None else "", model)
                if not isinstance(session_id, str) or not isinstance(text, str):
                    self._json(400, {"error": "sessionId and text are required"})
                    return

                session = self._session_or_404(session_id)
                if session is None:
                    return

                user_text = (text or "").strip()
                session_messages = session.get("messages", [])
                now = utc_now()
                att_clean: list[dict] = []
                raw_atts = payload.get("attachments")
                if isinstance(raw_atts, list):
                    for a in raw_atts:
                        if not isinstance(a, dict):
                            continue
                        st = a.get("stored")
                        if not isinstance(st, str):
                            continue
                        pth = _safe_upload_file_path(uploads_root, session_id, st.replace("\\", "/"))
                        if pth is None or not pth.is_file():
                            continue
                        nm = str(a.get("name") or pth.name)
                        mime = str(a.get("mime") or "")
                        att_clean.append(
                            {
                                "name": nm,
                                "mime": mime,
                                "stored": st.replace("\\", "/"),
                                "kind": str(a.get("kind") or _attachment_kind(nm, mime)),
                            }
                        )
                if not user_text and not att_clean:
                    self._json(400, {"error": "message text or attachments required"})
                    return
                user_msg: dict = {"role": "user", "content": user_text or "(see attachments)", "createdAt": now}
                if att_clean:
                    user_msg["attachments"] = att_clean
                session_messages.append(user_msg)
                if session.get("title") == "New chat" and (user_text or att_clean):
                    session["title"] = derive_title_from_query(user_text or "Attachment")
                session["messages"] = session_messages
                store.save_session(session)

                num_ctx = ollama_show_num_ctx(ollama_host, eff_model) or 8192
                active_context_limit = context_max_messages
                full = build_ollama_messages(
                    session_messages, system_prompt, mode, active_context_limit, uploads_root, session_id
                )

                # SSE response
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                # We use a per-request SSE stream; explicitly close after `done`
                # so the browser completes `fetch()` and the UI can reliably exit
                # the "Assistant is typing..." state.
                self.send_header("Connection", "close")
                self.end_headers()

                def sse_event(event: str, data_obj: dict) -> None:
                    block = f"event: {event}\ndata: {json.dumps(data_obj, ensure_ascii=False)}\n\n".encode("utf-8")
                    self.wfile.write(block)
                    if hasattr(self.wfile, "flush"):
                        self.wfile.flush()

                try:
                    sse_event("status", {"status": "starting", "model": eff_model, "numCtx": num_ctx})
                except Exception:
                    return

                reply_text = ""
                last_usage: dict = {}
                try:
                    while True:
                        try:
                            for kind, chunk in ollama_chat_stream_iter(ollama_host, eff_model, full):
                                if kind == "token":
                                    reply_text += chunk
                                    try:
                                        sse_event("token", {"token": chunk})
                                    except (BrokenPipeError, ConnectionResetError, OSError):
                                        return
                                elif kind == "usage" and isinstance(chunk, dict):
                                    last_usage = dict(chunk)
                            break
                        except urllib.error.HTTPError as e:
                            detail = e.read().decode("utf-8", errors="replace") if e.fp else str(e)
                            can_retry = (
                                not reply_text
                                and is_ollama_runner_terminated_error(e.code, detail)
                                and active_context_limit > MIN_CONTEXT_MAX_MESSAGES
                            )
                            if not can_retry:
                                raise
                            next_limit = reduced_context_limit(active_context_limit)
                            if next_limit >= active_context_limit:
                                raise
                            active_context_limit = next_limit
                            full = build_ollama_messages(
                                session_messages, system_prompt, mode, active_context_limit, uploads_root, session_id
                            )
                            sse_event(
                                "status",
                                {
                                    "status": "retrying",
                                    "reason": "runner-terminated",
                                    "contextMaxMessages": active_context_limit,
                                },
                            )

                    usage_out = dict(last_usage)
                    usage_out["num_ctx"] = num_ctx
                    usage_out["model"] = eff_model
                    usage_out["requested_model"] = requested_label
                    pr = usage_out.get("prompt_eval_count")
                    ev = usage_out.get("eval_count")
                    if isinstance(pr, int) and isinstance(ev, int):
                        try:
                            sse_event(
                                "context",
                                {
                                    "prompt_eval_count": pr,
                                    "eval_count": ev,
                                    "total_tokens": pr + ev,
                                    "num_ctx": num_ctx,
                                    "model": eff_model,
                                },
                            )
                        except Exception:
                            pass

                    session_messages = session.get("messages", [])
                    assistant_message: dict = {
                        "role": "assistant",
                        "content": reply_text,
                        "createdAt": utc_now(),
                        "mode": mode,
                        "model": eff_model,
                        "requestedModel": requested_label,
                        "usage": usage_out,
                    }
                    if active_context_limit != context_max_messages:
                        assistant_message["contextLimit"] = active_context_limit
                    session_messages.append(assistant_message)
                    session["messages"] = session_messages
                    store.save_session(session)
                    sse_event("done", {"session": session})
                    self.close_connection = True
                    return

                except urllib.error.HTTPError as e:
                    detail = e.read().decode("utf-8", errors="replace") if e.fp else str(e)
                    try:
                        sse_event("error", {"error": format_ollama_http_error(e.code, detail)})
                    except Exception:
                        pass
                    self.close_connection = True
                    return
                except urllib.error.URLError as e:
                    try:
                        sse_event(
                            "error",
                            {"error": f"Cannot reach Ollama at http://{ollama_host}/ — {e.reason or e}"},
                        )
                    except Exception:
                        pass
                    self.close_connection = True
                    return
                except Exception as e:
                    try:
                        sse_event("error", {"error": f"Streaming failed: {e}"})
                    except Exception:
                        pass
                    return

            if path == "/api/chat":
                session_id = payload.get("sessionId")
                text = payload.get("text")
                mode = normalize_mode(str(payload.get("mode", "agent")))
                requested_raw = payload.get("model")
                requested_label = (
                    AUTO_MODEL
                    if (requested_raw is None or is_auto_model(str(requested_raw)))
                    else str(requested_raw or "").strip()[:120]
                )
                eff_model = resolve_effective_model(ollama_host, str(requested_raw) if requested_raw is not None else "", model)
                if not isinstance(session_id, str) or not isinstance(text, str):
                    self._json(400, {"error": "sessionId and text are required"})
                    return
                session = self._session_or_404(session_id)
                if session is None:
                    return

                user_text = (text or "").strip()
                session_messages = session.get("messages", [])
                now = utc_now()
                att_clean2: list[dict] = []
                raw_atts2 = payload.get("attachments")
                if isinstance(raw_atts2, list):
                    for a in raw_atts2:
                        if not isinstance(a, dict):
                            continue
                        st = a.get("stored")
                        if not isinstance(st, str):
                            continue
                        pth = _safe_upload_file_path(uploads_root, session_id, st.replace("\\", "/"))
                        if pth is None or not pth.is_file():
                            continue
                        nm = str(a.get("name") or pth.name)
                        mime = str(a.get("mime") or "")
                        att_clean2.append(
                            {
                                "name": nm,
                                "mime": mime,
                                "stored": st.replace("\\", "/"),
                                "kind": str(a.get("kind") or _attachment_kind(nm, mime)),
                            }
                        )
                if not user_text and not att_clean2:
                    self._json(400, {"error": "message text or attachments required"})
                    return
                user_msg2: dict = {"role": "user", "content": user_text or "(see attachments)", "createdAt": now}
                if att_clean2:
                    user_msg2["attachments"] = att_clean2
                session_messages.append(user_msg2)
                if session.get("title") == "New chat" and (user_text or att_clean2):
                    session["title"] = derive_title_from_query(user_text or "Attachment")

                num_ctx = ollama_show_num_ctx(ollama_host, eff_model) or 8192
                active_context_limit = context_max_messages
                full = build_ollama_messages(
                    session_messages, system_prompt, mode, active_context_limit, uploads_root, session_id
                )
                try:
                    reply, usage_ollama = ollama_chat(ollama_host, eff_model, full)
                except urllib.error.HTTPError as e:
                    detail = e.read().decode("utf-8", errors="replace") if e.fp else str(e)
                    can_retry = (
                        is_ollama_runner_terminated_error(e.code, detail)
                        and active_context_limit > MIN_CONTEXT_MAX_MESSAGES
                    )
                    if can_retry:
                        next_limit = reduced_context_limit(active_context_limit)
                        if next_limit < active_context_limit:
                            active_context_limit = next_limit
                            full = build_ollama_messages(
                                session_messages, system_prompt, mode, active_context_limit, uploads_root, session_id
                            )
                            try:
                                reply, usage_ollama = ollama_chat(ollama_host, eff_model, full)
                            except urllib.error.HTTPError as retry_err:
                                retry_detail = (
                                    retry_err.read().decode("utf-8", errors="replace") if retry_err.fp else str(retry_err)
                                )
                                self._json(502, {"error": format_ollama_http_error(retry_err.code, retry_detail)})
                                return
                            except urllib.error.URLError as retry_err:
                                self._json(
                                    502, {"error": f"Cannot reach Ollama at http://{ollama_host}/ — {retry_err.reason or retry_err}"}
                                )
                                return
                        else:
                            self._json(502, {"error": format_ollama_http_error(e.code, detail)})
                            return
                    else:
                        self._json(502, {"error": format_ollama_http_error(e.code, detail)})
                        return
                except urllib.error.URLError as e:
                    self._json(502, {"error": f"Cannot reach Ollama at http://{ollama_host}/ — {e.reason or e}"})
                    return

                usage_out = dict(usage_ollama) if isinstance(usage_ollama, dict) else {}
                usage_out["num_ctx"] = num_ctx
                usage_out["model"] = eff_model
                usage_out["requested_model"] = requested_label
                pr = usage_out.get("prompt_eval_count")
                ev = usage_out.get("eval_count")
                if isinstance(pr, int) and isinstance(ev, int):
                    usage_out["total_tokens"] = pr + ev

                assistant_message = {
                    "role": "assistant",
                    "content": reply,
                    "createdAt": utc_now(),
                    "mode": mode,
                    "model": eff_model,
                    "requestedModel": requested_label,
                    "usage": usage_out,
                }
                if active_context_limit != context_max_messages:
                    assistant_message["contextLimit"] = active_context_limit
                session_messages.append(assistant_message)
                session["messages"] = session_messages
                store.save_session(session)
                self._json(200, {"reply": reply, "session": session, "usage": usage_out})
                return

            self.send_error(404)

        def do_DELETE(self) -> None:
            path = urlparse(self.path).path
            if not self._require_auth():
                self._json(401, {"error": "Login required"})
                return
            if path.startswith("/api/sessions/"):
                session_id = path.rsplit("/", 1)[-1]
                if store.delete_session(session_id):
                    self._json(200, {"ok": True})
                else:
                    self._json(404, {"error": "Session not found"})
                return
            self.send_error(404)

    return ChatHandler


def main() -> int:
    env_context_max = CONTEXT_MAX_MESSAGES
    env_context_raw = os.getenv("WEB_CHAT_CONTEXT_MAX_MESSAGES")
    if env_context_raw:
        try:
            env_context_max = normalize_context_max_messages(int(env_context_raw), CONTEXT_MAX_MESSAGES)
        except ValueError:
            print(
                f"Ignoring invalid WEB_CHAT_CONTEXT_MAX_MESSAGES={env_context_raw!r}; using {CONTEXT_MAX_MESSAGES}.",
                file=sys.stderr,
            )

    parser = argparse.ArgumentParser(description="Local HTTP web UI for Ollama chat")
    parser.add_argument("--listen", default=DEFAULT_LISTEN, help=f"Bind address (default: {DEFAULT_LISTEN})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port (default: {DEFAULT_PORT})")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model name (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--ollama-host",
        default=DEFAULT_OLLAMA,
        metavar="HOST:PORT",
        help=f"Ollama API (default: {DEFAULT_OLLAMA})",
    )
    parser.add_argument(
        "--context-max-messages",
        type=int,
        default=env_context_max,
        help=(
            "Max recent user/assistant messages sent to Ollama each turn "
            f"(default: {env_context_max}, min: {MIN_CONTEXT_MAX_MESSAGES}, max: {MAX_CONTEXT_MAX_MESSAGES})"
        ),
    )
    rg = parser.add_mutually_exclusive_group()
    rg.add_argument("--role-file", type=Path, metavar="PATH", help="Optional system prompt .txt file")
    rg.add_argument("--role", metavar="NAME", help=f"Load roles/{{NAME}}.txt")
    args = parser.parse_args()
    context_max_messages = normalize_context_max_messages(args.context_max_messages, CONTEXT_MAX_MESSAGES)

    role_path = resolve_role_path(args.role, args.role_file)
    system_prompt: str | None = None
    if role_path is not None:
        if not role_path.is_file():
            print(f"Role file not found: {role_path}", file=sys.stderr)
            return 1
        try:
            system_prompt = load_role_file(role_path)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 1

    store = SessionStore(SESSIONS_DIR)
    uploads_root = SESSIONS_DIR / "_uploads"
    uploads_root.mkdir(parents=True, exist_ok=True)
    html = build_app_html(args.model, args.ollama_host, system_prompt is not None)
    handler = make_handler(
        args.model, args.ollama_host, system_prompt, html, store, context_max_messages, uploads_root
    )

    try:
        httpd = ThreadingHTTPServer((args.listen, args.port), handler)
    except OSError as e:
        print(f"Could not bind {args.listen}:{args.port} — {e}", file=sys.stderr)
        return 1

    url = f"http://{args.listen}:{args.port}/"
    print(f"Serving {url}", flush=True)
    print(f"Context window (messages): {context_max_messages}", flush=True)
    if not os.getenv("WEB_CHAT_PASSWORD_HASH") and os.getenv("WEB_CHAT_PASSWORD") is None:
        print(
            f"Auth: using default login (username '{LOGIN_USER}', password 'change-me'). "
            "Set WEB_CHAT_PASSWORD or WEB_CHAT_PASSWORD_HASH for anything beyond local demo.",
            file=sys.stderr,
        )
    print("Press Ctrl+C to stop.", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
