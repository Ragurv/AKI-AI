# AKI-AI Local Chat (`aki-ai-local-chat`)

Local web and CLI chat project for running Ollama models on a laptop (CPU-friendly), with login, session history, streaming, queueing, and role prompts.

## Project Files

- `web_chat.py`: local HTTP web UI with login, sessions, streaming, queue, settings, export.
- `cli_chat.py`: terminal chat client for direct model conversations.
- `role_chat.py`: terminal chat client using a custom role/system prompt file.
- `roles/example_assistant.txt`: starter role prompt.
- `chat_sessions/`: persisted per-chat JSON sessions.
- `chat_sessions/_uploads/`: uploaded attachments per session (created automatically).

## Features

- Local login and cookie session auth.
- Chat session list, rename, delete, search, and export (TXT/JSON).
- Streaming and non-streaming modes.
- Message queue while generation is in progress.
- ASK / PLANNING / AGENT response mode selector.
- LLM model selector with **Auto** default (picks smallest installed model by disk size per request).
- File **Attach** (images for multimodal models; text/code files inlined into the prompt).
- **Context** usage line after each reply (Ollama prompt/completion counts vs estimated context window).
- Theme support: Dark, Light, and System (Auto).
- Input spellcheck toggle and chat-height control.

## Security (before GitHub or any shared host)

Login is **not** hardcoded. Set one of:

| Variable | Purpose |
|----------|---------|
| `WEB_CHAT_USER` | Username (default: `demo`) |
| `WEB_CHAT_USERNAME` | Alias for `WEB_CHAT_USER` (for compatibility) |
| `WEB_CHAT_PASSWORD` | Plain password; hashed at startup (default: `change-me` if nothing else set) |
| `WEB_CHAT_PASSWORD_HASH` | Optional: precomputed SHA-256 hex (64 chars) instead of plain password |

For public demos, use a strong random password and **never** commit `.env` (see `.gitignore`).

## Prerequisites

- Python 3.10+.
- Ollama installed and running locally.
- At least one model pulled (example: `ollama pull dolphin-mistral`).
- No PyPI install is required for `web_chat.py`, `cli_chat.py`, or `role_chat.py` (stdlib-only project).

## Run

1. Install and start Ollama.
2. Pull model (example): `ollama pull dolphin-mistral`
3. Start web UI:
   - `python web_chat.py`
4. Open: `http://127.0.0.1:8088`

Default local login (when no auth env vars are set):

- Username: `demo`
- Password: `change-me`

Optional role file:

- `python web_chat.py --role-file roles\example_assistant.txt`

CLI modes:

- `python cli_chat.py`
- `python role_chat.py --role-file roles\example_assistant.txt`

## Demo for portfolio (DWDM / OTN + AI)

See **[demos/README.md](demos/README.md)** for a short screen-recording script and a synthetic OTN-style sample file you can attach in the UI.

## Publishing to GitHub

1. Create a new empty repo on GitHub (no README if you already have one locally).
   - Recommended repo name: `aki-ai-local-chat`
2. In this folder: `git init` → add remote → `git add` / `git commit` / `git push`.
3. Confirm `chat_sessions/` JSON and `_uploads/` are **ignored** (`.gitignore`) so chats and files are not pushed.
4. In the repo **About** description, add tags like `python`, `ollama`, `local-llm`, `streaming`, `portfolio`.

## Troubleshooting

- **Invalid username or password**
  - If you set `WEB_CHAT_USER` / `WEB_CHAT_PASSWORD` / `WEB_CHAT_PASSWORD_HASH`, restart `python web_chat.py` in the same terminal session where those env vars are set.
  - If unsure, clear the auth env vars and use default local login: `demo` / `change-me`.
- **Assistant appears stuck at “typing”**
  - Use the latest code and restart the server. The streaming endpoint now closes after `done` so the UI exits typing state correctly.
  - Hard refresh the browser (`Ctrl+F5`) to pick up updated frontend JS.
- **Ollama 500 / runner terminated**
  - Retry, reduce context pressure, or switch to a smaller model.

## AI-DLC Workflow

This project uses the AI-DLC workflow guidance integrated in Cursor rules. Keep changes aligned with the existing AI-DLC rule files in the repository and follow that loop for implementation, validation, and iteration.
