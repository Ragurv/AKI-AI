# AGENTS.md

## Cursor Cloud specific instructions

### Overview
AKI-AI Local Chat is a stdlib-only Python web/CLI chat application for Ollama LLMs. There are **zero PyPI dependencies** — the entire project uses only the Python standard library.

### Services

| Service | How to start | Notes |
|---------|-------------|-------|
| **Ollama** (LLM server) | `ollama serve` | Must be running before web_chat.py or CLI scripts. Binds to `127.0.0.1:11434`. Start in a background tmux session. |
| **Web UI** | `python3 web_chat.py --listen 0.0.0.0` | Serves on port 8088. Use `--listen 0.0.0.0` in Cloud VMs so the browser pane can reach it. |
| **CLI chat** | `python3 cli_chat.py` | Interactive terminal chat (requires Ollama running). |
| **Role chat** | `python3 role_chat.py --role-file roles/example_assistant.txt` | Terminal chat with a system prompt file. |

### Key caveats

- **Ollama must be started manually** — systemd is not available in Cloud Agent VMs. Run `ollama serve` in a tmux session before starting the app.
- **At least one model must be pulled** before the app can generate responses. Use `ollama pull qwen2:0.5b` for a small (~350 MB) test model, or `ollama pull dolphin-mistral` for the project's default model.
- **Default login credentials** (when no env vars set): username `demo`, password `change-me`.
- **No build step** — `web_chat.py` embeds all HTML/CSS/JS as Python string literals.
- **Lint**: `python3 -m ruff check .` (ruff is installed via the update script).
- **No automated test suite** exists in this repository.
- Chat sessions are persisted as JSON files in `chat_sessions/` (git-ignored).
