"""
Local role-based chat: same stack as cli_chat.py (Ollama + Dolphin-Mistral),
but the model always sees YOUR system role from a text file.

1) Put your instructions in a file, e.g. roles/my_assistant.txt
2) Run:
     python role_chat.py --role-file roles/my_assistant.txt
   Or name under roles/:
     python role_chat.py --role my_assistant

Commands while chatting:
  /q, /quit     exit
  /reload       re-read the role file from disk (same path you started with)

This is local assistant software. You are responsible for your prompts and usage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import urllib.error
import urllib.request

# Reuse defaults from dolphin flow
DEFAULT_MODEL = "dolphin-mistral"
DEFAULT_HOST = "127.0.0.1:11434"
ROLES_DIR = Path(__file__).resolve().parent / "roles"


def ollama_chat(host: str, model: str, messages: list[dict]) -> str:
    url = f"http://{host}/api/chat"
    body = json.dumps({"model": model, "messages": messages, "stream": False}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.load(resp)
    msg = data.get("message") or {}
    return (msg.get("content") or "").strip()


def load_role_text(role_file: Path) -> str:
    if not role_file.is_file():
        print(f"Role file not found: {role_file}", file=sys.stderr)
        raise SystemExit(1)
    text = role_file.read_text(encoding="utf-8").strip()
    if not text:
        print(f"Role file is empty: {role_file}", file=sys.stderr)
        raise SystemExit(1)
    return text


def resolve_role_path(role: str | None, role_file: Path | None) -> Path:
    if role_file is not None:
        return role_file.expanduser().resolve()
    if role:
        return (ROLES_DIR / f"{role}.txt").resolve()
    default = ROLES_DIR / "example_assistant.txt"
    if default.is_file():
        return default
    print(
        "Pass --role NAME (loads roles/NAME.txt) or --role-file PATH.txt\n"
        f"Optional template: {default}",
        file=sys.stderr,
    )
    raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Role-based chat with custom system prompt (Ollama)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"host:port (default: {DEFAULT_HOST})")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--role", metavar="NAME", help=f"Load roles/{{NAME}}.txt (dir: {ROLES_DIR})")
    g.add_argument("--role-file", type=Path, metavar="PATH", help="Full path to role .txt file")
    args = parser.parse_args()

    role_path = resolve_role_path(args.role, args.role_file)
    system_content = load_role_text(role_path)

    messages: list[dict] = [{"role": "system", "content": system_content}]
    print(f"Model: {args.model} | Role file: {role_path}")
    print("Commands: /reload  /q  /quit\n")

    while True:
        try:
            line = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        low = line.lower()
        if low in {"/q", "/quit", "/exit"}:
            break
        if low == "/reload":
            system_content = load_role_text(role_path)
            messages[0] = {"role": "system", "content": system_content}
            print("(Role reloaded from disk.)\n")
            continue

        messages.append({"role": "user", "content": line})
        try:
            reply = ollama_chat(args.host, args.model, messages)
        except urllib.error.URLError as e:
            print(f"Cannot reach Ollama at http://{args.host}/ — is it running?", file=sys.stderr)
            print(str(e.reason) if e.reason else str(e), file=sys.stderr)
            return 1
        print(f"Assistant: {reply}\n")
        messages.append({"role": "assistant", "content": reply})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
