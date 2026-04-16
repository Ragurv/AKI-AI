"""
Local CLI chat: Ollama + Dolphin-Mistral.

Start Ollama, then:
  ollama pull dolphin-mistral
  python cli_chat.py
  python cli_chat.py --pull

No GPU required: Ollama uses your CPU automatically.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

DEFAULT_MODEL = "dolphin-mistral"
DEFAULT_HOST = "127.0.0.1:11434"


def ollama_chat(host: str, model: str, messages: list[dict]) -> str:
    url = f"http://{host}/api/chat"
    body = json.dumps({"model": model, "messages": messages, "stream": False}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.load(resp)
    msg = data.get("message") or {}
    return (msg.get("content") or "").strip()


def cmd_pull(model: str) -> int:
    exe = shutil.which("ollama")
    if not exe:
        print("Ollama is not installed or not on PATH.", file=sys.stderr)
        print("Install from https://ollama.com/download then re-run.", file=sys.stderr)
        return 1
    print(f"Running: ollama pull {model}", flush=True)
    return subprocess.call([exe, "pull", model])


def main() -> int:
    parser = argparse.ArgumentParser(description="Local CLI chat via Ollama")
    parser.add_argument("--pull", action="store_true", help="Download the model (ollama pull)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Ollama host:port (default: {DEFAULT_HOST})")
    args = parser.parse_args()

    if args.pull:
        return cmd_pull(args.model)

    messages: list[dict] = []
    print(f"{args.model} — type /q or /quit to exit.\n")

    while True:
        try:
            line = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in {"/q", "/quit", "/exit"}:
            break

        messages.append({"role": "user", "content": line})
        try:
            reply = ollama_chat(args.host, args.model, messages)
        except urllib.error.URLError as e:
            print(f"Cannot reach Ollama at http://{args.host}/ — is it running?", file=sys.stderr)
            print(str(e.reason) if e.reason else str(e), file=sys.stderr)
            print("Install/start: https://ollama.com/download", file=sys.stderr)
            return 1
        print(f"Assistant: {reply}\n")
        messages.append({"role": "assistant", "content": reply})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

