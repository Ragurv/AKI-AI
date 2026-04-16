# Changelog

All notable changes in this project are listed here.

## 2026-04-09

### Added

- ASK / AGENT mode selector in web composer.
- Mode-aware backend behavior via additional system guidance per request mode.
- PLANNING mode selector with planning-first behavior (phases, trade-offs, risks, validation checklist).
- Assistant reply mode badge (mode + model) in chat UI.
- LLM model selector in composer with refresh action.
- Backend models endpoint (`/api/models`) to list installed Ollama models.
- Model filter box in settings for quick model searching.
- Model dropdown grouping by family (e.g., LLAMA, MISTRAL, QWEN).
- `README.md` project documentation.
- `CHANGELOG.md` maintenance log.
- Sidebar brand title: `AKI-AI`.
- Theme option: `System (Auto)` with OS theme listener.

### Changed

- Moved app actions to left sidebar lower section for a Claude-like layout.
- Converted sidebar action controls to stacked one-by-one buttons.
- Improved light-theme readability using shared theme variables for controls and borders.
- Added mode reference text in settings for ASK / PLANNING / AGENT.
- Requests now carry selected model and mode so backend uses the chosen LLM per message.
- Strengthened spellcheck handling with explicit input attributes and persisted settings application.

### Fixed

- Theme switching readability regressions in light mode.
- Spellcheck toggle not consistently applying in the input box.
- Send flow after model change: if selected model is unavailable, app now auto-retries with default model.

### Existing Capabilities Retained

- Login/logout and auth cookie flow.
- Session list/search/load/rename/delete.
- Stream and non-stream chat response flow.
- Request queue handling while model is busy.
- Export to TXT and JSON.
