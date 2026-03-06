# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Local AI image upscaler using Real-ESRGAN (2x/4x). Flask backend with vanilla JS frontend, SQLite persistence for projects/jobs, and SSE-based progress streaming. The RRDBNet model is implemented manually (no basicsr dependency) to avoid compilation issues on Apple Silicon.

## Running the App

```bash
./start.sh          # Creates venv, installs deps, starts server on :8080
```

First upscale auto-downloads model weights (~67MB per scale) from GitHub releases. Server runs at `http://localhost:8080`.

## Architecture

```
index.html (frontend)  →  app.py (Flask routes + SSE)  →  model.py (PyTorch inference)
                                    ↕
                              db.py (SQLite)
```

- **app.py** — Routes: page serving, file upload, SSE upscale streaming, multi-format download, project CRUD API (`/api/projects/*`). Uses an in-memory `jobs` dict for ephemeral SSE state and SQLite for persistent records.
- **db.py** — SQLite with WAL mode, `row_factory=sqlite3.Row`, FK cascade deletes. Two tables: `projects` and `jobs`. DB at `data/upscaler.db`, auto-created on startup via `init_db()`.
- **model.py** — Manual RRDBNet (23 RRDB blocks). Tile-based inference (400→256→128px fallback on OOM, then CPU). Device priority: MPS > CUDA > CPU. Do not modify unless changing the model architecture.
- **templates/index.html** — Single-file frontend (~1900 lines). Inline CSS + vanilla JS. State object + render functions. Event delegation on sidebar for surviving re-renders. Comparison slider uses `clipPath: inset()`. Download panel builds URLs client-side (no fetch needed for format/name changes).

## Key Patterns

**SSE streaming** (`/upscale/<job_id>`): Flask generator yields JSON events. Model runs in a background thread with a shared `progress_state` dict polled every 200ms. `upscale_lock` serializes GPU access (one upscale at a time).

**File cleanup**: `cleanup_old_files()` removes orphan files >1 hour old but skips any filename referenced in the jobs table (`get_all_job_filenames()`).

**Format conversion** (`/download/<filename>?format=png|jpeg|webp|pdf`): Pillow loads the output, converts to the requested format in a BytesIO buffer, served via `send_file()`. JPEG/PDF require RGB conversion (strip alpha).

## File Layout

- `uploads/` — Temporary uploaded images (auto-cleaned)
- `outputs/` — Upscaled results (auto-cleaned)
- `models/` — Downloaded `.pth` weights (persisted)
- `data/` — SQLite database (gitignored)
