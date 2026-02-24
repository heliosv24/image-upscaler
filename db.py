"""
SQLite persistence layer for Image Upscaler.

Schema:
  projects — groups of upscale jobs (like ChatGPT conversations)
  jobs     — individual upscale operations within a project

DB stored at data/upscaler.db with WAL mode for concurrent reads.
"""

import os
import sqlite3
import uuid
from datetime import datetime, timezone

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_PATH = os.path.join(DB_DIR, "upscaler.db")


def get_db():
    """Get a database connection with Row factory and pragmas set."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create tables if they don't exist. Call at app startup."""
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            original_name TEXT,
            upload_filename TEXT,
            output_filename TEXT,
            width INTEGER,
            height INTEGER,
            output_width INTEGER,
            output_height INTEGER,
            scale INTEGER,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL,
            completed_at TEXT,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_jobs_project_id ON jobs(project_id);
    """)
    conn.commit()
    conn.close()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _new_id():
    return str(uuid.uuid4())[:8]


# ── Project CRUD ──────────────────────────────────────────────

def create_project(name="Untitled Project"):
    conn = get_db()
    pid = _new_id()
    now = _now()
    conn.execute(
        "INSERT INTO projects (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (pid, name, now, now),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM projects WHERE id = ?", (pid,)).fetchone()
    conn.close()
    return dict(row)


def get_project(project_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_projects():
    conn = get_db()
    rows = conn.execute("""
        SELECT p.*, COUNT(j.id) AS job_count
        FROM projects p
        LEFT JOIN jobs j ON j.project_id = p.id
        GROUP BY p.id
        ORDER BY p.updated_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def rename_project(project_id, name):
    conn = get_db()
    now = _now()
    conn.execute(
        "UPDATE projects SET name = ?, updated_at = ? WHERE id = ?",
        (name, now, project_id),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_project(project_id):
    """Delete project and all its jobs. Returns list of filenames to clean up."""
    conn = get_db()
    jobs = conn.execute(
        "SELECT upload_filename, output_filename FROM jobs WHERE project_id = ?",
        (project_id,),
    ).fetchall()
    filenames = []
    for job in jobs:
        if job["upload_filename"]:
            filenames.append(("upload", job["upload_filename"]))
        if job["output_filename"]:
            filenames.append(("output", job["output_filename"]))
    conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    conn.commit()
    conn.close()
    return filenames


# ── Job CRUD ──────────────────────────────────────────────────

def create_job(project_id, original_name, upload_filename, width, height, scale):
    conn = get_db()
    jid = _new_id()
    now = _now()
    conn.execute(
        """INSERT INTO jobs
           (id, project_id, original_name, upload_filename, width, height, scale, status, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)""",
        (jid, project_id, original_name, upload_filename, width, height, scale, now),
    )
    # Touch project updated_at
    conn.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (now, project_id))
    conn.commit()
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (jid,)).fetchone()
    conn.close()
    return dict(row)


def update_job_complete(job_id, output_filename, output_width, output_height):
    conn = get_db()
    now = _now()
    conn.execute(
        """UPDATE jobs
           SET output_filename = ?, output_width = ?, output_height = ?,
               status = 'complete', completed_at = ?
           WHERE id = ?""",
        (output_filename, output_width, output_height, now, job_id),
    )
    # Touch project updated_at
    row = conn.execute("SELECT project_id FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row:
        conn.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (now, row["project_id"]))
    conn.commit()
    conn.close()


def update_job_error(job_id):
    conn = get_db()
    now = _now()
    conn.execute(
        "UPDATE jobs SET status = 'error', completed_at = ? WHERE id = ?",
        (now, job_id),
    )
    conn.commit()
    conn.close()


def get_job(job_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_jobs(project_id):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM jobs WHERE project_id = ? ORDER BY created_at DESC",
        (project_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_job(job_id):
    """Delete a single job. Returns filenames to clean up."""
    conn = get_db()
    row = conn.execute("SELECT upload_filename, output_filename FROM jobs WHERE id = ?", (job_id,)).fetchone()
    filenames = []
    if row:
        if row["upload_filename"]:
            filenames.append(("upload", row["upload_filename"]))
        if row["output_filename"]:
            filenames.append(("output", row["output_filename"]))
        conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        conn.commit()
    conn.close()
    return filenames


def get_all_job_filenames():
    """Return set of all filenames referenced by jobs (for cleanup logic)."""
    conn = get_db()
    rows = conn.execute("SELECT upload_filename, output_filename FROM jobs").fetchall()
    conn.close()
    filenames = set()
    for row in rows:
        if row["upload_filename"]:
            filenames.add(row["upload_filename"])
        if row["output_filename"]:
            filenames.add(row["output_filename"])
    return filenames
