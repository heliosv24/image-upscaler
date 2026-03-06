"""
Flask backend for Image Upscaler.

Routes:
  GET  /                       — Serve the frontend
  POST /upload                 — Upload an image, get job_id + dimensions
  GET  /upscale/<job_id>       — SSE stream: progress updates during upscaling
  GET  /download/<filename>    — Download upscaled image (with format conversion)
  GET  /outputs/<filename>     — Serve output image for preview
  GET  /uploads/<filename>     — Serve uploaded image for preview

  GET    /api/projects         — List all projects
  POST   /api/projects         — Create a project
  GET    /api/projects/<id>    — Get project + jobs
  PATCH  /api/projects/<id>    — Rename project
  DELETE /api/projects/<id>    — Delete project + cleanup files
"""

import io
import json
import os
import re
import time
import uuid
import threading
from flask import Flask, request, jsonify, send_from_directory, send_file, Response, render_template
from PIL import Image
from model import upscale_image
from db import init_db, create_project, get_project, list_projects, rename_project, \
    delete_project, create_job, update_job_complete, update_job_error, get_job, \
    list_jobs, get_all_job_filenames

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_PIXELS = 20_000_000  # 20 megapixels
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

# Serialize upscale jobs so we don't OOM with concurrent requests
upscale_lock = threading.Lock()

# In-memory job state for SSE progress (DB stores persistent record)
jobs = {}


# ── Page ──────────────────────────────────────────────────────

@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/app")
def app_page():
    return render_template("index.html")


# ── Projects API ──────────────────────────────────────────────

@app.route("/api/projects", methods=["GET"])
def api_list_projects():
    return jsonify(list_projects())


@app.route("/api/projects", methods=["POST"])
def api_create_project():
    data = request.get_json(silent=True) or {}
    name = data.get("name", "Untitled Project")
    project = create_project(name)
    return jsonify(project), 201


@app.route("/api/projects/<project_id>", methods=["GET"])
def api_get_project(project_id):
    project = get_project(project_id)
    if not project:
        return jsonify({"error": "Project not found"}), 404
    project["jobs"] = list_jobs(project_id)
    return jsonify(project)


@app.route("/api/projects/<project_id>", methods=["PATCH"])
def api_rename_project(project_id):
    data = request.get_json(silent=True) or {}
    name = data.get("name")
    if not name:
        return jsonify({"error": "Name required"}), 400
    project = rename_project(project_id, name)
    if not project:
        return jsonify({"error": "Project not found"}), 404
    return jsonify(project)


@app.route("/api/projects/<project_id>", methods=["DELETE"])
def api_delete_project(project_id):
    project = get_project(project_id)
    if not project:
        return jsonify({"error": "Project not found"}), 404
    filenames = delete_project(project_id)
    # Clean up files on disk
    for kind, fname in filenames:
        directory = UPLOAD_DIR if kind == "upload" else OUTPUT_DIR
        path = os.path.join(directory, fname)
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
    return jsonify({"ok": True})


# ── Upload ────────────────────────────────────────────────────

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported format. Use: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    # Check file size
    file.seek(0, 2)
    size = file.tell()
    file.seek(0)
    if size > MAX_FILE_SIZE:
        return jsonify({"error": f"File too large. Max {MAX_FILE_SIZE // 1024 // 1024}MB"}), 400

    # Save file
    job_id = str(uuid.uuid4())[:8]
    filename = f"{job_id}{ext}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)

    # Validate image and get dimensions
    try:
        with Image.open(filepath) as img:
            width, height = img.size
            pixels = width * height
            if pixels > MAX_PIXELS:
                os.remove(filepath)
                return jsonify({"error": f"Image too large ({pixels:,} pixels). Max {MAX_PIXELS:,} pixels."}), 400
    except Exception:
        os.remove(filepath)
        return jsonify({"error": "Could not read image file"}), 400

    # Store in-memory for SSE progress tracking
    jobs[job_id] = {
        "filename": filename,
        "filepath": filepath,
        "width": width,
        "height": height,
        "original_name": file.filename,
        "project_id": request.form.get("project_id"),
    }

    return jsonify({
        "job_id": job_id,
        "filename": filename,
        "width": width,
        "height": height,
        "original_name": file.filename,
    })


# ── Upscale (SSE) ────────────────────────────────────────────

@app.route("/upscale/<job_id>")
def upscale(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404

    scale = request.args.get("scale", "4", type=str)
    if scale not in ("2", "4", "8", "16"):
        return jsonify({"error": "Scale must be 2, 4, 8, or 16"}), 400
    scale = int(scale)

    job = jobs[job_id]
    input_path = job["filepath"]
    ext = os.path.splitext(job["filename"])[1]
    output_filename = f"{job_id}_{scale}x{ext}"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    project_id = job.get("project_id")
    original_name = job.get("original_name", "")

    # Create DB job record if we have a project
    db_job_id = None
    if project_id:
        db_job = create_job(
            project_id=project_id,
            original_name=original_name,
            upload_filename=job["filename"],
            width=job["width"],
            height=job["height"],
            scale=scale,
        )
        db_job_id = db_job["id"]

    def generate():
        # Acquire lock — only one upscale at a time
        yield f"data: {_sse_json('queued', 'Waiting for other jobs to finish...')}\n\n"

        with upscale_lock:
            try:
                def on_progress(stage, current, total):
                    if stage == "loading_model":
                        msg = "Loading AI model (downloading if first use)..."
                    elif stage == "processing":
                        pct = int(current / total * 100) if total > 0 else 0
                        msg = f"Processing tile {current}/{total} ({pct}%)"
                    elif stage.endswith(" processing"):
                        # Chained upscale pass, e.g. "Pass 1/2 (4x) processing"
                        pass_info = stage.replace(" processing", "")
                        pct = int(current / total * 100) if total > 0 else 0
                        msg = f"{pass_info} — tile {current}/{total} ({pct}%)"
                        stage = "processing"
                    elif stage == "saving":
                        msg = "Saving output..."
                    elif stage == "complete":
                        msg = "Done!"
                    else:
                        msg = stage

                    progress_state["stage"] = stage
                    progress_state["current"] = current
                    progress_state["total"] = total
                    progress_state["message"] = msg
                    progress_state["updated"] = True

                progress_state = {
                    "stage": "starting",
                    "current": 0,
                    "total": 0,
                    "message": "Starting...",
                    "updated": True,
                }

                result = {"error": None, "width": 0, "height": 0}

                def run_upscale():
                    try:
                        w, h = upscale_image(input_path, output_path, scale, on_progress)
                        result["width"] = w
                        result["height"] = h
                    except Exception as e:
                        result["error"] = str(e)
                        progress_state["stage"] = "error"
                        progress_state["message"] = str(e)
                        progress_state["updated"] = True

                thread = threading.Thread(target=run_upscale)
                thread.start()

                while thread.is_alive():
                    if progress_state["updated"]:
                        progress_state["updated"] = False
                        yield f"data: {_sse_json(progress_state['stage'], progress_state['message'], progress_state['current'], progress_state['total'])}\n\n"
                    time.sleep(0.2)

                thread.join()

                if result["error"]:
                    if db_job_id:
                        update_job_error(db_job_id)
                    yield f"data: {_sse_json('error', result['error'])}\n\n"
                else:
                    if db_job_id:
                        update_job_complete(db_job_id, output_filename, result["width"], result["height"])
                    yield f"data: {json.dumps({'stage': 'complete', 'message': 'Upscaling complete!', 'output_filename': output_filename, 'output_width': result['width'], 'output_height': result['height'], 'db_job_id': db_job_id})}\n\n"

            except Exception as e:
                if db_job_id:
                    update_job_error(db_job_id)
                yield f"data: {_sse_json('error', str(e))}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _sse_json(stage, message, current=0, total=0):
    return json.dumps({
        "stage": stage,
        "message": message,
        "current": current,
        "total": total,
    })


# ── Download (with format conversion) ────────────────────────

def _sanitize_filename(name):
    """Remove unsafe chars from a filename, keep alphanumeric, dash, underscore, dot."""
    name = re.sub(r'[^\w\-.]', '_', name)
    name = re.sub(r'_+', '_', name)
    return name.strip('_') or 'upscaled'


@app.route("/download/<filename>")
def download(filename):
    fmt = request.args.get("format", "").lower()
    custom_name = request.args.get("filename", "").strip()

    # No format param → serve original file (backward compat)
    if not fmt:
        return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)

    valid_formats = {"png", "jpeg", "webp", "pdf"}
    if fmt not in valid_formats:
        return jsonify({"error": f"Invalid format. Use: {', '.join(valid_formats)}"}), 400

    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.isfile(filepath):
        return jsonify({"error": "File not found"}), 404

    img = Image.open(filepath)

    # JPEG doesn't support alpha — convert to RGB
    if fmt == "jpeg" and img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")

    buf = io.BytesIO()
    if fmt == "png":
        img.save(buf, "PNG")
        mimetype = "image/png"
        ext = ".png"
    elif fmt == "jpeg":
        img.save(buf, "JPEG", quality=95)
        mimetype = "image/jpeg"
        ext = ".jpg"
    elif fmt == "webp":
        img.save(buf, "WEBP", quality=95)
        mimetype = "image/webp"
        ext = ".webp"
    elif fmt == "pdf":
        # Pillow can save images as PDF natively
        if img.mode == "RGBA":
            img = img.convert("RGB")
        img.save(buf, "PDF")
        mimetype = "application/pdf"
        ext = ".pdf"

    buf.seek(0)

    # Build download filename
    if custom_name:
        dl_name = _sanitize_filename(os.path.splitext(custom_name)[0]) + ext
    else:
        dl_name = os.path.splitext(filename)[0] + ext

    return send_file(buf, mimetype=mimetype, as_attachment=True, download_name=dl_name)


# ── Static file serving ──────────────────────────────────────

@app.route("/outputs/<filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/uploads/<filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


# ── Cleanup ───────────────────────────────────────────────────

def cleanup_old_files():
    """Remove orphan files older than 1 hour. Skip files referenced by DB jobs."""
    referenced = get_all_job_filenames()
    cutoff = time.time() - 3600
    for directory in [UPLOAD_DIR, OUTPUT_DIR]:
        for f in os.listdir(directory):
            if f in referenced:
                continue
            filepath = os.path.join(directory, f)
            if os.path.isfile(filepath) and os.path.getmtime(filepath) < cutoff:
                try:
                    os.remove(filepath)
                except OSError:
                    pass


if __name__ == "__main__":
    init_db()
    cleanup_old_files()
    print("Starting Image Upscaler on http://localhost:8080")
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)
