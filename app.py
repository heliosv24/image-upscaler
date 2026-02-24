"""
Flask backend for Image Upscaler.

Routes:
  GET  /                    — Serve the frontend
  POST /upload              — Upload an image, get job_id + dimensions
  GET  /upscale/<job_id>    — SSE stream: progress updates during upscaling
  GET  /download/<filename> — Download upscaled image
  GET  /outputs/<filename>  — Serve output image for preview
  GET  /uploads/<filename>  — Serve uploaded image for preview
"""

import os
import time
import uuid
import threading
from flask import Flask, request, jsonify, send_from_directory, Response, render_template
from PIL import Image
from model import upscale_image

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

# Store job info
jobs = {}


@app.route("/")
def index():
    return render_template("index.html")


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

    jobs[job_id] = {
        "filename": filename,
        "filepath": filepath,
        "width": width,
        "height": height,
    }

    return jsonify({
        "job_id": job_id,
        "filename": filename,
        "width": width,
        "height": height,
    })


@app.route("/upscale/<job_id>")
def upscale(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404

    scale = request.args.get("scale", "4", type=str)
    if scale not in ("2", "4"):
        return jsonify({"error": "Scale must be 2 or 4"}), 400
    scale = int(scale)

    job = jobs[job_id]
    input_path = job["filepath"]
    ext = os.path.splitext(job["filename"])[1]
    output_filename = f"{job_id}_{scale}x{ext}"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    def generate():
        # Acquire lock — only one upscale at a time
        yield f"data: {_sse_json('queued', 'Waiting for other jobs to finish...')}\n\n"

        with upscale_lock:
            try:
                last_stage = [None]

                def on_progress(stage, current, total):
                    last_stage[0] = stage

                    if stage == "loading_model":
                        msg = "Loading AI model (downloading if first use)..."
                    elif stage == "processing":
                        pct = int(current / total * 100) if total > 0 else 0
                        msg = f"Processing tile {current}/{total} ({pct}%)"
                    elif stage == "saving":
                        msg = "Saving output..."
                    elif stage == "complete":
                        msg = "Done!"
                    else:
                        msg = stage

                    # We can't yield from a callback, so we store progress
                    # and the SSE loop picks it up
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

                # Run upscale in a thread so we can stream progress
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

                # Stream progress via SSE
                while thread.is_alive():
                    if progress_state["updated"]:
                        progress_state["updated"] = False
                        yield f"data: {_sse_json(progress_state['stage'], progress_state['message'], progress_state['current'], progress_state['total'])}\n\n"
                    time.sleep(0.2)

                thread.join()

                # Send final state
                if result["error"]:
                    yield f"data: {_sse_json('error', result['error'])}\n\n"
                else:
                    import json
                    yield f"data: {json.dumps({'stage': 'complete', 'message': 'Upscaling complete!', 'output_filename': output_filename, 'output_width': result['width'], 'output_height': result['height']})}\n\n"

            except Exception as e:
                yield f"data: {_sse_json('error', str(e))}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _sse_json(stage, message, current=0, total=0):
    import json
    return json.dumps({
        "stage": stage,
        "message": message,
        "current": current,
        "total": total,
    })


@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


@app.route("/outputs/<filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/uploads/<filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


def cleanup_old_files():
    """Remove files older than 1 hour from uploads and outputs."""
    cutoff = time.time() - 3600
    for directory in [UPLOAD_DIR, OUTPUT_DIR]:
        for f in os.listdir(directory):
            filepath = os.path.join(directory, f)
            if os.path.isfile(filepath) and os.path.getmtime(filepath) < cutoff:
                try:
                    os.remove(filepath)
                except OSError:
                    pass


if __name__ == "__main__":
    # Cleanup old files on startup
    cleanup_old_files()
    print("Starting Image Upscaler on http://localhost:8080")
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)
