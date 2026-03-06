# Image Upscaler

I needed to upscale an image. I went online and every single site either wanted $10/month, plastered watermarks over the output, or wanted me to upload my files to some random server.

So I just built it myself.

Not in a weekend. Not after watching a tutorial series. I opened [Claude Code](https://claude.ai/code), described what I wanted, and had a working app in about an hour. The AI model powering those paid services — [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) — is open source. The only thing those sites were selling me was a wrapper around free technology, plus the privilege of sending my images to their servers.

That's the world we live in now. The gap between "I wish this existed" and "I built it" has collapsed. If you can describe what you want, you can build it. No accounts, no watermarks, no bullshit — just your images, on your machine, upscaled by the same neural network the paid tools use.

## Quick Start

```bash
git clone https://github.com/heliosv24/image-upscaler.git
cd image-upscaler
./start.sh
```

Open [http://localhost:8080](http://localhost:8080). That's it.

First run creates a venv, installs deps, and starts the server. Model weights (~67MB per scale) download automatically on first upscale.

## Features

- **AI upscaling** (Real-ESRGAN) — actual super-resolution, not bicubic stretching
- **2x, 4x, 8x, 16x** — higher scales chain multiple passes automatically
- **GPU accelerated** — Apple Silicon (MPS), NVIDIA (CUDA), or CPU fallback
- **Real-time progress** — watch tiles process live via SSE streaming
- **Before/after slider** — compare results instantly
- **Multi-format export** — PNG, JPEG, WebP, PDF
- **Project management** — organize upscales into projects, rename, delete
- **Fully local** — nothing touches the internet except downloading model weights once

## How It Works

1. **Upload** — drag and drop or browse. JPG, PNG, WebP, BMP, TIFF. Up to 50MB / 20MP.
2. **Pick your scale** — 2x, 4x, 8x, or 16x. Output dimensions shown before you commit.
3. **Watch it process** — real-time tile-by-tile progress via Server-Sent Events.
4. **Compare** — before/after slider to inspect the result.
5. **Download** — pick your format (PNG, JPEG, WebP, PDF) and custom filename.

## Architecture

```
landing.html (/)  -->  app.py  -->  index.html (/app)  -->  model.py (PyTorch)
                                         |
                                    db.py (SQLite)
```

- **Flask backend** with vanilla JS frontend and SQLite for project persistence
- **Manual RRDBNet implementation** — no basicsr dependency, which won't compile on Apple Silicon
- **Tile-based inference** with automatic OOM recovery (400px -> 256px -> 128px -> CPU fallback)
- **SSE streaming** for real-time progress updates during upscaling

## Requirements

- Python 3.9+
- ~500MB disk space (deps + model weights)
- macOS / Linux / Windows (Apple Silicon GPU preferred, CUDA works, CPU fallback always available)

## Troubleshooting

**Port already in use:** The app runs on port 8080. If that's taken, edit the port in `app.py` and `start.sh`.

**Slow first tile:** PyTorch compiles MPS kernels on the first tile. Subsequent tiles are faster.

**Out of memory:** Handled automatically — tile size reduces progressively, then falls back to CPU.

## Built With

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) — AI super-resolution model
- [PyTorch](https://pytorch.org/) — ML inference
- [Flask](https://flask.palletsprojects.com/) — web backend
- [Claude Code](https://claude.ai/code) — built the entire thing

## License

MIT
