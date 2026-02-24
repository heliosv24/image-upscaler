# Image Upscaler

A free, local AI image upscaler powered by [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN). The same model that paid online upscaling services use — running entirely on your machine.

Upload an image, pick **2x** or **4x**, watch the progress bar, download the result.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- **Real-ESRGAN AI upscaling** — state-of-the-art super-resolution, not simple interpolation
- **2x and 4x upscale** with output dimension preview before processing
- **Apple Silicon GPU acceleration** via Metal Performance Shaders (MPS), with automatic CPU fallback
- **Real-time progress** streamed to the browser via Server-Sent Events
- **Before/after comparison slider** to inspect the result
- **Drag-and-drop upload** — supports JPG, PNG, WebP, BMP, TIFF (up to 50MB / 20MP)
- **One-command setup** — creates a virtual environment, installs dependencies, downloads model weights on first use
- **Fully local** — nothing leaves your machine

## Quick Start

```bash
git clone https://github.com/heliosv24/image-upscaler.git
cd image-upscaler
./start.sh
```

Then open **http://localhost:8080** in your browser.

That's it. On first run, `start.sh` will:
1. Create a Python virtual environment
2. Install dependencies (~130MB: PyTorch, OpenCV, Flask, Pillow)
3. Launch the server

Model weights (~67MB per scale) are downloaded automatically the first time you upscale.

## Requirements

- **Python 3.9+**
- **macOS** with Apple Silicon (M1/M2/M3/M4) for GPU acceleration, or any system with a CUDA GPU, or CPU-only as fallback
- ~500MB disk space (dependencies + model weights)

## Usage

1. **Upload** — drag and drop an image onto the upload zone, or click to browse
2. **Choose scale** — select 2x or 4x (output dimensions shown below each option)
3. **Upscale** — click the button and watch the progress bar as tiles are processed
4. **Compare** — drag the before/after slider to inspect the result
5. **Download** — click Download to save the upscaled image

## How It Works

The app uses a manual PyTorch implementation of the **RRDBNet** architecture (Residual-in-Residual Dense Block Network) — the same generator network used by Real-ESRGAN. The official pretrained weights are loaded directly, producing identical results to the original model.

**Why not `pip install realesrgan`?** The official package depends on `basicsr`, which has C++ compilation issues on Apple Silicon with newer Python versions. The manual implementation (~200 lines of PyTorch) avoids this entirely while loading the exact same `.pth` weight files.

### Tile-Based Processing

Large images are split into 400×400 tiles with 10px overlap to keep GPU memory usage manageable. If a tile causes an out-of-memory error, the app automatically retries with smaller tiles (256, then 128) before falling back to CPU.

### Performance

| Image Size | Scale | Tiles | Time (MPS) |
|-----------|-------|-------|------------|
| 1920×1080 | 4x | ~15 | ~10-30s |
| 3840×2160 | 4x | ~60 | ~1-2min |
| 1920×1080 | 2x | ~15 | ~10-20s |

Times vary based on your specific hardware. First run includes a one-time ~67MB model download.

## Project Structure

```
image-upscaler/
├── model.py              # RRDBNet architecture + tile inference engine
├── app.py                # Flask backend (upload, SSE progress, download)
├── templates/
│   └── index.html        # Frontend (dark theme, drag-drop, comparison slider)
├── models/               # Auto-downloaded .pth weights (~67MB each)
├── uploads/              # Temporary upload directory (auto-cleaned)
├── outputs/              # Upscaled results (auto-cleaned)
├── requirements.txt
└── start.sh              # One-command setup + launch
```

## Troubleshooting

**Port already in use:** On macOS, AirPlay Receiver uses port 5000 by default. This app uses port 8080 to avoid the conflict. If 8080 is also taken, edit the port number in `app.py` and `start.sh`.

**Slow on first tile:** The first tile takes longer because PyTorch is compiling MPS kernels. Subsequent tiles are faster.

**Out of memory:** The app handles this automatically by reducing tile size. For very large images (8K+), processing may fall back to CPU which is slower but always works.

## License

MIT
