"""
Real-ESRGAN RRDBNet architecture and inference engine.

Manual implementation of the RRDBNet (Residual-in-Residual Dense Block Network)
used by Real-ESRGAN. This avoids the basicsr dependency which has compilation
issues on Apple Silicon.

Architecture:
  Input → conv_first → 23x RRDB blocks → trunk_conv → upsample layers → output
  Each RRDB = 3 cascaded ResidualDenseBlocks
  Each RDB = 5 conv layers with dense (skip) connections
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import urllib.request
import numpy as np
import cv2

MODEL_URLS = {
    2: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    4: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
}

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


class ResidualDenseBlock(nn.Module):
    """5-layer densely connected conv block with residual scaling."""

    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Residual scaling by 0.2
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block: 3 cascaded RDBs with residual scaling."""

    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """
    RRDBNet: Real-ESRGAN generator network.

    For x4plus (scale=4):
      - Input passes through directly
      - Two 2x nearest-neighbor upsamples = 4x total

    For x2plus (scale=2):
      - Input is pixel_unshuffled by 2 first (halves spatial dims, 4x channels)
      - Two 2x nearest-neighbor upsamples = 4x spatial
      - Net effect: 2x upscale
    """

    def __init__(self, scale=4, num_in_ch=3, num_out_ch=3, num_feat=64,
                 num_block=23, num_grow_ch=32):
        super().__init__()
        self.scale = scale

        if scale == 2:
            num_in_ch = num_in_ch * 4  # pixel_unshuffle increases channels

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Upsample layers (2x each, applied twice = 4x)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            x = F.pixel_unshuffle(x, 2)

        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        # Two 2x upsamples via nearest-neighbor interpolation + conv
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


def download_model(scale):
    """Download model weights from GitHub releases if not already cached."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    url = MODEL_URLS[scale]
    filename = os.path.basename(url)
    filepath = os.path.join(MODEL_DIR, filename)

    if os.path.exists(filepath):
        return filepath

    print(f"Downloading {filename} (~67MB)...")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            print(f"\r  {pct}% ({downloaded // 1024 // 1024}MB / {total_size // 1024 // 1024}MB)", end="", flush=True)

    urllib.request.urlretrieve(url, filepath, reporthook=progress_hook)
    print("\n  Download complete.")
    return filepath


def load_model(scale, device):
    """Load RRDBNet with pretrained weights."""
    filepath = download_model(scale)
    model = RRDBNet(scale=scale)

    state_dict = torch.load(filepath, map_location="cpu", weights_only=True)
    if "params_ema" in state_dict:
        state_dict = state_dict["params_ema"]
    elif "params" in state_dict:
        state_dict = state_dict["params"]

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)
    return model


def get_device():
    """Select best available device: MPS (Apple Silicon) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _upscale_single_pass(img, scale, device, progress_callback=None, pass_label=""):
    """Run a single Real-ESRGAN pass (2x or 4x) on a BGR numpy array. Returns upscaled BGR array."""
    # Wrap callback to prepend pass label for chained upscales
    if progress_callback and pass_label:
        raw_cb = progress_callback
        def progress_callback(stage, current, total):
            if stage == "processing":
                raw_cb(f"{pass_label} processing", current, total)
            elif stage == "loading_model":
                raw_cb("loading_model", 0, 0)
            else:
                raw_cb(stage, current, total)

    if progress_callback:
        progress_callback("loading_model", 0, 0)

    model = load_model(scale, device)

    tile_sizes = [400, 256, 128]
    result = None

    for tile_size in tile_sizes:
        try:
            result = _process_tiles(img, model, device, scale, tile_size, progress_callback)
            break
        except (torch.mps.OutOfMemoryError if hasattr(torch, 'mps') and hasattr(torch.mps, 'OutOfMemoryError') else RuntimeError) as e:
            if "out of memory" in str(e).lower() or "MPS" in str(e):
                print(f"OOM with tile_size={tile_size}, trying smaller...")
                if hasattr(torch, 'mps'):
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if tile_size == tile_sizes[-1]:
                    print("Falling back to CPU...")
                    device = torch.device("cpu")
                    model = model.to(device)
                    result = _process_tiles(img, model, device, scale, 256, progress_callback)
            else:
                raise

    if result is None:
        raise RuntimeError("Failed to upscale image")

    return result


# Chain definitions: maps a requested scale to a sequence of native model passes
CHAIN_PASSES = {
    2: [2],
    4: [4],
    8: [4, 2],    # 4x then 2x = 8x
    16: [4, 4],   # 4x then 4x = 16x
}


def upscale_image(img_path, output_path, scale, progress_callback=None):
    """
    Upscale an image using Real-ESRGAN with tile-based processing.

    Args:
        img_path: Path to input image
        output_path: Path to save upscaled image
        scale: Upscale factor (2, 4, 8, or 16)
        progress_callback: Optional callable(stage, current, total)
            stage: "loading_model", "processing", "saving", "complete", "error"
    """
    device = get_device()
    passes = CHAIN_PASSES.get(scale)
    if passes is None:
        raise ValueError(f"Unsupported scale: {scale}. Use 2, 4, 8, or 16.")

    # Read image
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    # Handle alpha channel
    has_alpha = img.shape[2] == 4 if len(img.shape) == 3 else False
    if has_alpha:
        alpha = img[:, :, 3]
        img = img[:, :, :3]

    # Run chained passes
    result = img
    for i, pass_scale in enumerate(passes):
        pass_label = f"Pass {i+1}/{len(passes)} ({pass_scale}x)" if len(passes) > 1 else ""
        result = _upscale_single_pass(result, pass_scale, device, progress_callback, pass_label)

    # Handle alpha channel upscale
    if has_alpha:
        alpha_upscaled = cv2.resize(alpha, (result.shape[1], result.shape[0]),
                                     interpolation=cv2.INTER_LANCZOS4)
        result = np.concatenate([result, alpha_upscaled[:, :, np.newaxis]], axis=2)

    if progress_callback:
        progress_callback("saving", 0, 0)

    cv2.imwrite(output_path, result)

    if progress_callback:
        progress_callback("complete", 0, 0)

    return result.shape[1], result.shape[0]  # width, height


def _process_tiles(img, model, device, scale, tile_size, progress_callback):
    """Process image in tiles to manage memory usage."""
    tile_pad = 10
    h, w = img.shape[:2]

    # Calculate number of tiles
    tiles_x = max(1, (w + tile_size - 1) // tile_size)
    tiles_y = max(1, (h + tile_size - 1) // tile_size)
    total_tiles = tiles_x * tiles_y

    # Output image
    output_h, output_w = h * scale, w * scale
    output = np.zeros((output_h, output_w, 3), dtype=np.uint8)

    current_tile = 0

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            # Calculate tile boundaries with padding
            x_start = tx * tile_size
            y_start = ty * tile_size
            x_end = min(x_start + tile_size, w)
            y_end = min(y_start + tile_size, h)

            # Add padding (clamped to image bounds)
            x_start_pad = max(0, x_start - tile_pad)
            y_start_pad = max(0, y_start - tile_pad)
            x_end_pad = min(w, x_end + tile_pad)
            y_end_pad = min(h, y_end + tile_pad)

            # Extract tile
            tile = img[y_start_pad:y_end_pad, x_start_pad:x_end_pad, :]

            # Convert to tensor: BGR→RGB, HWC→CHW, normalize to [0,1]
            tile_tensor = torch.from_numpy(tile[:, :, ::-1].copy()).float() / 255.0
            tile_tensor = tile_tensor.permute(2, 0, 1).unsqueeze(0)
            tile_tensor = tile_tensor.to(device)

            # Inference
            with torch.no_grad():
                output_tile = model(tile_tensor)

            # Convert back: CHW→HWC, RGB→BGR, denormalize
            output_tile = output_tile.squeeze(0).permute(1, 2, 0).cpu().numpy()
            output_tile = (output_tile[:, :, ::-1] * 255.0).clip(0, 255).astype(np.uint8)

            # Calculate output position (remove padding from output)
            out_x_start = x_start * scale
            out_y_start = y_start * scale
            out_x_end = x_end * scale
            out_y_end = y_end * scale

            # Padding offsets in output space
            pad_left = (x_start - x_start_pad) * scale
            pad_top = (y_start - y_start_pad) * scale
            pad_right = pad_left + (x_end - x_start) * scale
            pad_bottom = pad_top + (y_end - y_start) * scale

            # Place tile (without padding) into output
            output[out_y_start:out_y_end, out_x_start:out_x_end, :] = \
                output_tile[pad_top:pad_bottom, pad_left:pad_right, :]

            current_tile += 1
            if progress_callback:
                progress_callback("processing", current_tile, total_tiles)

    return output
