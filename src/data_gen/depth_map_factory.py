"""DepthMapFactory — pluggable (de)serialization of per-clip depth maps.

Encapsulates everything about how a clip's depth stack is quantized, written, and
read back, behind ONE small interface, so compression schemes can be A/B-tested
without touching the capture/collect pipeline. Motivated by probe-corpus-spec.md
§"depth storage": uint8 corrupts depth (256× coarser than uint16; the small
foreground bodies collapse to ~8 levels), so we keep **uint16 precision** and move
the size win into the *codec* instead of the *bit depth*.

Design:
  * Quantization is fixed and lossless-by-default: float metres → uint16 over the
    clip's own finite [d_min, d_max] (the canonical scheme; this module is now its
    single source of truth — clip_capture re-imports `quantize_depth_uint16`).
  * A **backend** decides only how the uint16 stack is stored on disk. Backends are
    registered in BACKENDS; adding one (e.g. a new codec) is a 2-function change.
  * Every encode writes a self-describing `depth_meta.json` sidecar (backend,
    d_min, d_max, shape, fps, codec opts, files), so `decode()` needs no prior
    knowledge of how the clip was written — the experiment is recorded with the data.

Backends shipped:
  * `npz`    — np.savez_compressed (zlib). Bit-exact. The legacy format/schema.
  * `ffv1`   — FFV1 lossless 16-bit video (.mkv, gray16le). Bit-exact, ~2.6× smaller
               than npz, and frame-by-frame scrubable. Recommended.
  * `hevc12` — libx265 over the top 12 bits (gray12le). LOSSY (drops the low 4 bits;
               optional x265 rate control). The "go smaller, measure the error"
               experiment knob. Default x265 lossless ⇒ error is only the 4-bit
               truncation (≤ 15/65535 of span).

This module is import-safe OUTSIDE Isaac (numpy + cv2 + ffmpeg subprocess only —
no omni/isaac), so depth can be re-encoded/inspected offline.

Read a clip back:
    from depth_map_factory import DepthMapFactory
    q = DepthMapFactory.decode(clip_dir)            # (T, H, W) uint16
    metres = DepthMapFactory.decode_meters(clip_dir)  # (T, H, W) float32
"""

import json
import os
import shutil
import subprocess

import numpy as np

SIDECAR = "depth_meta.json"


# --------------------------------------------------------------------------- #
# Canonical quantization (single source of truth; clip_capture re-imports this) #
# --------------------------------------------------------------------------- #
def quantize_depth_uint16(depth_stack):
    """Quantize a float depth stack to uint16 over its own finite [min, max].

    Returns (q_uint16, d_min, d_max). Non-finite samples (sky / no-hit) are pinned
    to d_max before quantizing. Invert with `dequantize_depth`.
    """
    finite = depth_stack[np.isfinite(depth_stack)]
    d_min = float(finite.min()) if finite.size else 0.0
    d_max = float(finite.max()) if finite.size else 1.0
    span = d_max - d_min if d_max > d_min else 1.0
    clipped = np.clip(
        np.nan_to_num(depth_stack, nan=d_max, posinf=d_max, neginf=d_min),
        d_min, d_max,
    )
    q = np.round((clipped - d_min) / span * 65535.0).astype(np.uint16)
    return q, d_min, d_max


def dequantize_depth(q, d_min, d_max):
    """uint16 code → float32 metres (inverse of quantize_depth_uint16)."""
    span = (d_max - d_min) if d_max > d_min else 1.0
    return (d_min + q.astype(np.float32) / 65535.0 * span).astype(np.float32)


# --------------------------------------------------------------------------- #
# Backends: each is (write, read) over the uint16 stack. Files are relative to  #
# the clip dir; the sidecar records which were written.                        #
# --------------------------------------------------------------------------- #
def _ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("ffmpeg not found on PATH; required by the video depth backends")
    return exe


def _npz_write(out_dir, q, fps, opts):
    # Legacy schema: depth + depth_min/depth_max kept inside the npz too, so an
    # old reader that ignores the sidecar still round-trips.
    np.savez_compressed(os.path.join(out_dir, "depth.npz"), depth=q,
                        depth_min=np.float32(opts["d_min"]),
                        depth_max=np.float32(opts["d_max"]))
    return ["depth.npz"]


def _npz_read(out_dir, meta):
    with np.load(os.path.join(out_dir, "depth.npz")) as d:
        return d["depth"]


def _video_write(out_dir, q, fps, fname, pix_fmt, codec, codec_args):
    """Pipe a uint16 (T,H,W) stack to ffmpeg as raw `pix_fmt`, encode to `fname`."""
    T, H, W = q.shape
    path = os.path.join(out_dir, fname)
    proc = subprocess.Popen(
        [_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo",
         "-pix_fmt", pix_fmt, "-s", f"{W}x{H}", "-r", str(float(fps)), "-i", "-",
         "-c:v", codec, *codec_args, "-pix_fmt", pix_fmt, path],
        stdin=subprocess.PIPE)
    proc.stdin.write(q.astype("<u2").tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg encode failed for {fname}")
    return [fname]


def _video_read(out_dir, meta, fname, pix_fmt):
    """Decode `fname` back to a uint16 (T,H,W) stack via raw `pix_fmt`."""
    T, H, W = meta["shape"]
    out = subprocess.run(
        [_ffmpeg(), "-loglevel", "error", "-i", os.path.join(out_dir, fname),
         "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"],
        capture_output=True)
    if out.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed for {fname}: {out.stderr.decode()[:200]}")
    return np.frombuffer(out.stdout, "<u2").reshape(T, H, W)


def _ffv1_write(out_dir, q, fps, opts):
    return _video_write(out_dir, q, fps, "depth.mkv", "gray16le", "ffv1",
                        ["-level", "3"])


def _ffv1_read(out_dir, meta):
    return _video_read(out_dir, meta, "depth.mkv", "gray16le")


def _hevc12_write(out_dir, q, fps, opts):
    # Top 12 bits only (lossy: drop low 4). x265 params from opts (default
    # lossless ⇒ the only error is the 4-bit truncation).
    q12 = (q >> 4).astype("<u2")           # 0..4095 packed in 16-bit gray12le
    x265 = opts.get("x265_params", "lossless=1")
    return _video_write(out_dir, q12, fps, "depth.mkv", "gray12le", "libx265",
                        ["-x265-params", x265])


def _hevc12_read(out_dir, meta):
    q12 = _video_read(out_dir, meta, "depth.mkv", "gray12le")
    return (q12.astype(np.uint16) << 4)    # back to the uint16 scale (low 4 bits = 0)


BACKENDS = {
    "npz":    (_npz_write, _npz_read, True),    # (write, read, lossless?)
    "ffv1":   (_ffv1_write, _ffv1_read, True),
    "hevc12": (_hevc12_write, _hevc12_read, False),
}


# --------------------------------------------------------------------------- #
# The factory                                                                  #
# --------------------------------------------------------------------------- #
class DepthMapFactory:
    """Encode/decode a clip's depth stack via a chosen storage backend.

    factory = DepthMapFactory(backend="ffv1", fps=30)
    meta = factory.encode(clip_dir, depth_float_stack)   # writes files + sidecar
    q   = DepthMapFactory.decode(clip_dir)               # (T,H,W) uint16, any backend
    m   = DepthMapFactory.decode_meters(clip_dir)        # (T,H,W) float32 metres
    """

    def __init__(self, backend="npz", fps=30.0, **codec_opts):
        if backend not in BACKENDS:
            raise ValueError(f"unknown depth backend {backend!r}; have {sorted(BACKENDS)}")
        self.backend = backend
        self.fps = fps
        self.codec_opts = codec_opts

    @property
    def lossless(self):
        return BACKENDS[self.backend][2]

    def encode(self, out_dir, depth_float_stack):
        """Quantize float metres → uint16, store via the backend, write the sidecar.

        Returns the metadata dict (also persisted to depth_meta.json)."""
        os.makedirs(out_dir, exist_ok=True)
        q, d_min, d_max = quantize_depth_uint16(np.asarray(depth_float_stack))
        write = BACKENDS[self.backend][0]
        opts = {"d_min": d_min, "d_max": d_max, **self.codec_opts}
        files = write(out_dir, q, self.fps, opts)
        meta = {
            "backend": self.backend, "lossless": self.lossless,
            "d_min": d_min, "d_max": d_max, "shape": list(q.shape),
            "fps": float(self.fps), "codec_opts": self.codec_opts, "files": files,
        }
        with open(os.path.join(out_dir, SIDECAR), "w") as f:
            json.dump(meta, f, indent=2)
        return meta

    @staticmethod
    def read_meta(out_dir):
        with open(os.path.join(out_dir, SIDECAR)) as f:
            return json.load(f)

    @staticmethod
    def decode(out_dir):
        """Self-describing uint16 read: dispatches on the sidecar's backend.

        Falls back to the legacy `depth.npz` if no sidecar exists (old clips)."""
        path = os.path.join(out_dir, SIDECAR)
        if not os.path.exists(path):
            with np.load(os.path.join(out_dir, "depth.npz")) as d:   # legacy
                return d["depth"]
        meta = DepthMapFactory.read_meta(out_dir)
        return BACKENDS[meta["backend"]][1](out_dir, meta)

    @staticmethod
    def decode_meters(out_dir):
        """uint16 → float32 metres, using the sidecar's (or legacy npz's) range."""
        q = DepthMapFactory.decode(out_dir)
        path = os.path.join(out_dir, SIDECAR)
        if os.path.exists(path):
            m = DepthMapFactory.read_meta(out_dir)
            return dequantize_depth(q, m["d_min"], m["d_max"])
        with np.load(os.path.join(out_dir, "depth.npz")) as d:       # legacy range
            return dequantize_depth(q, float(d["depth_min"]), float(d["depth_max"]))
