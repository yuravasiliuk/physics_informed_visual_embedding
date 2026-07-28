"""Verify uint16 depth -> 2x uint8 byte-planes -> lossless 8-bit codec -> uint16.

Tests the "byte-split into an 8-bit video" idea as an alternative to native
gray16 (depth_map_factory.py `ffv1`). A uint16 is exactly two bytes, so splitting
into a high/low uint8 plane and recombining is bit-exact *by construction*; the
only question is whether an 8-bit VIDEO CODEC preserves those planes. It does iff
the codec is LOSSLESS:

  * FFV1 gray8 (lossless)      -> bit-exact (the point of this script);
  * mp4v / H.264-8bit (lossy)  -> CATASTROPHIC: a 1-bit error in the HIGH plane is
                                  a 256-code (≈ whole-scene) depth jump, not noise.

Packing: the two planes are stacked vertically into one (T, 2H, W) gray8 video
(hi on top, lo on bottom) — a single luma plane, so no colorspace / chroma
subsampling can touch the bytes. We compare against native gray16 FFV1 for size.

Runs on synthetic depth, or a real clip dir via DepthMapFactory.decode (no Isaac):

    python src/data_gen/verify_byte_split_depth.py
    python src/data_gen/verify_byte_split_depth.py --clip ~/data/sim_physics/clip_00012
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np


def _ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise SystemExit("ffmpeg not found on PATH")
    return exe


# ---- the bit ops (endian-independent) -------------------------------------- #
def split_hi_lo(q):
    """uint16 (...) -> (hi_uint8, lo_uint8). Pure reinterpretation, no rounding."""
    q = np.asarray(q, dtype=np.uint16)
    return (q >> 8).astype(np.uint8), (q & 0xFF).astype(np.uint8)


def merge_hi_lo(hi, lo):
    """(hi_uint8, lo_uint8) -> uint16. Exact inverse of split_hi_lo.

    `+` is exact (identical to bitwise `|`) because the operands are disjoint:
    `hi<<8` has zeros in its low 8 bits and `lo` occupies only those bits. Keep the
    parens — `+` binds TIGHTER than `<<`, so `hi<<8 + lo` would wrongly mean
    `hi<<(8+lo)`.
    """
    return (hi.astype(np.uint16) << 8) + lo.astype(np.uint16)


# ---- video round-trip of the stacked byte-planes --------------------------- #
def _encode_gray8(stack8, path, codec, codec_args, fps=30):
    """Pipe a (T, Hh, W) uint8 stack to ffmpeg as raw gray8, encode to `path`.

    NOTE: `path` must have a CONTAINER extension (.mkv/.nut/.avi) — FFV1 is a
    codec, not a container. A bad extension makes ffmpeg exit before reading
    stdin, which surfaces here as a BrokenPipeError; we catch that and raise
    ffmpeg's actual stderr instead so the cause is obvious.
    """
    T, Hh, W = stack8.shape
    proc = subprocess.Popen(
        [_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo",
         "-pix_fmt", "gray", "-s", f"{W}x{Hh}", "-r", str(fps), "-i", "-",
         "-c:v", codec, *codec_args, path],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        proc.stdin.write(np.ascontiguousarray(stack8, np.uint8).tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass                                   # ffmpeg already died; report below
    ret = proc.wait()
    err = proc.stderr.read().decode(errors="replace").strip()
    if ret != 0:
        raise RuntimeError(f"ffmpeg encode failed ({codec}) -> {path!r}:\n{err}")


def _decode_gray8(path, shape):
    T, Hh, W = shape
    out = subprocess.run(
        [_ffmpeg(), "-loglevel", "error", "-i", path, "-f", "rawvideo",
         "-pix_fmt", "gray", "-"], capture_output=True)
    if out.returncode != 0:
        raise RuntimeError(f"decode failed: {out.stderr.decode()[:200]}")
    return np.frombuffer(out.stdout, np.uint8).reshape(T, Hh, W)


def roundtrip_byte_split(q16, codec, codec_args, workdir):
    """q16 (T,H,W) uint16 -> hi/lo planes stacked (T,2H,W) gray8 -> codec -> q16."""
    T, H, W = q16.shape
    hi, lo = split_hi_lo(q16)
    stack8 = np.concatenate([hi, lo], axis=1)          # (T, 2H, W): hi rows, lo rows
    path = os.path.join(workdir, f"split_{codec}.mkv" if "ffv1" in codec else f"split_{codec}.mp4")
    _encode_gray8(stack8, path, codec, codec_args)
    back8 = _decode_gray8(path, stack8.shape)
    hi_b, lo_b = back8[:, :H], back8[:, H:]
    return merge_hi_lo(hi_b, lo_b), os.path.getsize(path)


def roundtrip_native_gray16(q16, workdir):
    """Reference: store q16 directly as native gray16le FFV1 (the factory's way)."""
    T, H, W = q16.shape
    path = os.path.join(workdir, "native_gray16.mkv")
    proc = subprocess.Popen(
        [_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo",
         "-pix_fmt", "gray16le", "-s", f"{W}x{H}", "-r", "30", "-i", "-",
         "-c:v", "ffv1", "-level", "3", "-pix_fmt", "gray16le", path],
        stdin=subprocess.PIPE)
    proc.stdin.write(q16.astype("<u2").tobytes()); proc.stdin.close(); proc.wait()
    out = subprocess.run(
        [_ffmpeg(), "-loglevel", "error", "-i", path, "-f", "rawvideo",
         "-pix_fmt", "gray16le", "-"], capture_output=True).stdout
    return np.frombuffer(out, "<u2").reshape(T, H, W), os.path.getsize(path)


def _synthetic_uint16(T=64, H=256, W=256):
    """Smooth depth-like field + moving 'objects', spanning the full uint16 range."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    frames = []
    for t in range(T):
        f = 0.5 + 0.3 * np.sin((xx + yy + t * 4) / 40.0) + 0.1 * (xx / W)
        f[60:90, 60:min(W, 60 + t)] = 0.05            # a near 'object' that grows
        f = (f - f.min()) / (f.max() - f.min())
        frames.append(np.round(f * 65535).astype(np.uint16))
    return np.stack(frames)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", default=None,
                   help="clip dir to load real depth from (via DepthMapFactory); "
                        "default = synthetic")
    p.add_argument("--frames", type=int, default=64)
    p.add_argument("--size", type=int, default=256)
    args = p.parse_args()

    if args.clip:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from depth_map_factory import DepthMapFactory
        q16 = np.asarray(DepthMapFactory.decode(os.path.expanduser(args.clip)), np.uint16)
        print(f"loaded real depth {q16.shape} from {args.clip}")
    else:
        q16 = _synthetic_uint16(args.frames, args.size, args.size)
        print(f"synthetic depth {q16.shape}, range [{q16.min()}, {q16.max()}]")

    # 0) pure numpy split/merge — no codec, proves the bit ops alone.
    hi, lo = split_hi_lo(q16)
    exact_np = np.array_equal(merge_hi_lo(hi, lo), q16)
    print(f"\n[0] numpy split/merge (no codec): bit-exact = {exact_np}")

    work = tempfile.mkdtemp()
    raw_mb = q16.nbytes / 1e6

    # 1) byte-split through a LOSSLESS 8-bit codec (FFV1 gray8) — the real test.
    q_ffv1, sz_ffv1 = roundtrip_byte_split(q16, "ffv1", ["-level", "3"], work)
    err_ffv1 = int(np.abs(q_ffv1.astype(int) - q16.astype(int)).max())
    print(f"[1] byte-split -> FFV1 gray8 (LOSSLESS): bit-exact = {err_ffv1 == 0} "
          f"(max err {err_ffv1})  | {sz_ffv1/1e6:.2f} MB")

    # 2) byte-split through a LOSSY codec (mp4v) — expected to be destroyed.
    try:
        q_lossy, sz_lossy = roundtrip_byte_split(q16, "mpeg4", ["-q:v", "3"], work)
        err_lossy = int(np.abs(q_lossy.astype(int) - q16.astype(int)).max())
        print(f"[2] byte-split -> mp4v   (LOSSY):    bit-exact = {err_lossy == 0} "
              f"(max err {err_lossy} codes ≈ {err_lossy/65535*100:.1f}% of full range) "
              f"| {sz_lossy/1e6:.2f} MB  <- why lossy 8-bit cannot carry bytes")
    except Exception as e:
        print(f"[2] lossy path errored: {e}")

    # 3) reference: native gray16 FFV1 (depth_map_factory `ffv1`).
    q_nat, sz_nat = roundtrip_native_gray16(q16, work)
    err_nat = int(np.abs(q_nat.astype(int) - q16.astype(int)).max())
    print(f"[3] native gray16 FFV1 (reference):  bit-exact = {err_nat == 0} "
          f"(max err {err_nat})  | {sz_nat/1e6:.2f} MB")

    print(f"\nraw uint16 = {raw_mb:.2f} MB")
    ok = exact_np and err_ffv1 == 0 and err_nat == 0
    print("\nRESULT:", "PASS — lossless byte-split round-trips bit-exact" if ok
          else "FAIL — see above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
